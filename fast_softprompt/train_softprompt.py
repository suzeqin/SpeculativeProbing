#!/usr/bin/env python3
"""
Fast soft-prompt training using precomputed hidden states.
Only runs the MTP layer forward/backward — no base model needed.

Usage (qwen35_latest env):
  python fast_softprompt/train_softprompt.py \
    --cache fast_softprompt/cache_wildchat_9b.pt \
    --mtp_dir models/Qwen3.5-9B-mtp \
    --output_dir wildchat_output_softprompt_9b \
    --lr 1e-3 --epochs 10
"""

from __future__ import annotations

import argparse
import math
import os
import json
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "longcite"))

from qwen35_mtp import Qwen3_5MTP

RANDOM_SEED = 42


def _load_embed_weights(model_dir):
    """Load embed_tokens and lm_head weights directly from safetensors without loading full model.
    This avoids OOM for large models (e.g. 27B) where loading the full model + MTP doesn't fit."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        # Sharded model — find which shard has the weights
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        # Find embed_tokens key (may be "model.embed_tokens.weight" or "model.language_model.embed_tokens.weight")
        embed_key = None
        lm_head_key = None
        for k in weight_map:
            if "embed_tokens.weight" in k:
                embed_key = k
            if "lm_head.weight" in k:
                lm_head_key = k

        from safetensors.torch import load_file
        # Load only the needed shards
        shards_needed = set()
        if embed_key:
            shards_needed.add(weight_map[embed_key])
        if lm_head_key:
            shards_needed.add(weight_map[lm_head_key])

        weights = {}
        for shard in shards_needed:
            shard_path = os.path.join(model_dir, shard)
            shard_weights = load_file(shard_path)
            if embed_key and embed_key in shard_weights:
                weights["embed"] = shard_weights[embed_key]
            if lm_head_key and lm_head_key in shard_weights:
                weights["lm_head"] = shard_weights[lm_head_key]

        # If no separate lm_head (tied embeddings), reuse embed weights
        return weights["embed"], weights.get("lm_head", weights["embed"])
    else:
        # Single file or use transformers as fallback
        single_path = os.path.join(model_dir, "model.safetensors")
        if os.path.exists(single_path):
            from safetensors.torch import load_file
            all_weights = load_file(single_path)
            embed_key = [k for k in all_weights if "embed_tokens.weight" in k][0]
            lm_head_key = [k for k in all_weights if "lm_head.weight" in k][0]
            return all_weights[embed_key], all_weights[lm_head_key]
        else:
            # Fallback: load full model (for small models or non-safetensors)
            from transformers import AutoModelForCausalLM
            print("  Warning: falling back to full model load for embeddings")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
            )
            embed = base_model.model.embed_tokens.weight.detach().clone()
            lm_head = base_model.lm_head.weight.detach().clone()
            del base_model
            torch.cuda.empty_cache()
            return embed, lm_head


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def _resolve_label_token_id(tokenizer, label_token: str) -> int:
    token_ids = tokenizer.encode(label_token, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Label token '{label_token}' produced no token ids.")
    return token_ids[-1]


class CachedSoftPromptClassifier(nn.Module):
    """
    Soft-prompt classifier that works with precomputed hidden states.
    Only the MTP layer + soft prompt are used during forward pass.
    """

    def __init__(self, mtp_model, embed_weights, pos_token_id, neg_token_id,
                 soft_prompt_length=1, device="cuda", classifier_head="lm_head",
                 dropout=0.0, feature_noise=0.0):
        super().__init__()
        self.mtp_model = mtp_model
        self.embed_weights = embed_weights  # [vocab_size, hidden_dim], frozen
        self.pos_token_id = pos_token_id
        self.neg_token_id = neg_token_id
        self.device = device
        self.classifier_head_type = classifier_head
        self.dropout = nn.Dropout(dropout).to(device)
        self.feature_noise = feature_noise

        self.mtp_model.eval()
        for p in self.mtp_model.parameters():
            p.requires_grad = False

        pos_embed = embed_weights[pos_token_id]
        self.soft_prompt = nn.Parameter(
            pos_embed.detach()
            .to(device=device, dtype=torch.float32)
            .view(1, 1, -1)
            .repeat(1, soft_prompt_length, 1),
            requires_grad=True,
        )

        hidden_dim = mtp_model.lm_head.weight.size(1)
        if classifier_head == "linear":
            # Sigmoid head: single logit -> BCEWithLogitsLoss
            self.head = nn.Linear(hidden_dim, 1).to(device=device, dtype=torch.float32)
        elif classifier_head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 2),
            ).to(device=device, dtype=torch.float32)
        else:
            self.head = None  # will use mtp_model.lm_head + token-id readout

    def forward(self, hidden_states, token_ids, attention_mask, labels=None):
        """
        Args:
            hidden_states: [B, seq_len, hidden_dim] - precomputed base model hidden states
            token_ids: [B, seq_len] - original token ids (for computing input embeds)
            attention_mask: [B, seq_len]
            labels: [B] - 0 or 1
        """
        k = self.soft_prompt.size(1)  # soft_prompt_length

        # Compute input embeds from token ids using frozen embed weights
        inputs_embeds = self.embed_weights[token_ids.to(self.embed_weights.device)].to(self.device).clone()
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.training and self.feature_noise > 0:
            hidden_states = hidden_states + self.feature_noise * torch.randn_like(hidden_states)

        # For k > 1, extend tensors to accommodate extra soft prompt tokens
        if k > 1:
            B, seq_len, hdim = inputs_embeds.shape
            ext = k - 1
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, ext, hdim, dtype=inputs_embeds.dtype, device=self.device)], dim=1)
            hidden_states = torch.cat([hidden_states, hidden_states[:, -1:, :].expand(-1, ext, -1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros(B, ext, dtype=attention_mask.dtype, device=self.device)], dim=1)

        # Append soft prompt: shift embeds left by 1, append k soft prompt tokens
        # For k=1: shifted = [embed[1], ..., embed[n-1], sp[0]] — same length
        # For k>1: shifted = [embed[1], ..., embed[n-1], sp[0], ..., sp[k-1]] — length n+(k-1)
        batch_size = token_ids.size(0)
        sp = self.soft_prompt[0, :k, :].to(inputs_embeds.dtype)  # [k, hidden_dim]
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 0:
                continue
            new_valid_len = valid_len + (k - 1)
            # Shift left by 1, append k soft prompt tokens
            shifted = torch.cat(
                [inputs_embeds[b, 1:valid_len, :], sp], dim=0
            )  # [(valid_len - 1) + k] = [new_valid_len] tokens
            inputs_embeds[b, :new_valid_len, :] = shifted
            # Mark extra positions as valid
            attention_mask[b, valid_len:new_valid_len] = 1
            # Repeat last hidden state for extended positions
            if k > 1:
                hidden_states[b, valid_len:new_valid_len, :] = hidden_states[b, valid_len - 1:valid_len, :].expand(k - 1, -1)

        # MTP forward - only apply lm_head at the last valid position to save compute
        mtp_dtype = self.mtp_model.lm_head.weight.dtype
        out = self.mtp_model.model(
            hidden_states=hidden_states.to(dtype=mtp_dtype),
            inputs_embeds=inputs_embeds.to(dtype=mtp_dtype),
        )
        h_all = out[0]  # [B, seq_len, hidden_dim]

        last_valid_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(h_all.size(0), device=h_all.device)
        h_at_soft = h_all[batch_idx, last_valid_idx, :]  # [B, hidden_dim]
        h_at_soft = self.dropout(h_at_soft)

        if self.head is not None:
            if self.classifier_head_type == "linear":
                # Sigmoid head: single logit + BCEWithLogitsLoss.
                # Return [B, 2] as [-z, +z] so downstream argmax == (sigmoid(z) > 0.5).
                z = self.head(h_at_soft.float()).squeeze(-1)  # [B]
                logits = torch.stack([-z, z], dim=-1)
                loss = None
                if labels is not None:
                    loss = nn.BCEWithLogitsLoss()(z, labels.to(z.device, dtype=z.dtype))
                return logits, loss
            logits = self.head(h_at_soft.float())
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels.to(logits.device))
            return logits, loss

        # Reuse frozen lm_head, read yes/no token logits
        logits_at_soft = self.mtp_model.lm_head(h_at_soft).float()  # [B, vocab]
        pos_logits = logits_at_soft[:, self.pos_token_id]
        neg_logits = logits_at_soft[:, self.neg_token_id]
        logits = torch.stack([neg_logits, pos_logits], dim=-1)

        loss = None
        if labels is not None:
            token_labels = torch.where(
                labels.to(logits_at_soft.device) == 1,
                self.pos_token_id,
                self.neg_token_id,
            )
            loss = nn.CrossEntropyLoss()(logits_at_soft, token_labels)
        return logits, loss


class RecursiveSoftPromptClassifier(nn.Module):
    """
    Recursive soft-prompt classifier that mirrors MTP's autoregressive behavior.

    Instead of appending k tokens to the sequence in one pass, we run k sequential
    MTP forward passes with KV cache. Each step feeds the previous output hidden
    state back as input (like speculative decoding), with a learned soft-prompt
    embedding at each step. The KV cache lets later steps attend to all prefix positions.

    Step 0: Run MTP on full sequence with sp[0] appended → get hidden state h_0, KV cache
    Step 1: Run MTP with hidden_states=h_0, inputs_embeds=sp[1], KV cache → get h_1
    ...
    Step k-1: Read logits from final step
    """

    def __init__(self, mtp_model, embed_weights, pos_token_id, neg_token_id,
                 num_recursive_steps=1, device="cuda", classifier_head="lm_head",
                 dropout=0.0, feature_noise=0.0,
                 head_init="random", multi_step_loss=False):
        super().__init__()
        self.mtp_model = mtp_model
        self.embed_weights = embed_weights
        self.pos_token_id = pos_token_id
        self.neg_token_id = neg_token_id
        self.device = device
        self.num_recursive_steps = num_recursive_steps
        self.classifier_head_type = classifier_head
        self.dropout = nn.Dropout(dropout).to(device)
        self.feature_noise = feature_noise
        self.multi_step_loss = multi_step_loss

        self.mtp_model.eval()
        for p in self.mtp_model.parameters():
            p.requires_grad = False

        pos_embed = embed_weights[pos_token_id].detach().to(device=device, dtype=torch.float32)

        # One soft prompt embedding per recursive step
        self.soft_prompts = nn.ParameterList([
            nn.Parameter(pos_embed.clone().view(1, 1, -1), requires_grad=True)
            for _ in range(num_recursive_steps)
        ])

        hidden_dim = mtp_model.lm_head.weight.size(1)
        if classifier_head == "linear":
            # Sigmoid head: single logit -> BCEWithLogitsLoss
            self.head = nn.Linear(hidden_dim, 1).to(device=device, dtype=torch.float32)
            if head_init == "yes_minus_no":
                # Anchor head at the lm_head's yes-minus-no output direction.
                with torch.no_grad():
                    lm_w = mtp_model.lm_head.weight  # [vocab, hidden]
                    yn = (lm_w[pos_token_id].float() - lm_w[neg_token_id].float())
                    # Scale down so initial sigmoid isn't saturated.
                    scale = 1.0 / (hidden_dim ** 0.5)
                    self.head.weight.copy_(yn.view(1, -1).to(self.head.weight.device) * scale)
                    self.head.bias.zero_()
        elif classifier_head == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 100),
                nn.ReLU(),
                nn.Linear(100, 2),
            ).to(device=device, dtype=torch.float32)
        else:
            self.head = None

    def freeze_head(self):
        if self.head is not None:
            for p in self.head.parameters():
                p.requires_grad = False

    def forward(self, hidden_states, token_ids, attention_mask, labels=None):
        from transformers import DynamicCache

        inputs_embeds = self.embed_weights[token_ids.to(self.embed_weights.device)].to(self.device).clone()
        hidden_states = hidden_states.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.training and self.feature_noise > 0:
            hidden_states = hidden_states + self.feature_noise * torch.randn_like(hidden_states)

        batch_size = token_ids.size(0)
        mtp_dtype = self.mtp_model.lm_head.weight.dtype
        sp0 = self.soft_prompts[0][0, 0, :].to(inputs_embeds.dtype)

        # Shift embeds left by 1, append sp[0] at end of valid region
        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 0:
                continue
            inputs_embeds[b, :valid_len - 1, :] = inputs_embeds[b, 1:valid_len, :].clone()
            inputs_embeds[b, valid_len - 1, :] = sp0

        last_valid_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(batch_size, device=self.device)
        seq_len = inputs_embeds.size(1)

        # Collect the hidden state after each recursive step. Step 0 is the vector
        # at the final valid position after the first full-sequence pass; step s (s>=1)
        # is the 1-token output of the s-th recursive call.
        step_hiddens = []  # list of [B, d]

        if self.num_recursive_steps == 1:
            # Single step: run inner model only, then apply head at last position
            out = self.mtp_model.model(
                hidden_states=hidden_states.to(dtype=mtp_dtype),
                inputs_embeds=inputs_embeds.to(dtype=mtp_dtype),
            )
            h_all = out[0]  # [B, seq_len, hidden_dim]
            step_hiddens.append(h_all[batch_idx, last_valid_idx, :])
        else:
            # Step 0: full sequence with KV cache
            kv_cache = DynamicCache()
            out = self.mtp_model.model(
                hidden_states=hidden_states.to(dtype=mtp_dtype),
                inputs_embeds=inputs_embeds.to(dtype=mtp_dtype),
                use_cache=True,
                past_key_value=kv_cache,
            )
            h_all = out[0]  # [B, seq_len, hidden_dim] (already normed)
            kv_cache = out[1]

            h = h_all[batch_idx, last_valid_idx, :].unsqueeze(1)  # [B, 1, d]
            step_hiddens.append(h[:, 0, :])

            # Recursive steps with KV cache
            for step in range(1, self.num_recursive_steps):
                sp_step = self.soft_prompts[step][0, :, :].to(dtype=mtp_dtype)
                sp_step = sp_step.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, d]

                pos_idx = seq_len + step - 1
                pos3 = torch.tensor([[[pos_idx]]]).expand(3, batch_size, 1).to(self.device)
                cache_pos = torch.tensor([pos_idx], device=self.device)

                out = self.mtp_model.model(
                    hidden_states=h.to(dtype=mtp_dtype),
                    inputs_embeds=sp_step,
                    use_cache=True,
                    past_key_value=kv_cache,
                    position_ids=pos3,
                    cache_position=cache_pos,
                )
                h = out[0]  # [B, 1, d]
                kv_cache = out[1]
                step_hiddens.append(h[:, 0, :])

        # Run the head at every step.
        def head_forward(h_step):
            h_step_d = self.dropout(h_step)
            if self.head is not None:
                if self.classifier_head_type == "linear":
                    z = self.head(h_step_d.float()).squeeze(-1)
                    logits_s = torch.stack([-z, z], dim=-1)
                    loss_s = None
                    if labels is not None:
                        loss_s = nn.BCEWithLogitsLoss()(z, labels.to(z.device, dtype=z.dtype))
                    return logits_s, loss_s
                logits_s = self.head(h_step_d.float())
                loss_s = None
                if labels is not None:
                    loss_s = nn.CrossEntropyLoss()(logits_s, labels.to(logits_s.device))
                return logits_s, loss_s
            # lm_head mode
            logits_vocab = self.mtp_model.lm_head(h_step_d).float()
            pos_logits = logits_vocab[:, self.pos_token_id]
            neg_logits = logits_vocab[:, self.neg_token_id]
            logits_s = torch.stack([neg_logits, pos_logits], dim=-1)
            loss_s = None
            if labels is not None:
                token_labels = torch.where(
                    labels.to(logits_vocab.device) == 1,
                    self.pos_token_id,
                    self.neg_token_id,
                )
                loss_s = nn.CrossEntropyLoss()(logits_vocab, token_labels)
            return logits_s, loss_s

        per_step = [head_forward(h_s) for h_s in step_hiddens]
        # Expose per-step logits via an attribute so callers that want step-wise
        # accuracy can grab them after the forward pass; training loss still comes
        # via the usual (logits, loss) return.
        self.last_per_step_logits = torch.stack([pl for pl, _ in per_step], dim=1).detach()

        if self.multi_step_loss and labels is not None:
            losses = [ls for _, ls in per_step if ls is not None]
            loss = torch.stack(losses).mean() if losses else None
            return per_step[-1][0], loss

        # Default: only last-step loss / logits (preserves original behavior).
        return per_step[-1]


def pad_and_collate(batch_hidden, batch_ids, batch_labels):
    """Pad variable-length sequences into tensors."""
    max_len = max(h.size(0) for h in batch_hidden)
    hidden_dim = batch_hidden[0].size(1)
    B = len(batch_hidden)

    hidden_padded = torch.zeros(B, max_len, hidden_dim, dtype=batch_hidden[0].dtype)
    ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, (h, ids) in enumerate(zip(batch_hidden, batch_ids)):
        seq_len = h.size(0)
        hidden_padded[i, :seq_len] = h
        ids_padded[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        mask[i, :seq_len] = 1

    labels = torch.tensor(batch_labels, dtype=torch.long)
    return hidden_padded, ids_padded, mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Path to precomputed cache .pt file")
    parser.add_argument("--mmap_cache", action="store_true",
                        help="If set and the cache is mmap_v1 format, memory-map the hidden "
                             "states instead of fully loading them (OOM-safe for huge caches).")
    parser.add_argument("--mtp_dir", required=True, help="Path to MTP checkpoint directory")
    parser.add_argument("--base_model", default=None, help="Base model path (for config + tokenizer). Auto-detected from cache if not set.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=0.0,
                        help="If >0, apply cosine LR schedule from --lr down to --min_lr over --epochs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--soft_prompt_length", type=int, default=1)
    parser.add_argument("--recursive", action="store_true",
                        help="Use recursive MTP steps instead of appending tokens to sequence")
    parser.add_argument("--positive_token", default="yes")
    parser.add_argument("--negative_token", default="no")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate applied to the pooled hidden state before the classifier head.")
    parser.add_argument("--feature_noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to cached hidden_states (training only, disabled in eval).")
    parser.add_argument("--target_eval", type=int, default=200,
                        help="Target combined val+test size; borrow from train tail if cache has fewer.")
    parser.add_argument("--val_fraction", type=float, default=0.5,
                        help="Fraction of the eval pool used as validation; remainder is test.")
    parser.add_argument("--save_epoch_ckpts", action="store_true",
                        help="Save trainable-weights checkpoint after each epoch (small: only soft-prompt and head).")
    parser.add_argument("--head_init", choices=["random", "yes_minus_no"], default="random",
                        help="Classifier-head initialization. yes_minus_no uses lm_head[yes]-lm_head[no], scaled.")
    parser.add_argument("--freeze_head_after_epoch", type=int, default=0,
                        help="If >0, freeze head at the END of this epoch (1-indexed); soft prompt continues training.")
    parser.add_argument("--load_soft_prompt", type=str, default=None,
                        help="Path to a saved checkpoint (from --save_epoch_ckpts). Loads soft_prompt* weights and freezes them. Useful for 'stage 2' training where only the head is trained.")
    parser.add_argument("--multi_step_loss", action="store_true",
                        help="Compute training loss at every recursive step (mean) and "
                             "report per-step val/test accuracy. Recursive mode only.")
    parser.add_argument("--classifier_head", default="lm_head",
                        choices=["lm_head", "linear", "mlp"],
                        help="lm_head=reuse frozen lm_head + yes/no token logits; "
                             "linear=trainable Linear(hidden,2); mlp=Linear(hidden,100)+ReLU+Linear(100,2)")
    args = parser.parse_args()

    set_seed()

    print(f"Loading cache from {args.cache} (mmap={args.mmap_cache})...")
    from cache_io import load_cache
    cache = load_cache(args.cache, mmap=args.mmap_cache)
    all_ids = cache["token_ids"]
    all_labels = cache["labels"]
    all_hidden = cache["hidden_states"]
    base_model_name = args.base_model or cache["model"]
    task = cache["task"]

    from eval_split import make_val_test_split
    split = make_val_test_split(
        all_hidden, all_ids, all_labels,
        num_train_cached=cache["num_train"],
        num_test_cached=cache["num_test"],
        target_eval=args.target_eval,
        val_fraction=args.val_fraction,
    )
    num_train = split["num_train_used"]
    num_val = split["num_val"]
    num_test = split["num_test"]
    train_hidden = split["train_hidden"]
    train_ids = split["train_ids"]
    train_labels = split["train_labels"]
    val_hidden = split["val_hidden"]
    val_ids = split["val_ids"]
    val_labels = split["val_labels"]
    test_hidden = split["test_hidden"]
    test_ids = split["test_ids"]
    test_labels = split["test_labels"]

    print(f"  Task: {task}, Model: {base_model_name}")
    print(f"  Train: {num_train}, Val: {num_val}, Test: {num_test} "
          f"(borrowed {split['borrowed_from_train']} from train -> eval)")

    # Load MTP model
    print(f"Loading MTP model from {args.mtp_dir}...")
    # Resolve base model local path
    base_model_local = base_model_name
    if not os.path.isdir(base_model_local):
        from huggingface_hub import snapshot_download
        base_model_local = snapshot_download(base_model_name, local_files_only=True)

    mtp_model = Qwen3_5MTP.from_mtp_checkpoint(args.mtp_dir, base_model_local)
    device = "cuda"
    mtp_dtype = torch.bfloat16
    mtp_model = mtp_model.to(device=device, dtype=mtp_dtype).eval()

    # Copy embed weights from base model into MTP
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_local, trust_remote_code=True)

    # Extract embedding weights WITHOUT loading full model (critical for large models like 27B)
    print("Loading base model embeddings...")
    embed_weight, lm_head_weight = _load_embed_weights(base_model_local)
    embed_weights = embed_weight.to(device=device, dtype=torch.bfloat16)

    # Copy embed/lm_head into MTP
    with torch.no_grad():
        mtp_model.model.embed_tokens.weight.copy_(
            embed_weight.to(
                device=mtp_model.model.embed_tokens.weight.device,
                dtype=mtp_model.model.embed_tokens.weight.dtype,
            )
        )
        mtp_model.lm_head.weight.copy_(
            lm_head_weight.to(
                device=mtp_model.lm_head.weight.device,
                dtype=mtp_model.lm_head.weight.dtype,
            )
        )

    del embed_weight, lm_head_weight
    torch.cuda.empty_cache()
    print(f"  MTP loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pos_token_id = _resolve_label_token_id(tokenizer, args.positive_token)
    neg_token_id = _resolve_label_token_id(tokenizer, args.negative_token)

    if args.recursive:
        classifier = RecursiveSoftPromptClassifier(
            mtp_model=mtp_model,
            embed_weights=embed_weights,
            pos_token_id=pos_token_id,
            neg_token_id=neg_token_id,
            num_recursive_steps=args.soft_prompt_length,
            device=device,
            classifier_head=args.classifier_head,
            dropout=args.dropout,
            feature_noise=args.feature_noise,
            head_init=args.head_init,
            multi_step_loss=args.multi_step_loss,
        )
        print(f"  Using recursive MTP with {args.soft_prompt_length} steps, head={args.classifier_head}, "
              f"dropout={args.dropout}, feature_noise={args.feature_noise}, head_init={args.head_init}, "
              f"freeze_head_after_epoch={args.freeze_head_after_epoch}, "
              f"multi_step_loss={args.multi_step_loss}")
    else:
        classifier = CachedSoftPromptClassifier(
            mtp_model=mtp_model,
            embed_weights=embed_weights,
            pos_token_id=pos_token_id,
            neg_token_id=neg_token_id,
            soft_prompt_length=args.soft_prompt_length,
            device=device,
            classifier_head=args.classifier_head,
            dropout=args.dropout,
            feature_noise=args.feature_noise,
        )
        print(f"  Using MTP with sp_len={args.soft_prompt_length}, head={args.classifier_head}")

    target_names = {
        "wildchat": ["same_topic", "different_topic"],
        "verih": ["aligned", "conflict"],
        "longsafety": ["safe", "unsafe"],
        "pii": ["no_pii", "pii"],
        "adversarial_if": ["correct", "incorrect"],
        "nemotron_safety": ["safe", "unsafe"],
        "cot_repetitive": ["stay", "repetitive"],
        "cot_branched": ["stay", "branch"],  # new branched_thinking dataset
        "branched": ["stay", "branch"],
    }.get(task, ["class_0", "class_1"])

    # Optional stage-2 warm start: load soft prompt weights from a prior run and freeze them.
    if args.load_soft_prompt:
        print(f"Loading soft prompt weights from {args.load_soft_prompt} and freezing them.")
        ck = torch.load(args.load_soft_prompt, map_location="cpu", weights_only=False)
        state = ck.get("state_dict", ck)
        sp_state = {k: v for k, v in state.items() if k.startswith("soft_prompt")}
        missing, unexpected = classifier.load_state_dict(sp_state, strict=False)
        # We expect the only "missing" keys to be the head (or nothing if lm_head mode).
        # Unexpected keys would indicate a shape/naming mismatch.
        if unexpected:
            print(f"  WARNING unexpected keys: {unexpected}")
        for name, p in classifier.named_parameters():
            if name.startswith("soft_prompt"):
                p.requires_grad = False
        loaded_from_epoch = ck.get("epoch", "?")
        loaded_val = ck.get("val_acc", "?")
        print(f"  Loaded from epoch {loaded_from_epoch} (val_acc={loaded_val}); soft prompt frozen.")
        print(f"  Trainable params after freeze: {sum(p.numel() for p in classifier.parameters() if p.requires_grad):,}")

    # Training loop
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = None
    if args.min_lr > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        print(f"Cosine LR: {args.lr:.2e} -> {args.min_lr:.2e} over {args.epochs} epochs")

    best_eval_acc = -1.0
    best_epoch_metrics = None
    best_epoch = -1
    best_step = None  # only meaningful when --multi_step_loss
    history = []

    for epoch in range(args.epochs):
        classifier.train()
        # Shuffle train indices
        perm = list(range(num_train))
        random.shuffle(perm)

        running_loss, correct, seen = 0.0, 0, 0
        recent_correct = deque(maxlen=20)
        num_batches = math.ceil(num_train / args.batch_size)
        pbar = tqdm(range(0, num_train, args.batch_size), total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}")

        for s in pbar:
            e = min(s + args.batch_size, num_train)
            batch_idx = perm[s:e]
            batch_h = [train_hidden[i] for i in batch_idx]
            batch_i = [train_ids[i] for i in batch_idx]
            batch_l = [train_labels[i] for i in batch_idx]

            hidden_padded, ids_padded, mask, labels = pad_and_collate(batch_h, batch_i, batch_l)

            optimizer.zero_grad()
            logits, loss = classifier(hidden_padded, ids_padded, mask, labels)
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1).cpu()
                gold = labels.cpu()
                batch_matches = (preds == gold)
                correct += batch_matches.sum().item()
                seen += gold.numel()
                recent_correct.extend(batch_matches.tolist())
            rolling_acc = (sum(recent_correct) / len(recent_correct)) if recent_correct else 0.0
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{rolling_acc:.3f}", "lr": f"{cur_lr:.2e}"})

        # --- Validation eval (for model selection) ---
        def run_eval(hiddens, idss, lbls):
            preds, golds = [], []
            per_step_preds = None  # lazily init [S][N]
            classifier.eval()
            with torch.no_grad():
                for s in range(0, len(lbls), args.batch_size):
                    e = min(s + args.batch_size, len(lbls))
                    hp, ip, mk, lbl = pad_and_collate(hiddens[s:e], idss[s:e], lbls[s:e])
                    logits, _ = classifier(hp, ip, mk, None)
                    preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
                    golds.extend(lbl.tolist())
                    if args.multi_step_loss and hasattr(classifier, "last_per_step_logits"):
                        step_logits = classifier.last_per_step_logits  # [B, S, 2]
                        if per_step_preds is None:
                            per_step_preds = [[] for _ in range(step_logits.size(1))]
                        step_pred = torch.argmax(step_logits, dim=-1).cpu().tolist()
                        for si, row in enumerate(step_pred):
                            for s_idx, p_val in enumerate(row):
                                per_step_preds[s_idx].append(p_val)
            acc_v = accuracy_score(golds, preds)
            p_v, r_v, f_v, _ = precision_recall_fscore_support(golds, preds, average="binary", zero_division=0)
            per_step_acc = None
            if per_step_preds is not None:
                per_step_acc = [accuracy_score(golds, sp_preds) for sp_preds in per_step_preds]
            return preds, golds, acc_v, float(p_v), float(r_v), float(f_v), per_step_acc

        _, _, val_acc, val_pr, val_rc, val_f1, val_per_step_acc = run_eval(val_hidden, val_ids, val_labels)

        cur_lr = optimizer.param_groups[0]["lr"]
        # With multi-step loss we model-select on the best per-step val accuracy,
        # because the "last step" may not be the best readout position.
        if val_per_step_acc is not None:
            best_step_idx = int(np.argmax(val_per_step_acc))
            val_acc_select = float(val_per_step_acc[best_step_idx])
        else:
            best_step_idx = None
            val_acc_select = val_acc
        row = {"epoch": epoch + 1, "lr": cur_lr,
               "train_loss": running_loss / max(num_batches, 1),
               "train_acc": correct / max(seen, 1),
               "val_acc": val_acc,
               "val_per_step_acc": val_per_step_acc,
               "val_best_step": best_step_idx,
               "val_acc_select": val_acc_select}
        history.append(row)

        # Save per-epoch checkpoint: only the trainable parameters (soft prompt
        # vectors + optional head). The frozen MTP layer is not saved.
        if args.output_dir and args.save_epoch_ckpts:
            ckpt_dir = os.path.join(args.output_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            trainable_state = {
                k: v.detach().cpu().clone()
                for k, v in classifier.state_dict().items()
                if classifier.get_parameter(k).requires_grad if hasattr(classifier, "get_parameter") and k in dict(classifier.named_parameters())
            } if False else {
                # Simpler: save only soft_prompt* and head*; everything else is frozen.
                k: v.detach().cpu().clone()
                for k, v in classifier.state_dict().items()
                if k.startswith("soft_prompt") or k.startswith("head.") or k == "head.weight" or k == "head.bias"
            }
            torch.save({
                "epoch": epoch + 1, "val_acc": val_acc,
                "state_dict": trainable_state,
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

        marker = ""
        if val_acc_select > best_eval_acc:
            best_eval_acc = val_acc_select
            best_epoch = epoch + 1
            best_step = best_step_idx  # may be None when not using multi-step loss
            # Snapshot only the trainable weights (soft prompts + head) on CPU.
            # classifier.state_dict() also contains the frozen MTP head itself
            # (~5.5 GiB in bf16 for the 27B model). Cloning that on GPU is
            # wasteful, and the second best-epoch update OOMs because the
            # previous snapshot is still alive while the new one is built.
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in classifier.state_dict().items()
                if k.startswith("soft_prompt") or k.startswith("head.")
            }
            marker = " *best*"
        ps_str = ""
        if val_per_step_acc is not None:
            ps_str = " val_per_step=[" + ", ".join(f"{a:.3f}" for a in val_per_step_acc) + f"] best_step={best_step_idx}"
        print(f"Epoch {epoch+1}: lr={cur_lr:.2e}, loss={row['train_loss']:.4f}, "
              f"train_acc={row['train_acc']:.4f}, val_acc={val_acc:.4f}{ps_str}{marker}")

        if scheduler is not None:
            scheduler.step()

        # Optional: freeze the classifier head after epoch N. Rebuild optimizer so
        # it stops applying updates (and momentum) to the frozen head.
        if (args.freeze_head_after_epoch > 0 and
                epoch + 1 == args.freeze_head_after_epoch and
                hasattr(classifier, "freeze_head")):
            classifier.freeze_head()
            new_params = [p for p in classifier.parameters() if p.requires_grad]
            # Preserve current LR; keep cosine schedule by rebuilding on top of new optimizer.
            cur_lr_val = optimizer.param_groups[0]["lr"]
            optimizer = torch.optim.AdamW(new_params, lr=cur_lr_val, weight_decay=0.01)
            if scheduler is not None:
                remaining = args.epochs - (epoch + 1)
                if remaining > 0:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=remaining, eta_min=args.min_lr)
            print(f"  [Froze head after epoch {epoch+1}. "
                  f"Trainable params now: {sum(p.numel() for p in new_params):,}]")

    # --- Final test eval using val-selected best epoch ---
    if best_epoch > 0:
        # best_state holds only the trainable keys; the frozen MTP weights are
        # identical to the live model, so strict=False is expected here.
        classifier.load_state_dict(best_state, strict=False)
    test_preds, test_golds, test_acc, test_pr, test_rc, test_f1, test_per_step_acc = run_eval(
        test_hidden, test_ids, test_labels)
    test_report = classification_report(
        test_golds, test_preds, target_names=target_names,
        labels=list(range(len(target_names))), zero_division=0,
    )
    val_final_preds, val_final_golds, vA, vP, vR, vF, val_per_step_acc_final = run_eval(
        val_hidden, val_ids, val_labels)

    # When multi_step_loss is on, report headline numbers at the val-selected
    # best step — that is what "pick the best sp" means for this experiment.
    if best_step is not None and test_per_step_acc is not None:
        # Re-derive per-step precision/recall/f1 at the selected step on test.
        # We already have per-step preds inside run_eval; redo quickly here.
        def preds_at_step(hiddens, idss, lbls, step_idx):
            preds_ = []
            classifier.eval()
            with torch.no_grad():
                for s in range(0, len(lbls), args.batch_size):
                    e = min(s + args.batch_size, len(lbls))
                    hp, ip, mk, _ = pad_and_collate(hiddens[s:e], idss[s:e], lbls[s:e])
                    classifier(hp, ip, mk, None)
                    step_logits = classifier.last_per_step_logits[:, step_idx, :]
                    preds_.extend(torch.argmax(step_logits, dim=-1).cpu().tolist())
            return preds_
        test_preds_sel = preds_at_step(test_hidden, test_ids, test_labels, best_step)
        test_acc = accuracy_score(test_golds, test_preds_sel)
        test_pr, test_rc, test_f1, _ = precision_recall_fscore_support(
            test_golds, test_preds_sel, average="binary", zero_division=0)
        test_report = classification_report(
            test_golds, test_preds_sel, target_names=target_names,
            labels=list(range(len(target_names))), zero_division=0,
        )
        test_preds = test_preds_sel

    print(f"\nBest epoch (by val_acc): {best_epoch} (val_acc={best_eval_acc:.4f})"
          + (f", best_step={best_step}" if best_step is not None else ""))
    if test_per_step_acc is not None:
        print("Test per-step acc: [" + ", ".join(f"{a:.4f}" for a in test_per_step_acc) + "]")
    print(f"Test  | Accuracy: {test_acc:.4f} | Precision: {test_pr:.4f} | Recall: {test_rc:.4f} | F1: {float(test_f1):.4f}")
    print(test_report)

    best_epoch_metrics = {
        "accuracy": test_acc,
        "precision": float(test_pr),
        "recall": float(test_rc),
        "f1": float(test_f1),
        "per_step_acc": test_per_step_acc,
        "best_step": best_step,
    }
    val_metrics = {"accuracy": vA, "precision": vP, "recall": vR, "f1": vF,
                   "per_step_acc": val_per_step_acc_final}

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "config": {
                "task": task,
                "arch": "recursive_softprompt_cached" if args.recursive else "softprompt_cached",
                "base_model": base_model_name,
                "num_epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.lr, "soft_prompt_length": args.soft_prompt_length,
                "recursive": args.recursive, "classifier_head": args.classifier_head,
                "multi_step_loss": args.multi_step_loss,
                "best_epoch": best_epoch,
                "best_step": best_step,
                "num_train": num_train, "num_val": num_val, "num_test": num_test,
            },
            "training_history": history,
            "val_metrics": val_metrics,
            "metrics": best_epoch_metrics,  # test metrics (headline)
        }
        out_path = os.path.join(args.output_dir, "final_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
