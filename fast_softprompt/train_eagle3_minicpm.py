#!/usr/bin/env python3
"""Soft-prompt classifier on top of the MiniCPM4.1-8B Eagle3 head.

Mirrors train_softprompt.py's Rec sp=K linear+sigmoid setup, but the backbone
is the Eagle3 draft head (single transformer layer) rather than the Qwen3.5
MTP layer. Uses precomputed 3-layer hidden states from the MiniCPM base model
(see precompute_hidden_states_minicpm.py).

Usage:
  python fast_softprompt/train_eagle3_minicpm.py \
    --cache fast_softprompt/cache_<task>_mcpm_eagle3.pt \
    --eagle3_dir /path/to/MiniCPM4_1-8B-Eagle3-bf16 \
    --output_dir fast_softprompt/results_<task>_mcpm/eagle3_sp2 \
    --soft_prompt_length 2 --epochs 10 --batch_size 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support,
)
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import minicpm_compat  # noqa: E402  (transformers 5.x remote-code shim)
from minicpm_eagle3_wrapper import from_minicpm_eagle3_checkpoint

RANDOM_SEED = 42


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def _load_embed_from_safetensors(model_dir):
    """Load embed_tokens from a sharded safetensors repo (no full model load)."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # single file
        from safetensors.torch import load_file
        for fn in os.listdir(model_dir):
            if fn.endswith(".safetensors"):
                w = load_file(os.path.join(model_dir, fn))
                for k, v in w.items():
                    if "embed_tokens.weight" in k:
                        return v
        raise RuntimeError("no embed_tokens in " + model_dir)
    with open(index_path) as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    embed_key = next(k for k in wm if "embed_tokens.weight" in k)
    from safetensors.torch import load_file
    shard = load_file(os.path.join(model_dir, wm[embed_key]))
    return shard[embed_key]


class Eagle3SoftPromptClassifier(nn.Module):
    """Eagle3 backbone + learned soft-prompt + sigmoid linear head.

    The 3 fused hidden layers come from the frozen base model (precomputed).
    Soft-prompt tokens are appended like in train_softprompt.py's cached variant.
    A trainable Linear(hidden, 1) on the Eagle3 output produces one logit per
    sample; BCEWithLogitsLoss treats it as binary classification.
    """

    def __init__(self, eagle3, embed_weights, soft_prompt_length=1, device="cuda",
                 dropout=0.0):
        super().__init__()
        self.eagle3 = eagle3
        self.embed_weights = embed_weights
        self.device = device
        self.dropout = nn.Dropout(dropout).to(device)

        self.eagle3.eval()
        for p in self.eagle3.parameters():
            p.requires_grad = False

        hidden_dim = embed_weights.size(1)
        # Eagle3 has 4 input streams: inputs_embeds + 3 fused hidden states (low/mid/high).
        # Learn a separate soft-prompt vector per stream (shape [1, k, hidden] each).
        init = embed_weights[0].detach().to(device=device, dtype=torch.float32).view(1, 1, -1)
        self.soft_prompt_embed = nn.Parameter(init.clone().repeat(1, soft_prompt_length, 1), requires_grad=True)
        self.soft_prompt_h_low = nn.Parameter(torch.zeros(1, soft_prompt_length, hidden_dim, device=device), requires_grad=True)
        self.soft_prompt_h_mid = nn.Parameter(torch.zeros(1, soft_prompt_length, hidden_dim, device=device), requires_grad=True)
        self.soft_prompt_h_high = nn.Parameter(torch.zeros(1, soft_prompt_length, hidden_dim, device=device), requires_grad=True)
        self.head = nn.Linear(hidden_dim, 1).to(device=device, dtype=torch.float32)

    def forward(self, eagle3_hidden_list, token_ids, attention_mask, labels=None):
        k = self.soft_prompt_embed.size(1)

        inputs_embeds = self.embed_weights[token_ids.to(self.embed_weights.device)].to(self.device).clone()
        eagle3_hidden_list = [h.to(self.device) for h in eagle3_hidden_list]
        attention_mask = attention_mask.to(self.device)

        B = token_ids.size(0)
        if k > 1:
            _, seq_len, hdim = inputs_embeds.shape
            ext = k - 1
            inputs_embeds = torch.cat([inputs_embeds, torch.zeros(B, ext, hdim, dtype=inputs_embeds.dtype, device=self.device)], dim=1)
            eagle3_hidden_list = [
                torch.cat([h, torch.zeros(B, ext, hdim, dtype=h.dtype, device=self.device)], dim=1)
                for h in eagle3_hidden_list
            ]
            attention_mask = torch.cat([attention_mask, torch.zeros(B, ext, dtype=attention_mask.dtype, device=self.device)], dim=1)

        sp_e = self.soft_prompt_embed[0, :k, :].to(inputs_embeds.dtype)
        sp_h = [
            self.soft_prompt_h_low[0, :k, :].to(eagle3_hidden_list[0].dtype),
            self.soft_prompt_h_mid[0, :k, :].to(eagle3_hidden_list[1].dtype),
            self.soft_prompt_h_high[0, :k, :].to(eagle3_hidden_list[2].dtype),
        ]
        for b in range(B):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len <= 0:
                continue
            new_valid_len = valid_len + (k - 1)
            # Shift embeds left by 1 and append sp_e at end (like Rec training does)
            shifted = torch.cat([inputs_embeds[b, 1:valid_len, :], sp_e], dim=0)
            inputs_embeds[b, :new_valid_len, :] = shifted
            # Replace hidden states at the appended positions with the learned soft prompts.
            # For k=1: the last position (valid_len-1) becomes a learned sp_h token.
            # For k>1: positions [valid_len-1, valid_len+k-2] = sp_h[0:k].
            for li in range(3):
                eagle3_hidden_list[li][b, valid_len - 1 : valid_len + k - 1, :] = sp_h[li]
            attention_mask[b, valid_len:new_valid_len] = 1

        e3_dtype = self.eagle3.fc.weight.dtype
        out = self.eagle3(
            hidden_states_list=[h.to(dtype=e3_dtype) for h in eagle3_hidden_list],
            inputs_embeds=inputs_embeds.to(dtype=e3_dtype),
            return_dict=True,
        )
        last_valid = attention_mask.sum(dim=1).clamp(min=1) - 1
        bidx = torch.arange(B, device=self.device)
        h_final = out.hidden_states[bidx, last_valid, :]
        h_final = self.dropout(h_final)
        z = self.head(h_final.float()).squeeze(-1)  # [B]
        logits = torch.stack([-z, z], dim=-1)
        loss = None
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(z, labels.to(z.device, dtype=z.dtype))
        return logits, loss


def pad_and_collate(batch_eagle3_hidden, batch_ids, batch_labels):
    max_len = max(h[0].size(0) for h in batch_eagle3_hidden)
    hidden_dim = batch_eagle3_hidden[0][0].size(1)
    B = len(batch_eagle3_hidden)
    dtype = batch_eagle3_hidden[0][0].dtype

    h_padded = [torch.zeros(B, max_len, hidden_dim, dtype=dtype) for _ in range(3)]
    ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, (htup, ids) in enumerate(zip(batch_eagle3_hidden, batch_ids)):
        n = htup[0].size(0)
        for li in range(3):
            h_padded[li][i, :n] = htup[li]
        ids_padded[i, :n] = torch.tensor(ids, dtype=torch.long)
        mask[i, :n] = 1
    labels = torch.tensor(batch_labels, dtype=torch.long)
    return h_padded, ids_padded, mask, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--eagle3_dir", required=True)
    p.add_argument("--base_model", default="openbmb/MiniCPM4.1-8B")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--soft_prompt_length", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--target_eval", type=int, default=200)
    p.add_argument("--val_fraction", type=float, default=0.5)
    p.add_argument("--mmap_cache", action="store_true",
                    help="mmap the hidden-states tensor (requires mmap_v1 cache format).")
    args = p.parse_args()

    set_seed()

    print(f"Loading cache {args.cache} (mmap={args.mmap_cache})...")
    from cache_io import load_cache
    cache = load_cache(args.cache, mmap=args.mmap_cache)
    all_ids = cache["token_ids"]
    all_labels = cache["labels"]
    task = cache["task"]

    if "eagle3_hidden_states" in cache:
        # Legacy: list of 3-tuples, one per sample.
        all_e3 = cache["eagle3_hidden_states"]
    else:
        # mmap_v1 concatenated along feature dim. Expose a list-like view that
        # returns tuples of per-layer slices (zero-copy views).
        n_layers = int(cache.get("eagle3_num_layers", 3))
        d_layer = int(cache.get("eagle3_per_layer_dim",
                                 cache["hidden_dim"] // n_layers))
        flat_list = cache["hidden_states"]  # LazyHiddenStates — returns [L, 3d]
        class _EagleListView:
            def __init__(self, flat, n_layers, d_layer):
                self._flat = flat
                self._n_layers = n_layers
                self._d_layer = d_layer
            def __len__(self):
                return len(self._flat)
            def __getitem__(self, idx):
                if isinstance(idx, slice):
                    return [self[i] for i in range(*idx.indices(len(self)))]
                h = self._flat[idx]  # [L, 3d]
                return tuple(h[:, i * self._d_layer:(i + 1) * self._d_layer]
                             for i in range(self._n_layers))
        all_e3 = _EagleListView(flat_list, n_layers, d_layer)

    from eval_split import make_val_test_split
    split = make_val_test_split(
        all_e3, all_ids, all_labels,
        num_train_cached=cache["num_train"],
        num_test_cached=cache["num_test"],
        target_eval=args.target_eval,
        val_fraction=args.val_fraction,
    )
    num_train = split["num_train_used"]
    num_val = split["num_val"]
    num_test = split["num_test"]
    train_h = split["train_hidden"]
    train_ids = split["train_ids"]
    train_y = split["train_labels"]
    val_h = split["val_hidden"]
    val_ids = split["val_ids"]
    val_y = split["val_labels"]
    test_h = split["test_hidden"]
    test_ids_ = split["test_ids"]
    test_y = split["test_labels"]
    print(f"  Task: {task}, Train: {num_train}, Val: {num_val}, Test: {num_test} "
          f"(borrowed {split['borrowed_from_train']} from train -> eval)")

    # Resolve paths
    e3_dir = args.eagle3_dir
    if not os.path.isdir(e3_dir):
        from huggingface_hub import snapshot_download
        e3_dir = snapshot_download(e3_dir, local_files_only=True)

    base_local = args.base_model
    if not os.path.isdir(base_local):
        from huggingface_hub import snapshot_download
        base_local = snapshot_download(args.base_model, local_files_only=True)

    print(f"Loading Eagle3 from {e3_dir}...")
    e3 = from_minicpm_eagle3_checkpoint(e3_dir)
    device = "cuda"
    e3 = e3.to(device=device, dtype=torch.bfloat16).eval()
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print("Loading base embeddings...")
    embed_w = _load_embed_from_safetensors(base_local)
    embed_w = embed_w.to(device=device, dtype=torch.bfloat16)

    clf = Eagle3SoftPromptClassifier(
        eagle3=e3, embed_weights=embed_w,
        soft_prompt_length=args.soft_prompt_length, device=device,
        dropout=args.dropout,
    )
    trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    params = [p for p in clf.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scheduler = None
    if args.min_lr > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
        print(f"Cosine LR: {args.lr:.2e} -> {args.min_lr:.2e} over {args.epochs} epochs")

    target_names = {
        "wildchat": ["same_topic", "different_topic"],
        "verih": ["aligned", "conflict"],
        "adversarial_if": ["correct", "incorrect"],
        "nemotron_safety": ["safe", "unsafe"],
        "cot_repetitive": ["stay", "repetitive"],
        "cot_branched": ["stay", "branch"],
    }.get(task, ["class_0", "class_1"])

    best_eval_acc = -1.0
    best_metrics = None
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        clf.train()
        perm = list(range(num_train))
        random.shuffle(perm)
        running = 0.0
        correct = 0
        seen = 0
        recent = deque(maxlen=20)
        nb = math.ceil(num_train / args.batch_size)
        pbar = tqdm(range(0, num_train, args.batch_size), total=nb, desc=f"Epoch {epoch+1}/{args.epochs}")
        for s in pbar:
            e = min(s + args.batch_size, num_train)
            bidx = perm[s:e]
            bh = [train_h[i] for i in bidx]
            bi = [train_ids[i] for i in bidx]
            bl = [train_y[i] for i in bidx]
            h_padded, ids_p, mask, labels = pad_and_collate(bh, bi, bl)
            optimizer.zero_grad()
            logits, loss = clf(h_padded, ids_p, mask, labels)
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()
            running += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1).cpu()
                gold = labels.cpu()
                correct += (preds == gold).sum().item()
                seen += gold.numel()
                recent.extend((preds == gold).tolist())
            racc = sum(recent) / max(1, len(recent))
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{racc:.3f}", "lr": f"{cur_lr:.2e}"})

        def run_eval(hs, idss, ys):
            clf.eval()
            prs, gds = [], []
            with torch.no_grad():
                for s in range(0, len(ys), args.batch_size):
                    e = min(s + args.batch_size, len(ys))
                    hp, ip, mk, lb = pad_and_collate(hs[s:e], idss[s:e], ys[s:e])
                    lg, _ = clf(hp, ip, mk, None)
                    prs.extend(torch.argmax(lg, dim=-1).cpu().tolist())
                    gds.extend(lb.tolist())
            a = accuracy_score(gds, prs)
            p, r, f, _ = precision_recall_fscore_support(gds, prs, average="binary", zero_division=0)
            return prs, gds, a, float(p), float(r), float(f)

        _, _, val_acc, val_pr, val_rc, val_f1 = run_eval(val_h, val_ids, val_y)
        cur_lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch + 1, "lr": cur_lr,
               "train_loss": running / max(nb, 1),
               "train_acc": correct / max(seen, 1),
               "val_acc": val_acc}
        history.append(row)
        is_best = val_acc > best_eval_acc
        msg = (f"Epoch {epoch+1}: lr={cur_lr:.2e}, loss={row['train_loss']:.4f}, "
               f"train_acc={row['train_acc']:.4f}, val_acc={val_acc:.4f}")
        if is_best:
            best_eval_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
            msg += " *best*"
        print(msg)
        if scheduler is not None:
            scheduler.step()

    if best_epoch > 0:
        clf.load_state_dict(best_state)
    tp, tg, test_acc, test_pr, test_rc, test_f1 = run_eval(test_h, test_ids_, test_y)
    _, _, vA, vP, vR, vF = run_eval(val_h, val_ids, val_y)
    print(f"\nBest epoch (by val): {best_epoch} val_acc={best_eval_acc:.4f}")
    print(f"Test  | Acc: {test_acc:.4f} | P: {test_pr:.4f} | R: {test_rc:.4f} | F1: {test_f1:.4f}")
    print(classification_report(tg, tp, target_names=target_names, zero_division=0))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "config": {
                "task": task,
                "arch": "eagle3_softprompt_minicpm",
                "base_model": args.base_model,
                "eagle3_dir": args.eagle3_dir,
                "soft_prompt_length": args.soft_prompt_length,
                "num_epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "min_lr": args.min_lr,
                "dropout": args.dropout,
                "best_epoch": best_epoch,
                "num_train": num_train, "num_val": num_val, "num_test": num_test,
            },
            "training_history": history,
            "val_metrics": {"accuracy": vA, "precision": vP, "recall": vR, "f1": vF},
            "metrics": {"accuracy": test_acc, "precision": test_pr, "recall": test_rc, "f1": test_f1},
        }
        with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to {os.path.join(args.output_dir, 'final_results.json')}")


if __name__ == "__main__":
    main()
