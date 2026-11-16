#!/usr/bin/env python3
"""Probe adapters for the speculative-decoding harness.

Each adapter wraps a *trained* probe checkpoint and exposes a single
``__call__(engine)`` that classifies the currently committed sequence
(engine.committed / engine.base_hiddens / engine.draft_cache) and returns the
predicted label (1 = attack, 0 = benign). Probes never draft; SP probes append
k soft-prompt positions to the draft cache and crop them back afterwards, so
the decoding path is untouched.

Loading follows ``measure_overhead.py`` (the validated reproduction harness):
checkpoints are restored into the exact classifier classes used for training.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "fast_softprompt")):
    if p not in sys.path:
        sys.path.insert(0, p)

BF16 = torch.bfloat16
DEVICE = "cuda"


def _label_token_ids(base_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    pos = tok.encode("yes", add_special_tokens=False)[-1]
    neg = tok.encode("no", add_special_tokens=False)[-1]
    return pos, neg


class SPProbeQwen:
    """SP-k on the Qwen3.5 MTP head (recursive soft prompts, linear head)."""

    def __init__(self, ckpt_path: str, base_dir: str, mtp_dir: str, k: int,
                 dropout: float = 0.2):
        from qwen35_mtp import Qwen3_5MTP
        from train_softprompt import RecursiveSoftPromptClassifier, _load_embed_weights

        self.k = k
        self.mtp = Qwen3_5MTP.from_mtp_checkpoint(mtp_dir, base_dir).to(DEVICE, BF16).eval()
        embed_w, lm_head_w = _load_embed_weights(base_dir)
        self.embed_w = embed_w.to(DEVICE, BF16)
        # share base lm_head (the MTP checkpoint has no lm_head of its own).
        # PyTorch 2.8 rejects assigning a raw Tensor to a Parameter attribute;
        # wrap it so the weight is a proper nn.Parameter.
        with torch.no_grad():
            self.mtp.lm_head.weight = torch.nn.Parameter(lm_head_w.to(DEVICE, BF16))
            self.mtp.model.embed_tokens.weight = torch.nn.Parameter(self.embed_w)
        pos_id, neg_id = _label_token_ids(base_dir)
        clf = RecursiveSoftPromptClassifier(
            mtp_model=self.mtp, embed_weights=self.embed_w,
            pos_token_id=pos_id, neg_token_id=neg_id,
            num_recursive_steps=k, device=DEVICE, classifier_head="linear",
            dropout=dropout,
        )
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = clf.load_state_dict(ck["state_dict"], strict=False)
        assert not unexpected, f"unexpected keys in {ckpt_path}: {unexpected}"
        assert not [m for m in missing if m.startswith(("soft_prompt", "head"))], \
            f"missing trained keys in {ckpt_path}: {missing}"
        clf.eval()
        self.sp = torch.stack([p[0, 0] for p in clf.soft_prompts]).detach().to(DEVICE)  # [k, d]
        self.head = clf.head
        self.sp_bf16 = self.sp.to(BF16)

    @torch.no_grad()
    def __call__(self, engine):
        cache = engine.draft_cache
        m = engine.m
        h = engine.base_hiddens[m - 1].to(BF16).view(1, 1, -1)
        for i in range(self.k):
            pos = m - 1 + i
            pos3 = torch.tensor([[[pos]]], device=DEVICE).expand(3, 1, 1)
            cpos = torch.tensor([pos], device=DEVICE)
            out = self.mtp.model(hidden_states=h, inputs_embeds=self.sp_bf16[i].view(1, 1, -1),
                                 use_cache=True, past_key_value=cache,
                                 position_ids=pos3, cache_position=cpos)
            h = out[0]
        z = self.head(h[:, 0].float()).squeeze(-1)
        engine.probe_tail_crop(self.k)
        return int(z.item() > 0)


class SPProbeEagle3:
    """SP-k on the MiniCPM4.1 Eagle3 head (per-step fused soft prompts)."""

    def __init__(self, ckpt_path: str, base_dir: str, eagle3_dir: str, k: int,
                 dropout: float = 0.2):
        from train_eagle3_minicpm import Eagle3SoftPromptClassifier, _load_embed_from_safetensors
        from minicpm_eagle3_wrapper import from_minicpm_eagle3_checkpoint

        self.k = k
        self.e3 = from_minicpm_eagle3_checkpoint(eagle3_dir).to(DEVICE, BF16).eval()
        embed_w = _load_embed_from_safetensors(base_dir).to(DEVICE, BF16)
        clf = Eagle3SoftPromptClassifier(eagle3=self.e3, embed_weights=embed_w,
                                         soft_prompt_length=k, device=DEVICE,
                                         dropout=dropout)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = clf.load_state_dict(ck["state_dict"], strict=False)
        assert not unexpected, f"unexpected keys in {ckpt_path}: {unexpected}"
        assert not [m for m in missing if m.startswith(("soft_prompt", "head"))], \
            f"missing trained keys in {ckpt_path}: {missing}"
        clf.eval()
        self.sp_e = clf.soft_prompt_embed[0].detach().to(DEVICE)            # [k, d]
        self.sp_h = torch.stack([clf.soft_prompt_h_low[0], clf.soft_prompt_h_mid[0],
                                 clf.soft_prompt_h_high[0]]).detach().to(DEVICE)  # [3, k, d]
        self.head = clf.head
        self.sp_e_bf16 = self.sp_e.to(BF16)
        self.sp_h_bf16 = self.sp_h.to(BF16)

    @torch.no_grad()
    def __call__(self, engine):
        e3 = engine.e3
        cache = engine.draft_cache
        m = engine.m
        h_last = None
        for i in range(self.k):
            fused_i = e3.fc(torch.cat(
                [self.sp_h_bf16[li, i].view(1, 1, -1) for li in range(3)], dim=-1))
            hidden_norm = e3.midlayer.hidden_norm(fused_i)
            attn_input = torch.cat([hidden_norm, self.sp_e_bf16[i].view(1, 1, -1)], dim=-1)
            pos_ids = torch.tensor([[m - 1 + i]], device=DEVICE)
            pos_embs = e3.rotary_emb(fused_i, pos_ids)
            attn_out = e3.midlayer.self_attn(hidden_states=attn_input,
                                             position_embeddings=pos_embs,
                                             past_key_values=cache)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]
            h = fused_i + attn_out * e3.residual_scale
            h = h + e3.midlayer.mlp(e3.midlayer.post_attention_layernorm(h)) * e3.residual_scale
            h_last = h
        z = self.head(h_last[:, 0].float()).squeeze(-1)
        engine.probe_tail_crop(self.k)
        return int(z.item() > 0)


class MLPProbe:
    """Last-token MLP probe on the base model's last-layer hidden state."""

    def __init__(self, ckpt_path: str, hidden_dim: int, mlp_hidden: int = 2048,
                 dropout: float = 0.1):
        from train_mlp import MLPClassifier
        clf = MLPClassifier(hidden_dim, num_labels=2, mlp_hidden=mlp_hidden, dropout=dropout)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        clf.load_state_dict(ck["state_dict"], strict=True)
        self.clf = clf.to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, engine):
        h = engine.base_hiddens[engine.m - 1].float().unsqueeze(0)   # [1, 1, d]
        logits, _ = self.clf(h)
        return int(logits[0].argmax().item())


class MMProbe:
    """MultiMax probe over all mid-layer positions (hard max over sequence)."""

    def __init__(self, ckpt_path: str, hidden_dim: int, num_heads: int = 10,
                 mlp_width: int = 100, dropout: float = 0.1):
        from train_multimax import MultiMaxClassifier
        clf = MultiMaxClassifier(hidden_dim, num_heads=num_heads, mlp_width=mlp_width,
                                 dropout=dropout)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        clf.load_state_dict(ck["state_dict"], strict=True)
        self.clf = clf.to(DEVICE).eval()

    @torch.no_grad()
    def __call__(self, engine):
        # engine.mid_hidden: list of per-position mid-layer hiddens [d]
        h = torch.stack(engine.mid_hidden).unsqueeze(0)              # [1, m, d]
        mask = torch.ones(1, h.size(1), device=DEVICE)
        logits, _ = self.clf(h, mask)
        return int(logits[0].argmax().item())


def load_probe(kind: str, model_tag: str, task: str, probes_root: str,
               spec: dict, k: int | None = None):
    """Load a trained probe for (model_tag, task). kind in sp/mlp/mm.

    spec keys accept either base_dir/head_dir (measure_overhead style) or
    base/head (run_bench style).
    """
    base = spec.get("base_dir") or spec.get("base")
    head = spec.get("head_dir") or spec.get("head")
    if kind == "mlp":
        return MLPProbe(os.path.join(probes_root, f"{task}_{model_tag}", "mlp", "best.pt"),
                        spec["hidden"])
    if kind == "mm":
        return MMProbe(os.path.join(probes_root, f"{task}_{model_tag}", "multimax", "best.pt"),
                       spec["hidden"])
    if kind.startswith("sp"):
        ckpt = os.path.join(probes_root, f"{task}_{model_tag}", f"sp{k}", "best.pt")
        if spec["kind"] == "mtp":
            return SPProbeQwen(ckpt, base, head, k)
        return SPProbeEagle3(ckpt, base, head, k)
    raise ValueError(kind)
