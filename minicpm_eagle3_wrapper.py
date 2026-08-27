"""Eagle3 wrapper for MiniCPM4.1-8B.

The MiniCPM-Eagle3 head has the same weight shape as Llama-Eagle3 (a single
transformer layer taking 2*hidden concat of fused hidden + embeddings), but
MiniCPM applies a per-residual scaling factor of `scale_depth / sqrt(L)`
(for MiniCPM4.1-8B: 1.4 / sqrt(32) ≈ 0.2475) inside every decoder layer.

We mirror `modeling_minicpm.py`'s residual scaling on the Eagle3 midlayer,
otherwise the midlayer's output grows uncontrolled relative to the scale the
weights were trained at.

Also: the 3-layer fusion uses base-model layers (2, L/2, L-3), confirmed from
cpm.cu's `minicpm4_eagle.cuh`. For MiniCPM4.1-8B: [2, 16, 29].
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class MiniCPMEagle3Output:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class MiniCPMEagle3(nn.Module):
    """Eagle3 draft head with MiniCPM-style residual scaling."""

    def __init__(self, config, fc, midlayer, norm, lm_head, d2t, rotary_emb,
                 scale_depth: float, num_hidden_layers_base: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.fc = fc
        self.midlayer = midlayer
        self.rotary_emb = rotary_emb
        self.norm = norm
        self.lm_head = lm_head
        self.register_buffer("d2t", d2t)
        # MiniCPM residual scale: scale_depth / sqrt(L_base)
        self.residual_scale = float(scale_depth) / math.sqrt(float(num_hidden_layers_base))

    def forward(self, hidden_states_list, inputs_embeds, return_dict=True):
        # Fuse 3 base-model hidden states via FC: 3*hidden -> hidden
        combined = torch.cat(hidden_states_list, dim=-1)
        fused = self.fc(combined)

        # Pre-norm input
        hidden_norm = self.midlayer.hidden_norm(fused)

        # Concat with embeddings for Eagle3's 2*hidden-input attention
        attn_input = torch.cat([hidden_norm, inputs_embeds], dim=-1)

        batch, seq_len, _ = attn_input.shape
        position_ids = torch.arange(seq_len, device=attn_input.device).unsqueeze(0).expand(batch, -1)
        position_embeddings = self.rotary_emb(fused, position_ids)

        attn_out = self.midlayer.self_attn(
            hidden_states=attn_input,
            position_embeddings=position_embeddings,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        # MiniCPM residual: h = residual + f(h) * (scale_depth / sqrt(L))
        h = fused + attn_out * self.residual_scale
        h = h + self.midlayer.mlp(self.midlayer.post_attention_layernorm(h)) * self.residual_scale

        h_normed = self.norm(h)
        draft_logits = self.lm_head(h_normed)
        target_logits = torch.zeros(
            batch, seq_len, int(self.d2t.max().item()) + 1,
            dtype=draft_logits.dtype, device=draft_logits.device,
        )
        target_logits[:, :, self.d2t] = draft_logits

        if return_dict:
            return MiniCPMEagle3Output(logits=target_logits, hidden_states=h)
        return target_logits, h


def from_minicpm_eagle3_checkpoint(eagle3_dir, base_config_for_scale=None):
    """Load the MiniCPM-Eagle3 draft head.

    Args:
        eagle3_dir: path to MiniCPM4_1-8B-Eagle3-bf16 directory
        base_config_for_scale: optional dict-like config of the *base* model
            (for scale_depth / num_hidden_layers). If None, defaults are used
            for MiniCPM4.1-8B: scale_depth=1.4, num_hidden_layers=32.
    """
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
    )

    with open(os.path.join(eagle3_dir, "config.json")) as f:
        e3_config = json.load(f)

    llama_cfg = LlamaConfig(
        vocab_size=e3_config.get("vocab_size", 73448),
        hidden_size=e3_config["hidden_size"],
        intermediate_size=e3_config["intermediate_size"],
        num_hidden_layers=1,
        num_attention_heads=e3_config["num_attention_heads"],
        num_key_value_heads=e3_config["num_key_value_heads"],
        hidden_act=e3_config.get("hidden_act", "silu"),
        max_position_embeddings=e3_config.get("max_position_embeddings", 65536),
        rms_norm_eps=e3_config.get("rms_norm_eps", 1e-6),
        rope_theta=e3_config.get("rope_theta", 10000.0),
        tie_word_embeddings=False,
    )

    weights_path = os.path.join(eagle3_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    else:
        from safetensors.torch import load_file
        weights_path = os.path.join(eagle3_dir, "model.safetensors")
        ckpt = load_file(weights_path)

    hidden = llama_cfg.hidden_size
    draft_vocab = e3_config.get("draft_vocab_size", 32000)

    fc = nn.Linear(3 * hidden, hidden, bias=False)
    fc.weight.data.copy_(ckpt["fc.weight"])

    midlayer = LlamaDecoderLayer(llama_cfg, layer_idx=0)
    midlayer.hidden_norm = LlamaRMSNorm(hidden, eps=llama_cfg.rms_norm_eps)

    num_heads = llama_cfg.num_attention_heads
    num_kv = llama_cfg.num_key_value_heads
    head_dim = hidden // num_heads
    midlayer.self_attn.q_proj = nn.Linear(2 * hidden, num_heads * head_dim, bias=False)
    midlayer.self_attn.k_proj = nn.Linear(2 * hidden, num_kv * head_dim, bias=False)
    midlayer.self_attn.v_proj = nn.Linear(2 * hidden, num_kv * head_dim, bias=False)

    midlayer_state = {k[len("midlayer."):]: v for k, v in ckpt.items() if k.startswith("midlayer.")}
    midlayer.load_state_dict(midlayer_state, strict=False)

    norm = LlamaRMSNorm(hidden, eps=llama_cfg.rms_norm_eps)
    norm.weight.data.copy_(ckpt["norm.weight"])

    lm_head = nn.Linear(hidden, draft_vocab, bias=False)
    lm_head.weight.data.copy_(ckpt["lm_head.weight"])

    d2t = ckpt["d2t"]
    rotary_emb = LlamaRotaryEmbedding(config=llama_cfg)

    # MiniCPM base-model scale parameters
    if base_config_for_scale is None:
        scale_depth = 1.4
        num_hidden_layers_base = 32
    else:
        scale_depth = float(base_config_for_scale.get("scale_depth", 1.4))
        num_hidden_layers_base = int(base_config_for_scale.get("num_hidden_layers", 32))

    return MiniCPMEagle3(llama_cfg, fc, midlayer, norm, lm_head, d2t,
                         rotary_emb=rotary_emb,
                         scale_depth=scale_depth,
                         num_hidden_layers_base=num_hidden_layers_base)
