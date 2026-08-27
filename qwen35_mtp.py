#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _load_qwen35_classes() -> tuple[type, type, type, type]:
    cfg_mod = importlib.import_module("transformers.models.qwen3_5.configuration_qwen3_5")
    model_mod = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    return (
        getattr(cfg_mod, "Qwen3_5TextConfig"),
        getattr(model_mod, "Qwen3_5DecoderLayer"),
        getattr(model_mod, "Qwen3_5RMSNorm"),
        getattr(model_mod, "Qwen3_5TextRotaryEmbedding"),
    )


def _to_mtp_positions(position_ids: Optional[torch.Tensor], seq_len: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if position_ids is None:
        base = torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1)
        return base.unsqueeze(0).expand(3, batch_size, seq_len)
    if position_ids.ndim == 2:
        return position_ids.unsqueeze(0).expand(3, position_ids.shape[0], position_ids.shape[1])
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:]
    if position_ids.ndim == 3 and position_ids.shape[0] == 3:
        return position_ids
    raise ValueError(f"Unsupported position_ids shape for Qwen3.5 MTP: {tuple(position_ids.shape)}")


@dataclass
class Qwen3_5MTPOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    past_key_value: Any = None


class Qwen3_5MultiTokenPredictor(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        _, Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding = _load_qwen35_classes()
        self.hidden_size = cfg.hidden_size
        self.num_mtp_layers = int(getattr(cfg, "mtp_num_hidden_layers", 1))
        eps = cfg.rms_norm_eps

        # MTP block in vLLM uses full-attention decoder layers.
        mtp_cfg = deepcopy(cfg)
        mtp_cfg.num_hidden_layers = self.num_mtp_layers
        mtp_cfg.layer_types = ["full_attention"] * self.num_mtp_layers
        mtp_cfg._attn_implementation = "sdpa"

        self.embed_tokens = nn.Embedding(cfg.vocab_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.layers = nn.ModuleList(Qwen3_5DecoderLayer(mtp_cfg, i) for i in range(self.num_mtp_layers))
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(self.hidden_size, eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(self.hidden_size, eps)
        self.norm = Qwen3_5RMSNorm(self.hidden_size, eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=mtp_cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        spec_step_idx: int = 0,
        past_key_value: Any = None,
        use_cache: bool = False,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        cache_position = kwargs.pop("cache_position", None)
        if kwargs:
            # Keep behavior explicit to avoid silent mismatches.
            raise TypeError(f"Unexpected kwargs for Qwen3_5MultiTokenPredictor: {list(kwargs.keys())}")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if x.shape[:2] != hidden_states.shape[:2]:
            raise ValueError(
                f"inputs and hidden_states must share batch/seq dims, got {tuple(x.shape[:2])} vs {tuple(hidden_states.shape[:2])}"
            )
        x = torch.cat([self.pre_fc_norm_embedding(x), self.pre_fc_norm_hidden(hidden_states)], dim=-1)
        x = self.fc(x)

        pos3 = _to_mtp_positions(position_ids, seq_len=x.shape[1], batch_size=x.shape[0], device=x.device)
        pos_emb = self.rotary_emb(x, pos3)

        x = self.layers[spec_step_idx % self.num_mtp_layers](
            hidden_states=x,
            position_embeddings=pos_emb,
            attention_mask=attention_mask,
            position_ids=pos3,
            past_key_values=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return self.norm(x), (past_key_value if use_cache else None)


class Qwen3_5MTP(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        self.config = cfg
        self.model = Qwen3_5MultiTokenPredictor(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if bool(getattr(cfg, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def _config_from_base_dir(base_dir: str | Path) -> Any:
        Qwen3_5TextConfig, _, _, _ = _load_qwen35_classes()
        root = json.loads((Path(base_dir) / "config.json").read_text())
        text_cfg = dict(root["text_config"])
        text_cfg["tie_word_embeddings"] = bool(root.get("tie_word_embeddings", False))
        return Qwen3_5TextConfig(**text_cfg)

    @classmethod
    def from_mtp_checkpoint(
        cls,
        mtp_dir: str | Path,
        base_model_dir: Optional[str | Path] = None,
        strict: bool = False,
    ) -> "Qwen3_5MTP":
        from safetensors.torch import load_file

        mtp_dir = Path(mtp_dir)
        if base_model_dir is None:
            manifest = mtp_dir / "mtp_manifest.json"
            if manifest.exists():
                base_model_dir = json.loads(manifest.read_text()).get("source_model")
        if base_model_dir is None:
            raise ValueError("Provide base_model_dir or include source_model in mtp_manifest.json")

        model = cls(cls._config_from_base_dir(base_model_dir))
        state = load_file(str(mtp_dir / "mtp.safetensors"))
        state = {("model." + k[4:]) if k.startswith("mtp.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=strict)
        return model

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Qwen3_5MTPOutput | tuple[torch.Tensor, torch.Tensor, Any]:
        h, past = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            **kwargs,
        )
        logits = F.linear(h, self.lm_head.weight)
        if return_dict:
            return Qwen3_5MTPOutput(logits=logits, hidden_states=h, past_key_value=past)
        return logits, h, past
