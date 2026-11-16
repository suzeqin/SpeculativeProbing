"""Compatibility shim for the MiniCPM4.1-8B remote code on transformers 5.x.

The upstream `modeling_minicpm.py` imports `is_torch_fx_available` from
`transformers.utils.import_utils`; that helper was removed in transformers 5.
The symbol is only used to decide whether to wrap an attention-mask helper for
FX tracing, so returning False is safe for inference/training here.

Import this module before calling `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` on the MiniCPM checkout, then call
`fix_minicpm_rotary(model)` once the model is on its device.
"""
from __future__ import annotations

import torch

try:
    import transformers.utils.import_utils as _import_utils

    if not hasattr(_import_utils, "is_torch_fx_available"):
        _import_utils.is_torch_fx_available = lambda: False
except Exception:  # pragma: no cover - transformers is a hard dependency anyway
    pass


def fix_minicpm_rotary(model, seq_len: int = 8192) -> None:
    """Rebuild the non-persistent rotary state of the MiniCPM remote code.

    transformers 5.x instantiates remote-code models on the meta device and
    materialises weights afterwards; the legacy MiniCPM rotary implementation
    fills its non-persistent `inv_freq` / `cos_cached` / `sin_cached` buffers
    in __init__, so they end up empty or garbage (zeros / inf) and attention
    runs without positional information - the model degenerates (greedy
    generation collapses to repeated tokens). Recomputing both `inv_freq` and
    the cos/sin caches after loading restores the model.

    The cache covers `seq_len` positions (the model's own forward rebuilds it
    automatically for longer inputs; long/short factor choice is identical for
    this checkpoint because rope_scaling long_factor == short_factor).
    """
    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        return
    device = next(model.parameters()).device
    for layer in layers:
        rot = getattr(getattr(layer, "self_attn", None), "rotary_emb", None)
        if rot is None or not hasattr(rot, "_set_cos_sin_cache"):
            continue
        # inv_freq is config-derived: rebuild it (meta init leaves inf/zeros)
        dim = int(getattr(rot, "dim", 0)) or int(rot.inv_freq.numel() * 2)
        base = float(getattr(rot, "base", 10000.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        rot.inv_freq = inv_freq.to(device)
        rot._set_cos_sin_cache(seq_len=seq_len, device=device, dtype=torch.bfloat16)
