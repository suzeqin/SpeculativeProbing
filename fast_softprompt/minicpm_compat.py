"""Compatibility shim for the MiniCPM4.1-8B remote code on transformers 5.x.

The upstream `modeling_minicpm.py` imports `is_torch_fx_available` from
`transformers.utils.import_utils`; that helper was removed in transformers 5.
The symbol is only used to decide whether to wrap an attention-mask helper for
FX tracing, so returning False is safe for inference/training here.

Import this module before calling `AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)` on the MiniCPM checkout.
"""
from __future__ import annotations

try:
    import transformers.utils.import_utils as _import_utils

    if not hasattr(_import_utils, "is_torch_fx_available"):
        _import_utils.is_torch_fx_available = lambda: False
except Exception:  # pragma: no cover - transformers is a hard dependency anyway
    pass
