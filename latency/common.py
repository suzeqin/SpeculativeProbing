#!/usr/bin/env python3
"""Shared utilities for the Speculative Probing latency benchmark.

This module reuses the *exact* model code from the reproduction
(`fast_softprompt/train_softprompt.py`, `train_mlp.py`, `train_multimax.py`,
`train_eagle3_minicpm.py`) so the measured forward passes are the same modules
that produced the accuracy numbers in `results/`.

Two deployment modes are measured for Speculative Probing:

* ``standalone``: the probe is run as implemented in the training/eval code,
  i.e. the draft head (MTP / Eagle3) is executed over the whole sequence.
  This is the cost when the draft head's KV cache is *not* already resident.

* ``incremental``: the draft head's KV cache over the verified prefix is already
  resident (the speculative-decoding deployment assumed in the paper, Sec. 3.3).
  Only the ``k`` soft-prompt positions are evaluated, so the added cost is
  ``k`` single-token draft-head steps + the classifier head.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SP_DIR = os.path.join(ROOT, "fast_softprompt")
for p in (ROOT, SP_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------------
# Experiment registry
# ----------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "qwen3.5-4b": {
        "display": "Qwen3.5-4B",
        "tag": "4b",
        "result_tag": "4b",
        "base_dir": "models/Qwen3.5-4B",
        "head_kind": "mtp",
        "head_dir": "models/Qwen3.5-4B-mtp",
        "hidden": 2560,
        "caches": {
            "last": "cache_{task}_4b.pt",
            "mid": "cache_{task}_4b_mid8.pt",
        },
        "mid_layer": 8,
    },
    "qwen3.5-9b": {
        "display": "Qwen3.5-9B",
        "tag": "9b",
        "result_tag": "9b",
        "base_dir": "models/Qwen3.5-9B",
        "head_kind": "mtp",
        "head_dir": "models/Qwen3.5-9B-mtp",
        "hidden": 4096,
        "caches": {
            "last": "cache_{task}_9b.pt",
            "mid": "cache_{task}_9b_mid.pt",
        },
        "mid_layer": 16,
    },
    "minicpm4.1-8b": {
        "display": "MiniCPM4.1-8B",
        "tag": "mcpm",
        "result_tag": "mcpm",
        "base_dir": "models/MiniCPM4.1-8B",
        "head_kind": "eagle3",
        "head_dir": "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16",
        "hidden": 4096,
        "caches": {
            "last": "cache_{task}_mcpm.pt",
            "mid": "cache_{task}_mcpm_mid.pt",
            "eagle3": "cache_{task}_mcpm_eagle3.pt",
        },
        "mid_layer": 16,
    },
}

TASK_TARGET_EVAL = {"verih": 645, "nemotron_safety": 800}

# Accuracy of the completed reproduction (`results/`), used to pair each
# latency number with the accuracy of the same configuration.
RESULT_DIRS = {"4b": "4b", "9b": "9b", "mcpm": "mcpm"}
METHOD_RESULT_DIRS = {
    "mlp": "mlp",
    "mm": "multimax",
    "sp1": "sp1",
    "sp2": "sp2",
    "sp5": "sp5",
}


def abs_path(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


# ----------------------------------------------------------------------------
# Cache / split helpers
# ----------------------------------------------------------------------------

def load_cache_meta(path: str) -> Dict[str, Any]:
    """Load cache metadata + mmap'd hidden-state view (no big RAM copy)."""
    from cache_io import load_cache

    return load_cache(path, mmap=True)


def split_test_indices(
    num_train: int,
    num_test_cached: int,
    target_eval: int,
    val_fraction: float = 0.5,
    seed: int = 1337,
) -> List[int]:
    """Reproduce ``eval_split.make_val_test_split`` indices without materializing data."""
    total = num_train + num_test_cached
    needed_from_train = max(0, target_eval - num_test_cached)
    needed_from_train = min(needed_from_train, num_train)
    eval_start = num_train - needed_from_train
    eval_indices = list(range(eval_start, total))
    rng = random.Random(seed)
    rng.shuffle(eval_indices)
    n_val = int(round(val_fraction * len(eval_indices)))
    return eval_indices[n_val:]


def load_reproduction_accuracy(task: str, result_tag: str) -> Dict[str, Any]:
    """Read the accuracy of the already-completed reproduction, if present."""
    out: Dict[str, Any] = {}
    for method, sub in METHOD_RESULT_DIRS.items():
        p = os.path.join(ROOT, "results", f"{task}_{result_tag}", sub, "final_results.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                d = json.load(f)
            out[method] = {
                "accuracy": d.get("metrics", {}).get("accuracy"),
                "best_epoch": d.get("config", {}).get("best_epoch"),
                "path": os.path.relpath(p, ROOT),
            }
        except Exception as exc:  # pragma: no cover
            out[method] = {"error": str(exc)}
    return out


# ----------------------------------------------------------------------------
# Probe construction (identical modules/weights-shapes to the reproduction)
# ----------------------------------------------------------------------------

def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_mlp(hidden_dim: int, device: torch.device, seed: int = 0):
    from train_mlp import MLPClassifier

    set_deterministic_seed(seed)
    m = MLPClassifier(hidden_dim, num_labels=2, mlp_hidden=1024, dropout=0.1)
    return m.to(device).eval()


def build_multimax(hidden_dim: int, device: torch.device, seed: int = 0):
    from train_multimax import MultiMaxClassifier

    set_deterministic_seed(seed)
    m = MultiMaxClassifier(hidden_dim, num_heads=10, mlp_width=100, num_labels=2, dropout=0.1)
    return m.to(device).eval()


class QwenSPBundle:
    """Holds the frozen MTP head + frozen base embeddings shared by SP-1/2/5."""

    def __init__(self, base_dir: str, mtp_dir: str, device: torch.device):
        from qwen35_mtp import Qwen3_5MTP
        from train_softprompt import _load_embed_weights

        self.device = device
        self.base_dir = base_dir
        mtp = Qwen3_5MTP.from_mtp_checkpoint(mtp_dir, base_model_dir=base_dir)
        self.dtype = next(mtp.parameters()).dtype
        self.mtp = mtp.to(device=device, dtype=torch.bfloat16).eval()
        for p in self.mtp.parameters():
            p.requires_grad = False
        embed, _ = _load_embed_weights(base_dir)
        self.embed_weights = embed.to(device=device, dtype=torch.bfloat16)

    def build(self, k: int, seed: int = 0):
        from train_softprompt import RecursiveSoftPromptClassifier

        set_deterministic_seed(seed)
        clf = RecursiveSoftPromptClassifier(
            self.mtp,
            self.embed_weights,
            pos_token_id=0,
            neg_token_id=1,
            num_recursive_steps=k,
            device=str(self.device),
            classifier_head="linear",
            dropout=0.0,
            head_init="random",
        )
        return clf.eval()


class Eagle3SPBundle:
    """Holds the frozen MiniCPM Eagle3 head + frozen base embeddings."""

    def __init__(self, base_dir: str, eagle3_dir: str, device: torch.device):
        import minicpm_compat  # noqa: F401
        from minicpm_eagle3_wrapper import from_minicpm_eagle3_checkpoint
        from train_eagle3_minicpm import _load_embed_from_safetensors

        self.device = device
        eagle3 = from_minicpm_eagle3_checkpoint(eagle3_dir)
        self.dtype = next(eagle3.parameters()).dtype
        self.eagle3 = eagle3.to(device=device, dtype=torch.bfloat16).eval()
        for p in self.eagle3.parameters():
            p.requires_grad = False
        embed = _load_embed_from_safetensors(base_dir)
        self.embed_weights = embed.to(device=device, dtype=torch.bfloat16)

    def build(self, k: int, seed: int = 0):
        from train_eagle3_minicpm import Eagle3SoftPromptClassifier

        set_deterministic_seed(seed)
        clf = Eagle3SoftPromptClassifier(
            self.eagle3,
            self.embed_weights,
            soft_prompt_length=k,
            device=str(self.device),
            dropout=0.0,
        )
        return clf.eval()


def build_sp_bundle(spec: Dict[str, Any], device: torch.device):
    if spec["head_kind"] == "mtp":
        return QwenSPBundle(abs_path(spec["base_dir"]), abs_path(spec["head_dir"]), device)
    return Eagle3SPBundle(abs_path(spec["base_dir"]), abs_path(spec["head_dir"]), device)


# ----------------------------------------------------------------------------
# Timing primitives
# ----------------------------------------------------------------------------

class Timer:
    """CUDA-event per-call timing with L2 flushing between calls.

    ``time_gpu`` returns one GPU-time sample (microseconds) per call.  Because
    every call is bracketed by its own pair of events, the sample is the GPU
    execution time of that call only (kernel launch gaps are excluded as long
    as the launch queue stays ahead of the GPU).
    """

    def __init__(self, device: torch.device, flush_l2: bool = True, flush_mb: int = 32):
        self.device = device
        self.flush = None
        if flush_l2:
            n = int(flush_mb * 1024 * 1024 // 4)
            self.flush = torch.empty(n, dtype=torch.float32, device=device)
        self._empty_cache_every = 0

    def _flush(self):
        if self.flush is not None:
            self.flush.zero_()

    def time_gpu(
        self,
        fn: Callable[[], Any],
        reps: int,
        warmup: int = 2,
        between: Optional[Callable[[], None]] = None,
    ) -> List[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(reps)
        ]
        for s, e in events:
            self._flush()
            if between is not None:
                between()
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) * 1e3 for s, e in events]

    def time_cpu(self, fn: Callable[[], Any], reps: int, warmup: int = 2) -> List[float]:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        out: List[float] = []
        for _ in range(reps):
            self._flush()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e6)
        return out


def summarize(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return {}
    n = len(vals)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[idx]

    return {
        "n": n,
        "mean": statistics.fmean(vals),
        "median": pct(0.5),
        "p10": pct(0.1),
        "p90": pct(0.9),
        "min": vals[0],
        "max": vals[-1],
        "std": statistics.pstdev(vals) if n > 1 else 0.0,
    }


# ----------------------------------------------------------------------------
# GPU state / calibration
# ----------------------------------------------------------------------------

def gpu_snapshot(device_index: int = 0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["torch_device_name"] = torch.cuda.get_device_name(device_index)
        props = torch.cuda.get_device_properties(device_index)
        out["total_memory_gb"] = round(props.total_memory / 1e9, 2)
        out["sm_count"] = props.multi_processor_count
    except Exception as exc:  # pragma: no cover
        out["error"] = str(exc)
    try:
        phys = os.environ.get("CUDA_VISIBLE_DEVICES", str(device_index)).split(",")[0]
        q = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,"
                "clocks.current.sm,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
                "-i",
                phys,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if q.returncode == 0:
            parts = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
            keys = ["index", "name", "util_gpu_pct", "util_mem_pct", "memory_used_mib",
                    "clock_sm_mhz", "temperature_c", "power_w"]
            out["nvidia_smi"] = dict(zip(keys, parts))
    except Exception as exc:  # pragma: no cover
        out["nvidia_smi_error"] = str(exc)
    return out


def calibrate(device: torch.device, timer: Timer) -> Dict[str, Any]:
    """Reference microbenchmarks, measured at the same time as the probes.

    These let a reader quantify how much the (shared) GPU was contended:
    the dense bf16 matmul peak for an RTX A6000 is ~155 TFLOP/s and the HBM
    peak is ~768 GB/s.
    """
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device=device)
    mm = timer.time_gpu(lambda: torch.mm(a, b), reps=20, warmup=3)
    flops = 2 * 4096 ** 3
    tf = flops / (min(mm) * 1e-6) / 1e12

    # memory-bandwidth probe: read 1 GiB, write 1 GiB
    src = torch.empty(256 * 1024 * 1024, dtype=torch.float32, device=device)
    dst = torch.empty_like(src)
    cp = timer.time_gpu(lambda: dst.copy_(src), reps=10, warmup=2, between=None)
    gbs = (2 * src.numel() * 4) / (min(cp) * 1e-6) / 1e9

    # small GEMV (decode-like, weights streamed)
    w = torch.randn(8192, 4096, dtype=torch.bfloat16, device=device)
    x = torch.randn(1, 4096, dtype=torch.bfloat16, device=device)
    gv = timer.time_gpu(lambda: torch.mm(x, w.t()), reps=20, warmup=3)
    gemv_gbs = (w.numel() * 2) / (min(gv) * 1e-6) / 1e9

    del a, b, src, dst, w, x
    torch.cuda.empty_cache()
    return {
        "matmul_4096_bf16_tflops_min": tf,
        "matmul_4096_bf16_ms_min": min(mm),
        "memcpy_gbs_min": gbs,
        "gemv_8192x4096_gbs_min": gemv_gbs,
        "raw": {"matmul_ms": mm, "memcpy_ms": cp, "gemv_ms": gv},
    }


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def move_examples(
    hidden: Sequence[torch.Tensor],
    idxs: Sequence[int],
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> List[torch.Tensor]:
    return [hidden[i].to(device=device, dtype=dtype, non_blocking=False) for i in idxs]


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def human(x: float, unit: str = "us") -> str:
    if unit == "us":
        if x >= 1000:
            return f"{x/1000:.2f} ms"
        return f"{x:.0f} us"
    return f"{x:.3f}"


def profile_one_call(fn: Callable[[], Any]) -> Dict[str, Any]:
    """Sum of CUDA kernel durations for exactly one profiled call."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    total = 0.0
    cnt = 0
    for e in prof.key_averages():
        d = float(getattr(e, "self_device_time_total", 0.0) or 0.0)
        total += d
        cnt += int(e.count)
    return {"cuda_us": total, "num_device_ops": cnt}


def summarize_profile_samples(samples: Sequence[float], n_ops: int = 0) -> Dict[str, Any]:
    import statistics as _st

    vals = sorted(float(v) for v in samples)
    return {
        "cuda_us_per_call": vals[0],
        "cuda_us_median": _st.median(vals),
        "cuda_us_mean": _st.fmean(vals),
        "cuda_us_all": list(samples),
        "num_device_ops": n_ops,
        "trials": len(vals),
    }


def profile_calls(fn: Callable[[], Any], warmup: int = 2, trials: int = 5) -> Dict[str, Any]:
    """Device-side (CUDA kernel) time of one call, repeated ``trials`` times."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    n_ops = 0
    for _ in range(trials):
        r = profile_one_call(fn)
        samples.append(r["cuda_us"])
        n_ops = max(n_ops, r["num_device_ops"])
    return summarize_profile_samples(samples, n_ops)


