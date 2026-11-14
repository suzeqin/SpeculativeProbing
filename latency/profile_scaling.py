#!/usr/bin/env python3
"""Device-time scaling of every probe with sequence length.

Profiles the same probe modules as ``bench_latency.py`` on a fixed set of
test-set examples spanning the length distribution (no base model needed:
the cached base features are the probe inputs).  Interleaves methods across
trials and takes the min per method, so a co-tenant's contention burst cannot
systematically favour one method.

Usage:
  python latency/profile_scaling.py --model qwen3.5-4b --task verih \
      --points 6 --trials 5 --out latency/results/scaling_qwen3.5-4b_verih.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
from bench_latency import build_methods, prepare_task_data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(C.MODEL_SPECS))
    ap.add_argument("--task", required=True, choices=sorted(C.TASK_TARGET_EVAL))
    ap.add_argument("--methods", default="mlp,mm,sp1,sp2,sp5,sp1_inc,sp2_inc,sp5_inc")
    ap.add_argument("--points", type=int, default=6)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    spec = C.MODEL_SPECS[args.model]
    methods_wanted = [m.strip() for m in args.methods.split(",") if m.strip()]
    out_path = args.out or os.path.join(
        HERE, "results", f"scaling_{args.model}_{args.task}.json")

    # First load the metadata to get the test-set indices/lengths.
    caches = spec["caches"]
    meta_last = C.load_cache_meta(C.abs_path(caches["last"].format(task=args.task)))
    all_idx = C.split_test_indices(meta_last["num_train"], meta_last["num_test"],
                                   C.TASK_TARGET_EVAL[args.task])
    lengths = [len(meta_last["token_ids"][i]) for i in all_idx]
    order = sorted(range(len(all_idx)), key=lambda k: lengths[k])
    qs = [0.05] if args.points == 1 else [0.05 + 0.9 * k / (args.points - 1) for k in range(args.points)]
    picks = sorted({order[min(len(order) - 1, int(q * (len(order) - 1)))] for q in qs})
    idx_points = [all_idx[k] for k in picks]
    print(f"[{args.model}/{args.task}] points: "
          f"{[lengths[k] for k in picks]} (n={len(idx_points)})", flush=True)
    del meta_last

    data = prepare_task_data(spec, args.task, device, indices=idx_points)
    methods, prepare_hooks, release_hooks, between, keepalive = build_methods(
        spec, data, device, methods_wanted, seed=0)
    method_names = [m for m in methods_wanted if m in methods]
    print("methods:", method_names, flush=True)

    result: Dict[str, Any] = {
        "config": {
            "model": args.model, "task": args.task, "methods": method_names,
            "points": args.points, "trials": args.trials, "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device), "torch": torch.__version__,
        },
        "gpu": C.gpu_snapshot(torch.cuda.current_device()),
        "lengths": data["lengths"],
        "indices": data["idxs"],
        "probe_params": keepalive.get("params", {}),
        "points": [],
    }
    n_ex = len(data["lengths"])
    for j in range(n_ex):
        for name in method_names:
            if name in prepare_hooks:
                prepare_hooks[name](j)
        for name in method_names:  # untimed warmup
            with torch.no_grad():
                methods[name](j)
            if name in between:
                between[name]()
        samples: Dict[str, List[float]] = {name: [] for name in method_names}
        for _t in range(args.trials):
            for name in method_names:
                if name in between:
                    between[name]()
                samples[name].append(C.profile_one_call(lambda fn=methods[name], i=j: fn(i))["cuda_us"])
                if name in between:
                    between[name]()
        entry = {"length": data["lengths"][j],
                 "methods": {m: C.summarize_profile_samples(v) for m, v in samples.items()}}
        result["points"].append(entry)
        print(f"  n={entry['length']:5d}  " +
              "  ".join(f"{m}={entry['methods'][m]['cuda_us_per_call']:.0f}us" for m in method_names),
              flush=True)
        for name in method_names:
            if name in release_hooks:
                release_hooks[name](j)

    C.write_json(out_path, result)
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
