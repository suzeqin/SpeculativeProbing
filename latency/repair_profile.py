#!/usr/bin/env python3
"""Re-profile the per-length device-time buckets of an existing result JSON.

Needed when a measurement-harness bug or a contention burst invalidated one of
the three probe-profile buckets.  Base-model entries in the JSON are kept
(they are not affected); probe entries are re-measured with the current
harness and the JSON is updated in place.

Usage:
  python latency/repair_profile.py --model qwen3.5-4b --task verih \
      --json latency/results/qwen3.5-4b_verih.json [--trials 7]
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
    ap.add_argument("--json", required=True)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    with open(args.json) as f:
        res = json.load(f)
    by_len_old = res["device_time_us_by_length"]
    buckets = {b: {"index": v["index"], "length": v["length"]} for b, v in by_len_old.items()}
    print("buckets:", {b: (v["length"], v["index"]) for b, v in buckets.items()}, flush=True)

    spec = C.MODEL_SPECS[args.model]
    device = torch.device(args.device)
    # ``index`` in the JSON is the position inside the test split; map it back
    # to the cache index.
    meta_last = C.load_cache_meta(C.abs_path(spec["caches"]["last"].format(task=args.task)))
    all_idx = C.split_test_indices(meta_last["num_train"], meta_last["num_test"],
                                   C.TASK_TARGET_EVAL[args.task])
    idxs = [all_idx[buckets[b]["index"]] for b in ("short", "median", "long")]
    data = prepare_task_data(spec, args.task, device, indices=idxs)
    assert data["lengths"] == [buckets[b]["length"] for b in ("short", "median", "long")]

    methods_wanted = list(res.get("methods_measured", []))
    methods, prepare_hooks, release_hooks, between, keepalive = build_methods(
        spec, data, device, methods_wanted, seed=int(res["config"].get("seed", 0)))
    method_names = [m for m in methods_wanted if m in methods]

    for bi, bucket in enumerate(("short", "median", "long")):
        j = bi
        for name in method_names:
            if name in prepare_hooks:
                prepare_hooks[name](j)
        for name in method_names:
            with torch.no_grad():
                methods[name](j)
            if name in between:
                between[name]()
        samples: Dict[str, List[float]] = {name: [] for name in method_names}
        for _t in range(args.trials):
            for name in method_names:
                samples[name].append(C.profile_one_call(
                    lambda fn=methods[name], i=j: fn(i))["cuda_us"])
                if name in between:
                    between[name]()
        for name in method_names:
            by_len_old[bucket]["methods"][name] = C.summarize_profile_samples(samples[name])
            if name in release_hooks:
                release_hooks[name](j)
        print(f"  {bucket} n={buckets[bucket]['length']}: "
              + "  ".join(f"{m}={by_len_old[bucket]['methods'][m]['cuda_us_per_call']:.0f}"
                          for m in method_names), flush=True)

    # Refresh the aggregate view used by the report.
    dev_all = res.setdefault("device_time_us", {})
    for name in method_names:
        dev_all[name] = by_len_old["median"]["methods"][name]
    res["repaired_profile"] = True
    res["repair_trials"] = args.trials
    with open(args.json, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print("updated", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
