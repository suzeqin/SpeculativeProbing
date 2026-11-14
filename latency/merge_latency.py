#!/usr/bin/env python3
"""Aggregate the per-run latency JSONs into CSV / Markdown tables.

Usage:
  python latency/merge_latency.py [--results latency/results] [--out latency/results]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

ORDER = ["mlp", "mm", "sp1", "sp2", "sp5", "sp1_inc", "sp2_inc", "sp5_inc",
         "sp1_core", "sp2_core", "sp5_core"]
LABELS = {
    "mlp": "MLP probe",
    "mm": "MultiMax",
    "sp1": "SP-1 (standalone)",
    "sp2": "SP-2 (standalone)",
    "sp5": "SP-5 (standalone)",
    "sp1_inc": "SP-1 (KV-cached)",
    "sp2_inc": "SP-2 (KV-cached)",
    "sp5_inc": "SP-5 (KV-cached)",
    "sp1_core": "SP-1 (no draft lm_head)",
    "sp2_core": "SP-2 (no draft lm_head)",
    "sp5_core": "SP-5 (no draft lm_head)",
}
MODEL_ORDER = ["qwen3.5-4b", "qwen3.5-9b", "minicpm4.1-8b"]
TASK_ORDER = ["verih", "nemotron_safety"]


def ms(v: Optional[float], digits: Optional[int] = None) -> str:
    """Format a value already expressed in milliseconds."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if digits is not None:
        return f"{v:.{digits}f}"
    if v < 0.01:
        return f"{v:.4f}"
    if v < 0.1:
        return f"{v:.3f}"
    return f"{v:.2f}"


def pct(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}%"


def load_runs(results_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name in ("latency_summary.json", "latency_table.json"):
            continue
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            continue
        if d.get("status") != "ok" or "config" not in d:
            continue
        d["_path"] = path
        runs.append(d)
    return runs


def acc_for(d: Dict[str, Any], method: str) -> Optional[float]:
    base = method.replace("_inc", "").replace("_core", "")
    a = d.get("accuracy_reproduced", {}).get(base, {})
    return a.get("accuracy")


def method_metrics(d: Dict[str, Any], method: str) -> Dict[str, Any]:
    prof = d.get("device_time_us_by_length", {}).get("median", {}).get("methods", {})
    dev_min = prof.get(method, {}).get("cuda_us_per_call")
    dev = prof.get(method, {}).get("cuda_us_median", dev_min)
    if dev is None:
        dev = d.get("device_time_us", {}).get(method, {}).get("cuda_us_median")
    if dev_min is None:
        dev_min = d.get("device_time_us", {}).get(method, {}).get("cuda_us_per_call")
    dev_all = d.get("device_time_us", {})
    summ = d.get("summary_us", {}).get("probes", {}).get(method, {})
    pemin = d.get("summary_us", {}).get("per_example_min", {}).get(method, {})
    cpu = d.get("cpu_sync_us", {}).get(method, {})
    pre_dev = prof.get("base_prefill", {}).get("cuda_us_median",
                                                 prof.get("base_prefill", {}).get("cuda_us_per_call"))
    if pre_dev is None:
        pre_dev = dev_all.get("base_prefill", {}).get("cuda_us_median",
                                                      dev_all.get("base_prefill", {}).get("cuda_us_per_call"))
    pre_sum = d.get("summary_us", {}).get("base", {}).get("prefill", {})
    pre_min = d.get("summary_us", {}).get("base_per_example_min", {}).get("prefill", {})
    dec = prof.get("base_decode", {}).get("cuda_us_median",
                                          prof.get("base_decode", {}).get("cuda_us_per_call"))
    if dec is None:
        dec = dev_all.get("base_decode", {}).get("cuda_us_median",
                                                 dev_all.get("base_decode", {}).get("cuda_us_per_call"))
    # fall back: per-token decode event time
    if not dec:
        dec_sum = d.get("decode_us_per_token", {})
        dec = dec_sum.get("median")
    total_pre = sum(x for x in d.get("per_example_base_us", {}).get("prefill", [])
                    if x == x)
    total_probe = sum(x for x in d.get("per_example_us", {}).get(method, []) if x == x)
    return {
        "device_ms": (dev / 1000.0) if dev else None,
        "device_ms_min": (dev_min / 1000.0) if dev_min else None,
        # per-example min, then median over examples: a robust "best-case" latency
        "event_min_ms": (pemin.get("median", float("nan")) / 1000.0) if pemin else None,
        "event_min_global_ms": (summ.get("min", float("nan")) / 1000.0) if summ else None,
        "event_median_ms": (summ.get("median", float("nan")) / 1000.0) if summ else None,
        "event_p10_ms": (summ.get("p10", float("nan")) / 1000.0) if summ else None,
        "event_p90_ms": (summ.get("p90", float("nan")) / 1000.0) if summ else None,
        "prefill_event_min_ms": (pre_min.get("median", float("nan")) / 1000.0) if pre_min else None,
        "prefill_event_median_ms": (pre_sum.get("median", float("nan")) / 1000.0) if pre_sum else None,
        "cpu_sync_ms": (cpu.get("median", float("nan")) / 1000.0) if cpu else None,
        "params": d.get("probe_params", {}).get(method.replace("_inc", "").replace("_core", "")),
        "accuracy": acc_for(d, method),
        "prefill_device_ms": (pre_dev / 1000.0) if pre_dev else None,
        "prefill_event_ms": (pre_sum.get("median", float("nan")) / 1000.0) if pre_sum else None,
        "decode_device_ms": (dec / 1000.0) if dec else None,
        "overhead_pct_device": (100.0 * dev / pre_dev) if (dev and pre_dev) else None,
        "overhead_pct_latency": (100.0 * total_probe / total_pre) if (total_pre and total_probe) else None,
        "overhead_pct_decode": (100.0 * dev / dec) if (dev and dec) else None,
        "peak_mem_mb": d.get("probe_peak_mem_mb", {}).get(method),
        "dev_by_len": {
            b: (
                d.get("device_time_us_by_length", {}).get(b, {}).get("methods", {}).get(method, {})
                .get("cuda_us_median",
                     d.get("device_time_us_by_length", {}).get(b, {}).get("methods", {}).get(method, {})
                     .get("cuda_us_per_call"))
            ) for b in ("short", "median", "long")
        },
        "len_by_len": {
            b: d.get("device_time_us_by_length", {}).get(b, {}).get("length") for b in ("short", "median", "long")
        },
    }


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv

    cols = ["model", "task", "method", "params", "accuracy", "device_ms", "device_ms_min",
            "event_min_ms", "event_median_ms", "event_p10_ms", "event_p90_ms", "cpu_sync_ms",
            "prefill_device_ms", "prefill_event_min_ms", "prefill_event_median_ms",
            "prefill_event_ms", "decode_device_ms", "overhead_pct_device",
            "overhead_pct_latency", "overhead_pct_decode", "peak_mem_mb",
            "dev_short_ms", "dev_median_ms", "dev_long_ms", "len_short", "len_median", "len_long"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({
                **{c: r.get(c) for c in cols if c in r},
                "dev_short_ms": (r["dev_by_len"]["short"] / 1000) if r["dev_by_len"]["short"] else None,
                "dev_median_ms": (r["dev_by_len"]["median"] / 1000) if r["dev_by_len"]["median"] else None,
                "dev_long_ms": (r["dev_by_len"]["long"] / 1000) if r["dev_by_len"]["long"] else None,
                "len_short": r["len_by_len"]["short"],
                "len_median": r["len_by_len"]["median"],
                "len_long": r["len_by_len"]["long"],
            })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = load_runs(args.results)
    if not runs:
        print("no result JSONs found")
        return 1
    runs.sort(key=lambda d: (MODEL_ORDER.index(d["config"]["model"])
                             if d["config"]["model"] in MODEL_ORDER else 99,
                             TASK_ORDER.index(d["config"]["task"])
                             if d["config"]["task"] in TASK_ORDER else 99))

    rows: List[Dict[str, Any]] = []
    for d in runs:
        model = d["config"]["model"]
        task = d["config"]["task"]
        for method in ORDER:
            if method in d.get("methods_measured", []):
                m = method_metrics(d, method)
                rows.append({"model": model, "task": task, "method": method, **m})
    write_csv(rows, os.path.join(args.out, "latency_summary.csv"))
    with open(os.path.join(args.out, "latency_summary.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    lines: List[str] = []
    lines.append("# Speculative-Probing inference-time overhead (measured)\n")
    lines.append("Latency measured on the reproduction checkpoints at batch size 1 "
                 "(bf16, PyTorch eager, one NVIDIA RTX A6000 shared with other tenants). "
                 "**Latency** = median over test examples of the per-example minimum CUDA-event "
                 "window over 5 repetitions (best-case per-call latency, suppresses co-tenant "
                 "bursts). **GPU-kernel time** = summed CUDA kernel durations from "
                 "torch.profiler (median-length example); for multi-step KV-cached probes this "
                 "profiler number is inflated by CUPTI instrumentation, so latency is the "
                 "headline. `KV-cached` = the paper's deployment: the draft head's prefix KV "
                 "cache is already resident and only the k soft-prompt positions are evaluated.\n")

    for d in runs:
        model = d["config"]["model"]
        task = d["config"]["task"]
        disp = d["config"]["model_display"]
        n_ex = d.get("num_examples")
        lens = d.get("lengths") or []
        lines.append(f"## {disp} — {task}\n")
        if lens:
            lines.append(f"{n_ex} test examples, {min(lens)}–{max(lens)} tokens "
                         f"(mean {statistics.mean(lens):.0f}).\n")
        by_method = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        pre_dev = next((r["prefill_device_ms"] for r in by_method.values() if r["prefill_device_ms"]), None)
        pre_ev = next((r["prefill_event_ms"] for r in by_method.values() if r["prefill_event_ms"]), None)
        dec_dev = next((r["decode_device_ms"] for r in by_method.values() if r["decode_device_ms"]), None)
        if pre_dev:
            pre_ev_min = next((r["prefill_event_min_ms"] for r in by_method.values() if r.get("prefill_event_min_ms")), None)
        lines.append(f"Reference: base prefill **{pre_ev_min:.2f} ms** measured latency; "
                     f"base decode **{ms(dec_dev)} ms** per token; "
                     f"prefill GPU-kernel time {ms(pre_dev)} ms.\n")
        lines.append("| Method | Params | Latency (ms) | Median latency (ms) | % of prefill (latency) | % of prefill (GPU) | GPU-kernel time (ms) | Accuracy |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for method in ORDER:
            r = by_method.get(method)
            if not r:
                continue
            params = r.get("params")
            if params is None:
                pstr = "—"
            elif params < 1e6:
                pstr = f"{params/1e3:.1f}K"
            else:
                pstr = f"{params/1e6:.2f}M"
            acc = "—" if r.get("accuracy") is None else f"{100*r['accuracy']:.1f}"
            dec_pct = r.get("overhead_pct_decode")
            if dec_pct is None and r.get("decode_device_ms") and r.get("event_min_ms"):
                dec_pct = 100.0 * r["event_min_ms"] / r["decode_device_ms"]
            lines.append(
                f"| {LABELS.get(method, method)} | {pstr} | "
                f"{ms(r['event_min_ms'])} | {ms(r['event_median_ms'])} | "
                f"{pct(r['overhead_pct_latency'])} | "
                f"{pct(r['overhead_pct_device'])} | "
                f"{ms(r['device_ms'])} | {acc} |")
        lines.append("")

    # length buckets from the per-example measured latencies (robust: many samples)
    lines.append("## Measured latency vs sequence length (median per-example latency, ms)\n")
    lines.append("Test examples split into length terciles; each cell is the median over "
                 "examples in the bucket of the per-example best-of-5 latency.\n")
    hdr_methods = ["mlp", "mm", "sp1", "sp2", "sp5", "sp1_inc", "sp2_inc", "sp5_inc"]
    lines.append("| Model | Task | bucket | n tokens | n examples | prefill | "
                 + " | ".join(LABELS[m] for m in hdr_methods) + " |")
    lines.append("|---|---|---|---:|---:|---:|" + "---:|" * len(hdr_methods))
    for d in runs:
        lens = d.get("lengths") or []
        pe = d.get("per_example_us") or {}
        base = (d.get("per_example_base_us") or {}).get("prefill") or []
        if not lens or not pe:
            continue
        order = sorted(range(len(lens)), key=lambda k: lens[k])
        n = len(order)
        cuts = [int(0.333 * (n - 1)), int(0.667 * (n - 1))]
        groups = {"short": order[:cuts[0] + 1],
                  "mid": order[cuts[0] + 1:cuts[1] + 1],
                  "long": order[cuts[1] + 1:]}
        model, task = d["config"]["model"], d["config"]["task"]
        for bname, idx in groups.items():
            if not idx:
                continue
            med_len = statistics.median([lens[i] for i in idx])
            cells = []
            for m in hdr_methods:
                vals = [pe[m][i] for i in idx if m in pe and pe[m][i] == pe[m][i]]
                cells.append(ms(statistics.median(vals) / 1000.0) if vals else "—")
            bvals = [base[i] for i in idx if i < len(base) and base[i] == base[i]]
            lines.append(f"| {d['config']['model_display']} | {task} | {bname} | {med_len:.0f} | {len(idx)} | "
                         f"{ms(statistics.median(bvals)/1000.0) if bvals else '—'} | " + " | ".join(cells) + " |")
    lines.append("")

    # compact cross-model headline tables
    lines.append("## Compact GPU-time table (ms per example)\n")
    lines.append("| Model | Task | prefill | MLP | MM | SP-1 | SP-2 | SP-5 | SP-1 KV | SP-2 KV | SP-5 KV |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        pre = next((r["prefill_device_ms"] for r in by.values() if r["prefill_device_ms"]), None)
        cells = []
        for m in ("mlp", "mm", "sp1", "sp2", "sp5", "sp1_inc", "sp2_inc", "sp5_inc"):
            cells.append(ms(by.get(m, {}).get("device_ms")))
        lines.append(f"| {d['config']['model_display']} | {task} | {ms(pre)} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Compact overhead table (% of base prefill GPU time)\n")
    lines.append("| Model | Task | MLP | MM | SP-1 | SP-2 | SP-5 | SP-1 KV | SP-2 KV | SP-5 KV |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        cells = [pct(by.get(m, {}).get("overhead_pct_device")) for m in
                 ("mlp", "mm", "sp1", "sp2", "sp5", "sp1_inc", "sp2_inc", "sp5_inc")]
        lines.append(f"| {d['config']['model_display']} | {task} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Compact accuracy table (%)\n")
    lines.append("| Model | Task | MLP | MM | SP-1 | SP-2 | SP-5 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        cells = []
        for m in ("mlp", "mm", "sp1", "sp2", "sp5"):
            a = by.get(m, {}).get("accuracy")
            cells.append("—" if a is None else f"{100*a:.1f}")
        lines.append(f"| {d['config']['model_display']} | {task} | " + " | ".join(cells) + " |")
    lines.append("")

    # length scaling table
    lines.append("## Scaling with sequence length (GPU time, ms)\n")
    lines.append("Test-set 10th / 50th / 90th percentile lengths.\n")
    lines.append("| Model | Task | Method | short | median | long |")
    lines.append("|---|---|---|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        for method in ORDER:
            if method not in d.get("methods_measured", []):
                continue
            r = next((x for x in rows if x["model"] == model and x["task"] == task and x["method"] == method), None)
            if not r:
                continue
            lens = r["len_by_len"]
            vals = [r["dev_by_len"][b] for b in ("short", "median", "long")]
            vals = [(v / 1000.0) if v else None for v in vals]
            lines.append(f"| {d['config']['model_display']} | {task} | {LABELS.get(method, method)} "
                         f"| {ms(vals[0])} (n={lens['short']}) | {ms(vals[1])} (n={lens['median']}) "
                         f"| {ms(vals[2])} (n={lens['long']}) |")
    lines.append("")

    # end-to-end
    lines.append("## End-to-end request latency (prompt classification, eager latency)\n")
    lines.append("A request that only needs the prompt classified pays `prefill + probe`; "
                 "a request that also generates 128 tokens pays `prefill + 128 · decode + probe` "
                 "(the probe overlaps nothing in this eager setup, so it is additive).\n")
    lines.append("| Model | Task | Baseline prefill (ms) | +MLP | +MM | +SP-1 | +SP-2 | +SP-5 | "
                 "Baseline 128-tok gen (ms) | +SP-5 KV-cached |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        pre = next((r["prefill_event_min_ms"] for r in by.values() if r.get("prefill_event_min_ms")), None)
        dec = next((r["decode_device_ms"] for r in by.values() if r["decode_device_ms"]), None)
        if not pre:
            continue

        def add(m):
            r = by.get(m)
            return f"{pre + r['event_min_ms']:.2f}" if r and r["event_min_ms"] == r["event_min_ms"] else "—"

        gen = pre + 128 * dec if dec else None
        sp5i = by.get("sp5_inc", {}).get("event_min_ms")
        lines.append(f"| {d['config']['model_display']} | {task} | {pre:.2f} | {add('mlp')} | {add('mm')} | "
                     f"{add('sp1')} | {add('sp2')} | {add('sp5')} | "
                     f"{gen:.2f} | " + (f"{gen + sp5i:.2f}" if (gen and sp5i == sp5i) else "—") + " |")
    lines.append("")

    # GPU-time end-to-end scenario (device kernels only)
    lines.append("## End-to-end GPU-time overhead (base prefill + 128 decode tokens)\n")
    lines.append("Prompt-classification request = `prefill + probe`; "
                 "monitoring a 128-token generation = `prefill + 128·decode + probe`. "
                 "All terms are measured CUDA-kernel time for the median-length test example.\n")
    lines.append("| Model | Task | prefill (ms) | 128-token gen (ms) | SP-1 KV (ms) | SP-5 KV (ms) | "
                 "overhead adding SP-1 KV | overhead adding SP-5 KV |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        pre = next((r["prefill_device_ms"] for r in by.values() if r["prefill_device_ms"]), None)
        dec = next((r["decode_device_ms"] for r in by.values() if r["decode_device_ms"]), None)
        if not pre or not dec:
            continue
        gen = pre + 128 * dec
        s1 = by.get("sp1_inc", {}).get("device_ms")
        s5 = by.get("sp5_inc", {}).get("device_ms")
        o1 = f"{100*s1/(gen+s1):.3f}%" if s1 else "—"
        o5 = f"{100*s5/(gen+s5):.3f}%" if s5 else "—"
        lines.append(f"| {d['config']['model_display']} | {task} | {pre:.2f} | {gen:.2f} | "
                     f"{ms(s1)} | {ms(s5)} | {o1} | {o5} |")
    lines.append("")

    # per-step decomposition of SP-k (measured latency)
    lines.append("## SP-k cost decomposition (KV-cached, measured latency)\n")
    lines.append("Per-step cost = (latency(SP-k) − latency(SP-1)) / (k−1): the marginal cost of one "
                 "additional soft-prompt position through the draft head.\n")
    lines.append("| Model | Task | SP-1 (ms) | SP-2 (ms) | SP-5 (ms) | per extra step (ms) | "
                 "fraction of a base prefill per step |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for d in runs:
        model, task = d["config"]["model"], d["config"]["task"]
        by = {r["method"]: r for r in rows if r["model"] == model and r["task"] == task}
        v = {k: by.get(k, {}).get("event_min_ms") for k in ("sp1_inc", "sp2_inc", "sp5_inc")}
        if not all(v.values()):
            continue
        pre = next((r["prefill_event_min_ms"] for r in by.values() if r.get("prefill_event_min_ms")), None)
        per = (v["sp5_inc"] - v["sp1_inc"]) / 4.0
        lines.append(f"| {d['config']['model_display']} | {task} | {v['sp1_inc']:.2f} | "
                     f"{v['sp2_inc']:.2f} | {v['sp5_inc']:.2f} | {per:.2f} | "
                     f"{(100*per/pre):.2f}% |" if pre else
                     f"| {d['config']['model_display']} | {task} | {v['sp1_inc']:.2f} | "
                     f"{v['sp2_inc']:.2f} | {v['sp5_inc']:.2f} | {per:.2f} | — |")
    lines.append("")

    with open(os.path.join(args.out, "LATENCY_TABLE.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:60]))
    print(f"\nwrote {os.path.join(args.out, 'latency_summary.csv')}")
    print(f"wrote {os.path.join(args.out, 'LATENCY_TABLE.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
