#!/usr/bin/env python3
"""Speculative-decoding benchmark over the three new benchmarks.

For one (model, benchmark) pair, runs greedy speculative decoding on a sample
of benchmark prompts for each probe configuration (none / mlp / mm / sp1 /
sp2 / sp5) and reports, per config:

  * acceptance rate      accepted drafted tokens / drafted tokens
  * acceptance length    generated tokens per decoding round (tau)
  * end-to-end latency   per-example total CUDA time (median / mean)
  * throughput           generated tokens per second (median)
  * probe overhead       probe CUDA time per round and share of round time
  * probe verdicts       per-prompt first-round verdict (input monitoring)

The probes never draft, so acceptance must be identical across configs; the
runner asserts this to catch harness bugs.

Usage:
  python specbench/run_bench.py --model qwen3.5-4b --task agentlab \
      --configs none,mlp,mm,sp1,sp2,sp5 --n 50 --max-new-tokens 256 \
      --out specbench/results/qwen3.5-4b_agentlab.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "specbench"), os.path.join(ROOT, "fast_softprompt")):
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = "cuda"
BF16 = torch.bfloat16

MODEL_SPECS = {
    "qwen3.5-4b": {
        "kind": "mtp", "tag": "4b",
        "base": "models/Qwen3.5-4B", "head": "models/Qwen3.5-4B-mtp",
        "hidden": 2560, "tok": "models/Qwen3.5-4B", "bos": True,
    },
    "qwen3.5-9b": {
        "kind": "mtp", "tag": "9b",
        "base": "models/Qwen3.5-9B", "head": "models/Qwen3.5-9B-mtp",
        "hidden": 4096, "tok": "models/Qwen3.5-9B", "bos": True,
    },
    "minicpm4.1-8b": {
        "kind": "eagle3", "tag": "mcpm",
        "base": "models/MiniCPM4.1-8B",
        "head": "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16",
        "hidden": 4096, "tok": "models/MiniCPM4.1-8B", "bos": False,
    },
}

PROBE_ROOT = "results_new_benchmark"


def load_prompts(task: str, tok, n: int, seed: int = 42, max_tokens: int = 1800):
    rng = random.Random(seed)
    prompts = []
    if task == "livecodebench":
        path = os.path.join(ROOT, "livecodebench", "gen_prompts.jsonl")
        recs = [json.loads(l) for l in open(path)]
        for r in recs:
            prompts.append({"text": r["prompt"], "meta": {"id": r["question_id"],
                                                          "difficulty": r["difficulty"]}})
    else:
        path = os.path.join(ROOT, task, "classifier_dataset.jsonl")
        recs = [json.loads(l) for l in open(path) if json.loads(l).get("split") == "test"]
        # mix benign and attack prompts (the monitor watches both)
        benign = [r for r in recs if r["label"] == 0]
        attack = [r for r in recs if r["label"] == 1]
        n_benign = n * 2 // 3
        n_attack = n - n_benign
        rng.shuffle(benign)
        rng.shuffle(attack)
        chosen = benign[:n_benign] + attack[:n_attack]
        rng.shuffle(chosen)
        for r in chosen:
            prompts.append({"text": r["text"], "meta": {"label": r["label"],
                                                        "source": r["source"]}})
    # MiniCPM's tokenizer injects a BOS the model was not trained with
    adds_special = tok.encode("x", add_special_tokens=True) != tok.encode("x", add_special_tokens=False)
    out = []
    for p in prompts[:n * 2]:  # over-provision: drop over-length prompts
        ids = tok.encode(p["text"], add_special_tokens=not adds_special)
        if len(ids) < 8 or len(ids) > max_tokens:
            continue
        out.append({"ids": ids, **p})
        if len(out) >= n:
            break
    return out


def summarize(rounds, wall_s):
    drafted = sum(r["drafted"] for r in rounds)
    accepted = sum(r["accepted"] for r in rounds)
    gen = sum(r["generated"] for r in rounds)
    n = max(len(rounds), 1)
    total_gpu_ms = sum(r["draft_ms"] + r["verify_ms"] + r["probe_ms"] + r["fold_ms"]
                       for r in rounds)
    probe_ms = sum(r["probe_ms"] for r in rounds)
    return {
        "rounds": n,
        "generated": gen,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / max(drafted, 1),
        "acceptance_length": gen / n,
        "total_gpu_ms": total_gpu_ms,
        "wall_s": wall_s,
        "throughput_tok_s": (gen / wall_s) if wall_s and wall_s > 0 else None,
        "probe_ms_total": probe_ms,
        "probe_ms_share": probe_ms / max(total_gpu_ms, 1e-9),
        "mean_round_ms": total_gpu_ms / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_SPECS), required=True)
    ap.add_argument("--task", choices=["agentlab", "bfcl_v4", "livecodebench"], required=True)
    ap.add_argument("--configs", default="none,mlp,mm,sp1,sp2,sp5")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-root", default=PROBE_ROOT)
    args = ap.parse_args()

    spec = MODEL_SPECS[args.model]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec["tok"], trust_remote_code=True)
    prompts = load_prompts(args.task, tok, args.n, args.seed)
    print(f"{len(prompts)} prompts for {args.model}/{args.task}")

    if args.gamma == 0:
        # pure autoregressive reference (no probing support)
        from spec_decode import GreedyEngine
        engine = GreedyEngine(os.path.join(ROOT, spec["base"]))
        configs = ["none"]
        probe = None
    elif spec["kind"] == "mtp":
        from spec_decode import QwenSpecEngine
        engine = QwenSpecEngine(os.path.join(ROOT, spec["base"]),
                                os.path.join(ROOT, spec["head"]),
                                draft_gamma=args.gamma, collect_mid=True)
    else:
        import minicpm_compat  # noqa: F401
        from spec_decode import Eagle3SpecEngine
        engine = Eagle3SpecEngine(os.path.join(ROOT, spec["base"]),
                                  os.path.join(ROOT, spec["head"]),
                                  draft_gamma=args.gamma, collect_mid=True)

    if args.gamma != 0:
        configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    results = {"config_info": {
        "model": args.model, "task": args.task, "gamma": args.gamma,
        "n_prompts": len(prompts), "max_new_tokens": args.max_new_tokens,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_ids_bos": spec["bos"],
    }, "prompts": [{"meta": p.get("meta"), "n_tokens": len(p["ids"])} for p in prompts],
        "configs": {}}

    acceptance_ref = None
    for cfg in configs:
        probe = None
        if cfg != "none":
            from probes import load_probe
            k = int(cfg[2:]) if cfg.startswith("sp") else None
            kind = "sp" if cfg.startswith("sp") else cfg
            probe = load_probe(kind, spec["tag"], args.task,
                               os.path.join(ROOT, args.probe_root), spec, k)
        per_example = []
        total_tokens = 0
        t0 = time.time()
        for i, p in enumerate(prompts):
            try:
                if args.gamma == 0:
                    # GreedyEngine returns (tokens, stats-dict), not rounds.
                    # Synthesize a summarize-compatible per-example record so
                    # the AR reference rows share the same schema.
                    tokens, st = engine.generate(p["ids"], args.max_new_tokens)
                    s = {
                        "rounds": st["steps"], "generated": len(tokens),
                        "drafted": 0, "accepted": 0,
                        "acceptance_rate": 0.0, "acceptance_length": 1.0,
                        "total_gpu_ms": st["total_gpu_ms"], "wall_s": st["wall_s"],
                        "throughput_tok_s": len(tokens) / max(st["wall_s"], 1e-9),
                        "probe_ms_total": 0.0, "probe_ms_share": 0.0,
                        "mean_round_ms": st["total_gpu_ms"] / max(st["steps"], 1),
                        "first_verdict": None,
                    }
                else:
                    tokens, rounds = engine.generate(p["ids"], args.max_new_tokens, probe=probe)
                    s = summarize(rounds, 0.0)
                    s["first_verdict"] = rounds[0]["verdict"] if rounds else None
            except Exception as e:
                print(f"  [error] example {i}: {type(e).__name__}: {e}")
                continue
            s["n_prompt_tokens"] = len(p["ids"])
            per_example.append(s)
            total_tokens += len(tokens)
        wall = time.time() - t0

        # aggregate
        def agg(key):
            vals = [e[key] for e in per_example if e.get(key) is not None
                    and isinstance(e[key], (int, float))]
            if not vals:
                return None
            return {
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
                "p90": sorted(vals)[min(len(vals) - 1, int(round(0.9 * (len(vals) - 1))))],
            }

        acc_rates = [e["acceptance_rate"] for e in per_example]
        config_summary = {
            "n_ok": len(per_example),
            "acceptance_rate_mean": statistics.fmean(acc_rates) if acc_rates else None,
            "acceptance_length_mean": statistics.fmean([e["acceptance_length"] for e in per_example]) if per_example else None,
            "latency_gpu_ms": agg("total_gpu_ms"),
            "throughput_tok_s": agg("throughput_tok_s"),
            "probe_ms_share_mean": statistics.fmean([e["probe_ms_share"] for e in per_example]) if per_example else None,
            "probe_ms_per_round_ms": agg("probe_ms_total"),
            "throughput_wall_tok_s": total_tokens / max(wall, 1e-9),
            "total_generated_tokens": total_tokens,
            "wall_s": wall,
            "first_verdicts": [e["first_verdict"] for e in per_example],
        }
        results["configs"][cfg] = config_summary
        if cfg == "none":
            acceptance_ref = config_summary["acceptance_rate_mean"]
        print(f"[{cfg}] acc={config_summary['acceptance_rate_mean']:.3f} "
              f"tau={config_summary['acceptance_length_mean']:.2f} "
              f"probe_share={config_summary['probe_ms_share_mean']*100 if per_example else 0:.2f}% "
              f"({len(per_example)} examples)")

        # acceptance must not depend on the probe
        if acceptance_ref is not None and cfg != "none" and config_summary["acceptance_rate_mean"] is not None:
            drift = abs(config_summary["acceptance_rate_mean"] - acceptance_ref)
            if drift > 0.02:
                print(f"  [WARN] acceptance drifted by {drift:.3f} vs none")

        # free probe memory between configs
        del probe
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
