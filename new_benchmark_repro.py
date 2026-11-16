#!/usr/bin/env python3
"""One-click training of the 45 new-benchmark experiments.

3 benchmarks (agentlab / bfcl_v4 / livecodebench)
  x 3 models (Qwen3.5-4B, Qwen3.5-9B, MiniCPM4.1-8B)
  x 5 configs (SP-1, SP-2, SP-5, MLP, MM)

Protocol mirrors reproduce_all.py (the verified reproduction runner):
  * Qwen SP: recursive soft prompts + linear head, lr 3e-4 -> 1e-4, 10 epochs,
    batch 1, dropout 0.2.
  * MLP: hidden_dim 2048, lr 5e-4, batch 32, 10 epochs.
  * MultiMax: H=10 heads, width 100, mid-layer cache (4B layer 8 / 9B / MiniCPM 16),
    lr 5e-4, batch 32, 10 epochs.
  * MiniCPM SP: Eagle3 head, same SP hyperparameters.
  * target_eval: agentlab 800, bfcl_v4 800, livecodebench 600.

Usage:
  python new_benchmark_repro.py --gpus 4,5,6,7 --kinds qwen --tasks agentlab
  python new_benchmark_repro.py --gpus 7 --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TASKS = {"agentlab": 800, "bfcl_v4": 800, "livecodebench": 600}
QWEN = {"4b": {"name": "Qwen3.5-4B", "mid": 8}, "9b": {"name": "Qwen3.5-9B", "mid": 16}}
MC_BASE = "models/MiniCPM4.1-8B"
MC_E3 = "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16"
METHODS = ["sp5", "sp2", "sp1", "mlp", "multimax"]  # longest first


@dataclass
class Job:
    label: str
    kind: str
    size: str
    task: str
    method: str
    k: int | None
    cmd: list[str]
    out_dir: Path
    log_path: Path


def qwen_main(size: str, task: str) -> Path:
    return ROOT / f"cache_{task}_{size}.pt"


def qwen_mid(size: str, task: str) -> Path:
    if size == "4b":
        return ROOT / f"cache_{task}_4b_mid8.pt"
    return ROOT / f"cache_{task}_9b_mid.pt"


def mc_cache(task: str, suffix: str) -> Path:
    return ROOT / f"cache_{task}_mcpm{suffix}.pt"


def build_jobs(out: Path, python: str, combos, sizes, tasks, methods) -> list[Job]:
    jobs = []
    for method in METHODS:
        if method not in methods:
            continue
        for kind, size, task in combos:
            if size not in sizes or task not in tasks:
                continue
            target = TASKS[task]
            out_dir = out / f"{task}_{size}" / method
            k = int(method[2:]) if method.startswith("sp") else None

            if kind == "qwen":
                main = qwen_main(size, task)
                mid = qwen_mid(size, task)
                name = QWEN[size]["name"]
                if method.startswith("sp"):
                    cmd = [
                        python, "fast_softprompt/train_softprompt.py",
                        "--cache", str(main), "--mmap_cache",
                        "--mtp_dir", f"models/{name}-mtp",
                        "--output_dir", str(out_dir),
                        "--soft_prompt_length", str(k), "--recursive",
                        "--classifier_head", "linear",
                        "--lr", "3e-4", "--min_lr", "1e-4",
                        "--epochs", "10", "--batch_size", "1",
                        "--dropout", "0.2", "--target_eval", str(target),
                    ]
                elif method == "mlp":
                    cmd = [
                        python, "fast_softprompt/train_mlp.py",
                        "--cache", str(main), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--hidden_dim", "2048",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]
                else:
                    cmd = [
                        python, "fast_softprompt/train_multimax.py",
                        "--cache", str(mid), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--num_heads", "10", "--mlp_width", "100",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]
            else:
                if method.startswith("sp"):
                    cmd = [
                        python, "fast_softprompt/train_eagle3_minicpm.py",
                        "--cache", str(mc_cache(task, "_eagle3")), "--mmap_cache",
                        "--eagle3_dir", MC_E3, "--base_model", MC_BASE,
                        "--output_dir", str(out_dir),
                        "--soft_prompt_length", str(k),
                        "--epochs", "10", "--batch_size", "1",
                        "--dropout", "0.2", "--target_eval", str(target),
                    ]
                elif method == "mlp":
                    cmd = [
                        python, "fast_softprompt/train_mlp.py",
                        "--cache", str(mc_cache(task, "")), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--hidden_dim", "2048",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]
                else:
                    cmd = [
                        python, "fast_softprompt/train_multimax.py",
                        "--cache", str(mc_cache(task, "_mid")), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--num_heads", "10", "--mlp_width", "100",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]

            label = f"{method}_{task}_{size}"
            jobs.append(Job(label, kind, size, task, method, k, cmd, out_dir,
                            out / "logs" / f"{label}.log"))
    return jobs


def check_assets(combos, sizes, tasks) -> list[str]:
    errors = []
    for kind, size, task in combos:
        if size not in sizes or task not in tasks:
            continue
        if kind == "qwen":
            for p in (qwen_main(size, task), qwen_mid(size, task)):
                if not p.exists():
                    errors.append(f"missing cache: {p}")
                elif not Path(str(p) + ".hidden").exists():
                    errors.append(f"missing hidden sidecar: {p}.hidden")
        else:
            for p in (mc_cache(task, ""), mc_cache(task, "_mid"), mc_cache(task, "_eagle3")):
                if not p.exists():
                    errors.append(f"missing cache: {p}")
                elif not Path(str(p) + ".hidden").exists():
                    errors.append(f"missing hidden sidecar: {p}.hidden")
            for d in (MC_BASE, MC_E3):
                if not (ROOT / d).exists():
                    errors.append(f"missing model dir: {d}")
    return errors


def run_job(job: Job, gpu: int) -> tuple[bool, float, str]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    if job.kind == "minicpm":
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    t0 = time.time()
    with open(job.log_path, "w") as log:
        proc = subprocess.run(job.cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    return proc.returncode == 0, time.time() - t0, str(job.log_path)


def verify(job: Job) -> dict:
    result = {
        "label": job.label, "size": job.size, "task": job.task,
        "method": job.method, "status": "FAIL", "reason": "",
        "acc": None, "f1": None,
    }
    path = job.out_dir / "final_results.json"
    if not path.exists():
        result["reason"] = f"missing {path}"
        return result
    try:
        d = json.load(open(path))
    except Exception as e:
        result["reason"] = f"invalid JSON: {e}"
        return result

    cfg = d.get("config", {})
    metrics = d.get("metrics", {})
    history = d.get("training_history", [])
    acc, f1 = metrics.get("accuracy"), metrics.get("f1")
    result["acc"], result["f1"] = acc, f1
    problems = []
    if not isinstance(acc, (int, float)) or not math.isfinite(acc) or not (0 <= acc <= 1):
        problems.append(f"bad accuracy={acc}")
    if not isinstance(f1, (int, float)) or not math.isfinite(f1) or not (0 <= f1 <= 1):
        problems.append(f"bad f1={f1}")
    if not history:
        problems.append("empty training_history")
    if job.method.startswith("sp"):
        if cfg.get("soft_prompt_length") != job.k:
            problems.append(f"soft_prompt_length={cfg.get('soft_prompt_length')} != {job.k}")
        if job.kind == "qwen":
            if cfg.get("recursive") is not True:
                problems.append("recursive is not True")
            if cfg.get("classifier_head") != "linear":
                problems.append(f"classifier_head={cfg.get('classifier_head')}")
    elif job.method == "mlp":
        if cfg.get("mlp_hidden") != 2048:
            problems.append(f"mlp_hidden={cfg.get('mlp_hidden')} != 2048")
    else:
        if cfg.get("num_heads") != 10 or cfg.get("mlp_width") != 100:
            problems.append(f"MultiMax config mismatch")
    if problems:
        result["status"] = "FAIL"
        result["reason"] = "; ".join(problems)
    else:
        result["status"] = "PASS"
    return result


def write_summary(out: Path, results: list[dict]):
    lines = [
        "# New-benchmark training results",
        "",
        "| Size | Task | Method | Acc | F1 | Status | Notes |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["task"], x["size"], x["method"])):
        acc = f"{r['acc']*100:.2f}" if isinstance(r["acc"], (int, float)) else "—"
        f1 = f"{r['f1']*100:.2f}" if isinstance(r["f1"], (int, float)) else "—"
        lines.append(
            f"| {r['size']} | {r['task']} | {r['method']} | {acc} | {f1} "
            f"| {r['status']} | {r['reason']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_new_benchmark")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "4,5,6,7"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--sizes", default="4b,9b,mcpm")
    ap.add_argument("--tasks", default="agentlab,bfcl_v4,livecodebench")
    ap.add_argument("--methods", default="sp1,sp2,sp5,mlp,multimax")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        sys.exit(f"ERROR: current Python cannot import torch ({e}).")

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for s in sizes:
        if s not in ("4b", "9b", "mcpm"):
            sys.exit(f"unknown size: {s}")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"unknown task: {t}")
    for m in methods:
        if m not in METHODS:
            sys.exit(f"unknown method: {m}")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        sys.exit("no GPUs specified")
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    combos = [("qwen", "4b", t) for t in TASKS] + [("qwen", "9b", t) for t in TASKS] \
        + [("minicpm", "mcpm", t) for t in TASKS]
    jobs = build_jobs(out, args.python, combos, sizes, tasks, methods)

    # skip jobs that already have results
    pending = []
    for job in jobs:
        if (job.out_dir / "final_results.json").exists():
            print(f"[skip] {job.label} (final_results.json exists)")
            continue
        pending.append(job)
    print(f"{len(pending)}/{len(jobs)} jobs to run on GPUs {gpus}")
    if args.dry_run:
        for job in pending:
            print(f"  {job.label:28s} -> {job.out_dir}")
        return

    missing = check_assets(combos, sizes, tasks)
    if missing:
        for m in missing:
            print(f"[asset ERROR] {m}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)
    job_q = queue.Queue()
    for job in pending:
        job_q.put(job)
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    results = []
    lock = threading.Lock()

    def worker():
        while True:
            try:
                job = job_q.get_nowait()
            except queue.Empty:
                return
            gpu = gpu_q.get()
            try:
                print(f"[start] {job.label} on GPU {gpu}", flush=True)
                ok, elapsed, log = run_job(job, gpu)
                print(f"[{'done' if ok else 'FAIL'}] {job.label} "
                      f"({elapsed/60:.1f} min) log={log}", flush=True)
                with lock:
                    results.append({"label": job.label, "ok": ok,
                                    "elapsed_min": elapsed, "log": log})
            finally:
                gpu_q.put(gpu)
                job_q.task_done()

    threads = [threading.Thread(target=worker) for _ in range(len(gpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    verified = [verify(job) for job in pending]
    write_summary(out, verified)
    failed = [r for r in verified if r["status"] == "FAIL"]
    print(f"\n{len(verified) - len(failed)} PASS, {len(failed)} FAIL")
    if failed:
        for r in failed:
            print(f"  FAIL {r['label']}: {r['reason']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
