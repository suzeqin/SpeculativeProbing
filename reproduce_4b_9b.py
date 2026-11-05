#!/usr/bin/env python3
"""Rerun all 4B/9B experiments with the paper protocol and verify the results.

Protocol:
  * SP-1/2/5 : main hidden-state cache, paper hyperparameters.
  * MLP      : main hidden-state cache, hidden_dim=2048 (paper Appendix B).
  * MultiMax : _mid cache at the paper mid layer (4B=8, 9B=16), H=10, width=100.

Outputs go to --out (default: results_verified/) so the existing results are
not overwritten. Each run is checked for:
  * final_results.json exists and is valid;
  * no NaN in training history;
  * method-specific config (k / hidden_dim / heads+width);
  * accuracy / F1 in [0, 1];
  * accuracy within --tolerance pp of the reference (default: results_strict/).

Usage:
  bash run_4b_9b_all.sh --gpus 4,5,6,7
  bash run_4b_9b_all.sh --dry-run
  bash run_4b_9b_all.sh --sizes 4b --tasks verih --methods mlp --out /tmp/smoke
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
TASKS = {"verih": 645, "nemotron_safety": 800}
SIZES = {
    "4b": {
        "model": "models/Qwen3.5-4B",
        "mtp": "models/Qwen3.5-4B-mtp",
        "mid_layer": 8,
    },
    "9b": {
        "model": "models/Qwen3.5-9B",
        "mtp": "models/Qwen3.5-9B-mtp",
        "mid_layer": 16,
    },
}
METHODS = ["sp5", "sp2", "sp1", "mlp", "multimax"]  # longest first


@dataclass
class Job:
    label: str
    size: str
    task: str
    method: str
    k: int | None
    cmd: list[str]
    out_dir: Path
    log_path: Path


def main_cache(size: str, task: str) -> Path:
    return ROOT / f"cache_{task}_{size}.pt"


def mid_cache(size: str, task: str) -> Path:
    # Existing naming: 9B uses cache_<task>_9b_mid.pt (layer 16);
    # 4B's paper-protocol layer-8 cache is cache_<task>_4b_mid8.pt.
    if size == "4b":
        return ROOT / f"cache_{task}_4b_mid8.pt"
    return ROOT / f"cache_{task}_9b_mid.pt"


def load_meta(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def check_caches(sizes, tasks, need_mid, build: bool, python: str) -> list[str]:
    """Return a list of fatal errors. Optionally build missing caches."""
    errors = []
    missing_main = []
    missing_mid = []
    for size in sizes:
        for task in tasks:
            main = main_cache(size, task)
            if not main.exists():
                missing_main.append((size, task))
                errors.append(f"missing main cache: {main}")
            else:
                meta = load_meta(main)
                if int(meta.get("num_train", 0)) <= 0 or int(meta.get("num_test", 0)) <= 0:
                    errors.append(f"bad main cache metadata: {main}")
            if need_mid:
                mid = mid_cache(size, task)
                if not mid.exists():
                    missing_mid.append((size, task))
                    errors.append(f"missing mid cache: {mid}")
                else:
                    meta = load_meta(mid)
                    expected = SIZES[size]["mid_layer"]
                    if int(meta.get("layer", -999)) != expected:
                        errors.append(
                            f"mid cache {mid} has layer={meta.get('layer')}, expected {expected}")

    if build and (missing_main or missing_mid):
        print("[caches] building missing caches (this can take hours)...")
        print("[caches] WARNING: a rebuilt main cache may use a different data "
              "snapshot/split than results_strict; comparisons may then fail.")
        for size, task in missing_main:
            cmd = [
                python, "fast_softprompt/precompute_hidden_states.py",
                "--model", SIZES[size]["model"], "--task", task,
                "--output", str(main_cache(size, task)),
                "--num_train", "10000", "--num_test", str(TASKS[task]),
                "--max_length", "2048", "--batch_size", "4", "--mid_layer", "16",
            ]
            print("[caches] ", " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
        # A freshly built main cache also contains _mid layer 16. For 4B the
        # paper protocol needs layer 8, so rebuild that mid cache explicitly.
        for size, task in missing_main:
            main = main_cache(size, task)
            if SIZES[size]["mid_layer"] != 16:
                mid = mid_cache(size, task)
                cmd = [
                    python, "fast_softprompt/precompute_mid_from_cache.py",
                    "--cache", str(main), "--model", SIZES[size]["model"],
                    "--mid_layer", str(SIZES[size]["mid_layer"]),
                    "--output", str(mid),
                    "--batch_size", "8" if task == "verih" else "16",
                ]
                print("[caches] ", " ".join(cmd))
                subprocess.run(cmd, cwd=ROOT, check=True)
        # Build mid caches that are missing while the main cache already exists.
        for size, task in missing_mid:
            if (size, task) in missing_main:
                continue  # already built above
            main = main_cache(size, task)
            if not main.exists():
                continue
            mid = mid_cache(size, task)
            cmd = [
                python, "fast_softprompt/precompute_mid_from_cache.py",
                "--cache", str(main), "--model", SIZES[size]["model"],
                "--mid_layer", str(SIZES[size]["mid_layer"]),
                "--output", str(mid),
                "--batch_size", "8" if task == "verih" else "16",
            ]
            print("[caches] ", " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
        # Re-check after building.
        errors = [e for e in errors if not (
            e.startswith("missing main cache") or e.startswith("missing mid cache"))]
    return errors


def build_jobs(out: Path, sizes, tasks, methods, python: str) -> list[Job]:
    jobs = []
    for method in METHODS:
        if method not in methods:
            continue
        for size in sizes:
            for task in tasks:
                target = TASKS[task]
                if method.startswith("sp"):
                    k = int(method[2:])
                    out_dir = out / f"{task}_{size}" / method
                    cmd = [
                        python, "fast_softprompt/train_softprompt.py",
                        "--cache", str(main_cache(size, task)), "--mmap_cache",
                        "--mtp_dir", SIZES[size]["mtp"],
                        "--output_dir", str(out_dir),
                        "--soft_prompt_length", str(k), "--recursive",
                        "--classifier_head", "linear",
                        "--lr", "3e-4", "--min_lr", "1e-4",
                        "--epochs", "10", "--batch_size", "1",
                        "--dropout", "0.2", "--target_eval", str(target),
                    ]
                elif method == "mlp":
                    out_dir = out / f"{task}_{size}" / "mlp"
                    cmd = [
                        python, "fast_softprompt/train_mlp.py",
                        "--cache", str(main_cache(size, task)), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--hidden_dim", "2048",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]
                else:  # multimax
                    out_dir = out / f"{task}_{size}" / "multimax"
                    cmd = [
                        python, "fast_softprompt/train_multimax.py",
                        "--cache", str(mid_cache(size, task)), "--mmap_cache",
                        "--output_dir", str(out_dir),
                        "--num_heads", "10", "--mlp_width", "100",
                        "--lr", "5e-4", "--epochs", "10", "--batch_size", "32",
                        "--target_eval", str(target),
                    ]
                label = f"{method}_{task}_{size}"
                jobs.append(Job(
                    label=label, size=size, task=task, method=method,
                    k=(int(method[2:]) if method.startswith("sp") else None),
                    cmd=cmd, out_dir=out_dir,
                    log_path=out / "logs" / f"{label}.log",
                ))
    return jobs


def run_job(job: Job, gpu: int) -> tuple[bool, float, str]:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(job.log_path, "w") as log:
        proc = subprocess.run(job.cmd, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    return proc.returncode == 0, elapsed, str(job.log_path)


def verify_one(job: Job, reference: Path, tolerance: float) -> dict:
    result = {
        "label": job.label, "size": job.size, "task": job.task,
        "method": job.method, "status": "FAIL", "reason": "",
        "acc": None, "f1": None, "ref_acc": None, "delta": None,
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
    acc = metrics.get("accuracy")
    f1 = metrics.get("f1")
    result["acc"], result["f1"] = acc, f1

    problems = []
    if not isinstance(acc, (int, float)) or not math.isfinite(acc) or not (0 <= acc <= 1):
        problems.append(f"bad accuracy={acc}")
    if not isinstance(f1, (int, float)) or not math.isfinite(f1) or not (0 <= f1 <= 1):
        problems.append(f"bad f1={f1}")
    if not history:
        problems.append("empty training_history")
    for h in history:
        for key in ("train_loss", "val_acc"):
            v = h.get(key)
            if isinstance(v, float) and not math.isfinite(v):
                problems.append(f"non-finite {key}")
                break
    if not isinstance(cfg.get("best_epoch"), int) or cfg.get("best_epoch", 0) <= 0:
        problems.append(f"bad best_epoch={cfg.get('best_epoch')}")
    if job.method.startswith("sp"):
        if cfg.get("soft_prompt_length") != job.k:
            problems.append(f"soft_prompt_length={cfg.get('soft_prompt_length')} != {job.k}")
        if cfg.get("recursive") is not True:
            problems.append("recursive is not True")
        if cfg.get("classifier_head") != "linear":
            problems.append(f"classifier_head={cfg.get('classifier_head')}")
    elif job.method == "mlp":
        if cfg.get("mlp_hidden") != 2048:
            problems.append(f"mlp_hidden={cfg.get('mlp_hidden')} != 2048")
    else:
        if cfg.get("num_heads") != 10 or cfg.get("mlp_width") != 100:
            problems.append(f"MultiMax config heads={cfg.get('num_heads')} width={cfg.get('mlp_width')}")

    ref_path = reference / f"{job.task}_{job.size}" / job.method / "final_results.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
        ref_acc = ref.get("metrics", {}).get("accuracy")
        result["ref_acc"] = ref_acc
        if isinstance(ref_acc, (int, float)) and isinstance(acc, (int, float)):
            delta = (acc - ref_acc) * 100.0
            result["delta"] = delta
            if abs(delta) > tolerance:
                problems.append(f"|Δ|={abs(delta):.2f} pp > tolerance {tolerance:.2f} pp")
    else:
        problems.append(f"reference not found: {ref_path}")

    if problems:
        result["reason"] = "; ".join(problems)
    else:
        result["status"] = "PASS"
    return result


def write_summary(out: Path, results: list[dict], reference: Path, tolerance: float):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    lines = [
        "# 4B/9B rerun verification",
        "",
        f"Reference: `{reference}`  ",
        f"Tolerance: {tolerance:.2f} pp  ",
        "",
        "| Size | Task | Method | Acc | F1 | Ref acc | Δ (pp) | Status | Notes |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in results:
        acc = f"{r['acc']*100:.2f}" if isinstance(r["acc"], (int, float)) else "—"
        f1 = f"{r['f1']*100:.2f}" if isinstance(r["f1"], (int, float)) else "—"
        ref = f"{r['ref_acc']*100:.2f}" if isinstance(r["ref_acc"], (int, float)) else "—"
        delta = f"{r['delta']:+.2f}" if isinstance(r["delta"], (int, float)) else "—"
        lines.append(
            f"| {r['size']} | {r['task']} | {r['method']} | {acc} | {f1} | {ref} | "
            f"{delta} | {r['status']} | {r['reason']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results_verified")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "4,5,6,7"),
                    help="comma-separated GPU ids")
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter for training subprocesses")
    ap.add_argument("--reference", default="results_strict",
                    help="reference result directory for comparison")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="allowed |accuracy delta| vs reference, in pp")
    ap.add_argument("--sizes", default="4b,9b")
    ap.add_argument("--tasks", default="verih,nemotron_safety")
    ap.add_argument("--methods", default="sp1,sp2,sp5,mlp,multimax")
    ap.add_argument("--build-caches", action="store_true",
                    help="build missing caches (can take hours; may use a different data snapshot)")
    ap.add_argument("--dry-run", action="store_true", help="print jobs and exit")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        sys.exit(f"ERROR: current Python cannot import torch ({e}).\n"
                 f"Run with the sp environment: /home/suzeqin/miniconda3/envs/sp/bin/python "
                 f"reproduce_4b_9b.py ...")

    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for s in sizes:
        if s not in SIZES:
            sys.exit(f"unknown size: {s}")
    for t in tasks:
        if t not in TASKS:
            sys.exit(f"unknown task: {t}")
    for m in methods:
        if m not in ("sp1", "sp2", "sp5", "mlp", "multimax"):
            sys.exit(f"unknown method: {m}")

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        sys.exit("no GPUs specified")
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    reference = (ROOT / args.reference).resolve() if not Path(args.reference).is_absolute() else Path(args.reference)

    print(f"Repository : {ROOT}")
    print(f"Python     : {args.python}")
    print(f"GPUs       : {gpus}")
    print(f"Output     : {out}")
    print(f"Reference  : {reference}")
    print(f"Tolerance  : {args.tolerance:.2f} pp")

    need_mid = "multimax" in methods
    errors = check_caches(sizes, tasks, need_mid, args.build_caches, args.python)
    if errors:
        for e in errors:
            print(f"[cache ERROR] {e}")
        print("\nRe-run with --build-caches to build missing caches, or create them manually:")
        print("  see RUN_COMMANDS.md §2.1/§2.3 and §3.1.")
        sys.exit(1)

    jobs = build_jobs(out, sizes, tasks, methods, args.python)
    if args.dry_run:
        print(f"\n{len(jobs)} jobs:")
        for job in jobs:
            print(f"  {job.label:24s} -> {job.out_dir}")
            print(f"    {' '.join(job.cmd)}")
        return

    print(f"\nRunning {len(jobs)} jobs with {len(gpus)} GPU(s)...")
    job_q = queue.Queue()
    for job in jobs:
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
                print(f"[{'done' if ok else 'FAIL'}] {job.label} ({elapsed/60:.1f} min) log={log}",
                      flush=True)
                with lock:
                    results.append({
                        "label": job.label, "ok": ok, "elapsed_min": elapsed,
                        "log": log, "job": job,
                    })
            finally:
                gpu_q.put(gpu)
                job_q.task_done()

    threads = [threading.Thread(target=worker) for _ in range(len(gpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify every requested job, whether or not the subprocess succeeded.
    jobs_by_label = {j.label: j for j in jobs}
    verified = []
    for job in jobs:
        verified.append(verify_one(job, reference, args.tolerance))
    verified.sort(key=lambda r: (r["size"], r["task"], r["method"]))

    write_summary(out, verified, reference, args.tolerance)
    failed = [r for r in verified if r["status"] != "PASS"]
    print(f"\n{len(verified) - len(failed)}/{len(verified)} passed.")
    if failed:
        print("Failures:")
        for r in failed:
            print(f"  {r['label']}: {r['reason']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
