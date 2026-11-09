#!/usr/bin/env python3
"""One-click reproduction of the final table (Qwen 4B/9B + MiniCPM4.1-8B).

Writes all results to --out (default: results2/) and verifies them against the
reference results in --reference (default: results/).

Experiments (30 total):
  Qwen3.5-4B/9B x VerIH/nemotron_safety x {SP-1, SP-2, SP-5, MLP, MultiMax}
  MiniCPM4.1-8B x VerIH/nemotron_safety x {SP-1, SP-2, SP-5, MLP, MultiMax}

Protocol:
  * Qwen MultiMax: _mid cache, 4B layer 8 / 9B layer 16.
  * Qwen MLP: hidden_dim=2048.
  * MiniCPM SP: Eagle3 head; MLP hidden_dim=2048; MultiMax mid layer 16.
  * SP hyperparameters: lr 3e-4 -> 1e-4, 10 epochs, batch 1, dropout 0.2.

Usage:
  bash run_all_repro.sh
  bash run_all_repro.sh --gpus 0,5,6,3
  bash run_all_repro.sh --dry-run
  bash run_all_repro.sh --sizes 4b --tasks verih --methods mlp --out /tmp/smoke
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
QWEN = {"4b": {"name": "Qwen3.5-4B", "mid": 8}, "9b": {"name": "Qwen3.5-9B", "mid": 16}}
MC_BASE = "models/MiniCPM4.1-8B"
MC_E3 = "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16"
METHODS = ["sp5", "sp2", "sp1", "mlp", "multimax"]  # longest first
COMBOS = [
    ("qwen", "4b", "verih"),
    ("qwen", "4b", "nemotron_safety"),
    ("qwen", "9b", "verih"),
    ("qwen", "9b", "nemotron_safety"),
    ("minicpm", "mcpm", "verih"),
    ("minicpm", "mcpm", "nemotron_safety"),
]


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


def load_meta(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


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
            mid = qwen_mid(size, task)
            if mid.exists():
                meta = load_meta(mid)
                expected = QWEN[size]["mid"]
                if int(meta.get("layer", -999)) != expected:
                    errors.append(f"{mid}: layer={meta.get('layer')}, expected {expected}")
            for d in (f"models/{QWEN[size]['name']}", f"models/{QWEN[size]['name']}-mtp"):
                if not (ROOT / d).exists():
                    errors.append(f"missing model dir: {d}")
        else:
            for p in (mc_cache(task, ""), mc_cache(task, "_mid"), mc_cache(task, "_eagle3")):
                if not p.exists():
                    errors.append(f"missing cache: {p}")
                elif not Path(str(p) + ".hidden").exists():
                    errors.append(f"missing hidden sidecar: {p}.hidden")
            mid = mc_cache(task, "_mid")
            if mid.exists():
                meta = load_meta(mid)
                if int(meta.get("layer", -999)) != 16:
                    errors.append(f"{mid}: layer={meta.get('layer')}, expected 16")
            e3 = mc_cache(task, "_eagle3")
            if e3.exists():
                meta = load_meta(e3)
                if not meta.get("eagle3_layer_ids"):
                    errors.append(f"{e3}: missing eagle3_layer_ids")
            for d in (MC_BASE, MC_E3):
                if not (ROOT / d).exists():
                    errors.append(f"missing model dir: {d}")
    return errors


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


def verify(job: Job, reference: Path, tolerance: float) -> dict:
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
    acc, f1 = metrics.get("accuracy"), metrics.get("f1")
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
        if job.kind == "qwen":
            if cfg.get("arch") != "recursive_softprompt_cached":
                problems.append(f"arch={cfg.get('arch')}")
            if cfg.get("recursive") is not True:
                problems.append("recursive is not True")
            if cfg.get("classifier_head") != "linear":
                problems.append(f"classifier_head={cfg.get('classifier_head')}")
        else:
            if cfg.get("arch") != "eagle3_softprompt_minicpm":
                problems.append(f"arch={cfg.get('arch')}")
    elif job.method == "mlp":
        if cfg.get("mlp_hidden") != 2048:
            problems.append(f"mlp_hidden={cfg.get('mlp_hidden')} != 2048")
    else:
        if cfg.get("num_heads") != 10 or cfg.get("mlp_width") != 100:
            problems.append(
                f"MultiMax config heads={cfg.get('num_heads')} width={cfg.get('mlp_width')}")

    ref_path = reference / f"{job.task}_{job.size}" / job.method / "final_results.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
        ref_acc = ref.get("metrics", {}).get("accuracy")
        result["ref_acc"] = ref_acc
        if isinstance(ref_acc, (int, float)) and isinstance(acc, (int, float)):
            result["delta"] = (acc - ref_acc) * 100.0
    else:
        problems.append(f"reference not found: {ref_path}")

    if problems:
        result["status"] = "FAIL"
        result["reason"] = "; ".join(problems)
    elif result["delta"] is not None and abs(result["delta"]) > tolerance:
        result["status"] = "WARN"
        result["reason"] = f"|Δ|={abs(result['delta']):.2f} pp > tolerance {tolerance:.2f} pp"
    else:
        result["status"] = "PASS"
    return result


def write_summary(out: Path, results: list[dict], reference: Path, tolerance: float):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    lines = [
        "# One-click reproduction verification",
        "",
        f"Reference: `{reference}`  ",
        f"Tolerance: {tolerance:.2f} pp  ",
        "",
        "| Size | Task | Method | Acc | F1 | Ref acc | Δ (pp) | Status | Notes |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(results, key=lambda x: (x["size"], x["task"], x["method"])):
        acc = f"{r['acc']*100:.2f}" if isinstance(r["acc"], (int, float)) else "—"
        f1 = f"{r['f1']*100:.2f}" if isinstance(r["f1"], (int, float)) else "—"
        ref = f"{r['ref_acc']*100:.2f}" if isinstance(r["ref_acc"], (int, float)) else "—"
        delta = f"{r['delta']:+.2f}" if isinstance(r["delta"], (int, float)) else "—"
        lines.append(
            f"| {r['size']} | {r['task']} | {r['method']} | {acc} | {f1} | "
            f"{ref} | {delta} | {r['status']} | {r['reason']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results2")
    ap.add_argument("--gpus", default=os.environ.get("GPUS", "4,5,6,7"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--reference", default="results")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="warn if |accuracy delta| vs reference exceeds this many pp")
    ap.add_argument("--sizes", default="4b,9b,mcpm")
    ap.add_argument("--tasks", default="verih,nemotron_safety")
    ap.add_argument("--methods", default="sp1,sp2,sp5,mlp,multimax")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow writing into a non-empty --out")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        sys.exit(f"ERROR: current Python cannot import torch ({e}).\n"
                 f"Run with the sp environment: /home/suzeqin/miniconda3/envs/sp/bin/python "
                 f"reproduce_all.py ...")

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
    reference = (ROOT / args.reference).resolve() if not Path(args.reference).is_absolute() else Path(args.reference)

    if out.exists() and any(out.rglob("final_results.json")) and not args.overwrite:
        sys.exit(f"ERROR: {out} already contains results; pass --overwrite to replace them.")

    print(f"Repository : {ROOT}")
    print(f"Python     : {args.python}")
    print(f"GPUs       : {gpus}")
    print(f"Output     : {out}")
    print(f"Reference  : {reference}")
    print(f"Tolerance  : {args.tolerance:.2f} pp")

    errors = check_assets(COMBOS, sizes, tasks)
    if errors:
        for e in errors:
            print(f"[asset ERROR] {e}")
        print("\nFix missing caches/models first; see RUN_COMMANDS.md.")
        sys.exit(1)

    jobs = build_jobs(out, args.python, COMBOS, sizes, tasks, methods)
    if args.dry_run:
        print(f"\n{len(jobs)} jobs:")
        for job in jobs:
            print(f"  {job.label:26s} -> {job.out_dir}")
            print(f"    {' '.join(job.cmd)}")
        return

    out.mkdir(parents=True, exist_ok=True)
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

    verified = [verify(job, reference, args.tolerance) for job in jobs]
    write_summary(out, verified, reference, args.tolerance)
    failed = [r for r in verified if r["status"] == "FAIL"]
    warned = [r for r in verified if r["status"] == "WARN"]
    print(f"\n{len(verified) - len(failed) - len(warned)} PASS, "
          f"{len(warned)} WARN, {len(failed)} FAIL")
    if failed:
        print("FAIL details:")
        for r in failed:
            print(f"  {r['label']}: {r['reason']}")
        sys.exit(1)
    if warned:
        print("WARN details (accuracy differs from reference beyond tolerance):")
        for r in warned:
            print(f"  {r['label']}: {r['reason']}")


if __name__ == "__main__":
    main()
