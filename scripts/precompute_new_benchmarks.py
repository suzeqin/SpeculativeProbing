#!/usr/bin/env python3
"""Precompute hidden-state caches for the three new benchmarks.

9 jobs = 3 tasks (agentlab / bfcl_v4 / livecodebench)
      x 3 models (Qwen3.5-4B, Qwen3.5-9B, MiniCPM4.1-8B).

Qwen jobs write the last-layer cache + the MultiMax mid-layer cache in one
pass; the MiniCPM job writes last / mid / eagle3 in one pass.

Usage: python scripts/precompute_new_benchmarks.py --gpus 4,5,6,7
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TASKS = ["agentlab", "bfcl_v4", "livecodebench"]

JOBS = []
for task in TASKS:
    JOBS.append({
        "label": f"qwen4b_{task}",
        "cmd": [PY, "fast_softprompt/precompute_hidden_states.py",
                "--model", "models/Qwen3.5-4B", "--task", task,
                "--output", f"cache_{task}_4b.pt",
                "--max_length", "2048", "--batch_size", "8", "--mid_layer", "8"],
    })
    JOBS.append({
        "label": f"qwen9b_{task}",
        "cmd": [PY, "fast_softprompt/precompute_hidden_states.py",
                "--model", "models/Qwen3.5-9B", "--task", task,
                "--output", f"cache_{task}_9b.pt",
                "--max_length", "2048", "--batch_size", "4", "--mid_layer", "16"],
    })
    JOBS.append({
        "label": f"mcpm_{task}",
        "cmd": [PY, "fast_softprompt/precompute_hidden_states_minicpm.py",
                "--task", task, "--output_prefix", f"cache_{task}_mcpm",
                "--base_model", "models/MiniCPM4.1-8B",
                "--max_length", "2048", "--batch_size", "2"],
    })


def run_job(job, gpu, log_dir):
    log_path = log_dir / f"{job['label']}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONUNBUFFERED"] = "1"
    if job["label"].startswith("mcpm"):
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(job["cmd"], cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    return proc.returncode == 0, time.time() - t0, str(log_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--kinds", default="qwen4b,qwen9b,mcpm")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    log_dir = ROOT / "logs_precompute_new"
    log_dir.mkdir(exist_ok=True)

    jobs = []
    for j in JOBS:
        kind = j["label"].split("_", 1)[0]
        task = j["label"][len(kind) + 1:]
        if kind in kinds and task in tasks:
            jobs.append(j)
    # long jobs first
    jobs.sort(key=lambda j: (0 if j["label"].startswith("mcpm")
                             else 1 if j["label"].startswith("qwen9b") else 2), reverse=True)
    print(f"{len(jobs)} precompute jobs on GPUs {gpus}")
    job_q = queue.Queue()
    for j in jobs:
        job_q.put(j)
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
                print(f"[start] {job['label']} on GPU {gpu}", flush=True)
                ok, elapsed, log = run_job(job, gpu, log_dir)
                print(f"[{'done' if ok else 'FAIL'}] {job['label']} "
                      f"({elapsed/60:.1f} min) log={log}", flush=True)
                with lock:
                    results.append({"label": job["label"], "ok": ok, "min": elapsed / 60})
            finally:
                gpu_q.put(gpu)
                job_q.task_done()

    threads = [threading.Thread(target=worker) for _ in range(len(gpus))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results)-len(failed)} done, {len(failed)} failed")
    if failed:
        for r in failed:
            print(f"  FAILED: {r['label']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
