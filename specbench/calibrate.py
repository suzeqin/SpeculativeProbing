#!/usr/bin/env python3
"""Calibration: validate the speculative-draft loops.

Runs a short greedy generation on a fixed prompt and reports the acceptance
statistics. A correct draft-loop convention gives a clearly above-chance
acceptance rate (~0.6-0.9 on greedy decoding); a broken convention collapses
to near 0. Used once per model family before running the benchmark.

Usage:
  python specbench/calibrate.py --family qwen --gpu 0
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "specbench"))

from spec_decode import QwenSpecEngine, Eagle3SpecEngine, GreedyEngine  # noqa: E402


def summarize(rounds):
    drafted = sum(r["drafted"] for r in rounds)
    accepted = sum(r["accepted"] for r in rounds)
    gen = sum(r["generated"] for r in rounds)
    n = len(rounds)
    return {
        "rounds": n,
        "tokens": gen,
        "drafted": drafted,
        "accepted": accepted,
        "acceptance_rate": accepted / max(drafted, 1),
        "acceptance_length": gen / max(n, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen", "eagle3"], required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. "
                                        "Artificial intelligence has transformed the modern world.")
    ap.add_argument("--new-tokens", type=int, default=64)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ROOTD = ROOT
    if args.family == "qwen":
        eng = QwenSpecEngine(os.path.join(ROOTD, "models/Qwen3.5-4B"),
                             os.path.join(ROOTD, "models/Qwen3.5-4B-mtp"), draft_gamma=3)
    else:
        eng = Eagle3SpecEngine(os.path.join(ROOTD, "models/MiniCPM4.1-8B"),
                               os.path.join(ROOTD, "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16"),
                               draft_gamma=3)
    from transformers import AutoTokenizer
    tok_dir = os.path.join(ROOTD, "models/Qwen3.5-4B" if args.family == "qwen"
                           else "models/MiniCPM4.1-8B")
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    ids = tok.encode(args.prompt, add_special_tokens=args.family != "eagle3")
    tokens, rounds = eng.generate(ids, args.new_tokens)
    print("prompt tokens:", len(ids))
    print(summarize(rounds))
    print("generated:", tok.decode(tokens)[:400])
    # sanity: greedy reference must match spec output prefix
    geng = GreedyEngine(os.path.join(ROOTD, "models/Qwen3.5-4B" if args.family == "qwen"
                                     else "models/MiniCPM4.1-8B"))
    gtoks, _ = geng.generate(ids, min(args.new_tokens, 32))
    match = tokens[:len(gtoks)] == gtoks
    print("greedy match:", match)
    if not match:
        print("spec:", tok.decode(tokens[:len(gtoks)])[:200])
        print("greedy:", tok.decode(gtoks)[:200])


if __name__ == "__main__":
    main()
