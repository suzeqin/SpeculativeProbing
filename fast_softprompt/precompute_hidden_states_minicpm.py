#!/usr/bin/env python3
"""Precompute MiniCPM4.1-8B hidden states: last layer + mid (L=16) + Eagle3 triple [2, 18, 31].

Writes 3 caches per task (last / mid / eagle3).

Usage:
  python fast_softprompt/precompute_hidden_states_minicpm.py \
    --task verih --output_prefix fast_softprompt/cache_verih_mcpm
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fast_softprompt"))

import minicpm_compat  # noqa: E402  (transformers 5.x remote-code shim)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MODEL_NAME = "openbmb/MiniCPM4.1-8B"
MID_LAYER = 16
# Layer indices for the Eagle3 3-layer fusion, matching the official cpm.cu
# implementation (minicpm4_eagle.cuh): i=2, num_layers/2, num_layers-3.
# For MiniCPM4.1-8B with L=32: [2, 16, 29]. Transformers' output_hidden_states[k]
# corresponds to the output of layer k-1, matching cpm.cu's `layer_output` at
# iteration i == k.
EAGLE3_LAYER_IDS = [2, 16, 29]

# Reuse loaders from the main precompute script.
from precompute_hidden_states import (
    load_wildchat_data, load_verih_data, load_adversarial_if_data,
    load_nemotron_safety_data, tokenize_records,
)
from precompute_hidden_states_cot import load_cot_data
from cache_io import save_cache_mmap, StreamingCacheWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wildchat", "verih", "adversarial_if",
                                            "nemotron_safety",
                                            "cot_repetitive", "cot_branched"], required=True)
    parser.add_argument("--base_model", default=MODEL_NAME,
                        help="HF id or local path to MiniCPM4.1-8B")
    parser.add_argument("--output_prefix", required=True,
                        help="Will write <prefix>.pt, <prefix>_mid.pt, <prefix>_eagle3.pt")
    parser.add_argument("--data", default=None,
                        help="For cot_* tasks, path to the labeled JSONL rollout file.")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_train", type=int, default=None)
    parser.add_argument("--num_test", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Load data
    if args.task == "wildchat":
        train_records, test_records = load_wildchat_data()
    elif args.task == "verih":
        train_records, test_records = load_verih_data()
    elif args.task == "adversarial_if":
        kw = {"max_length": args.max_length}
        if args.num_train is not None: kw["num_train"] = args.num_train
        if args.num_test is not None: kw["num_test"] = args.num_test
        train_records, test_records = load_adversarial_if_data(tokenizer, **kw)
    elif args.task == "nemotron_safety":
        kw = {"max_length": args.max_length}
        if args.num_train is not None: kw["num_train"] = args.num_train
        if args.num_test is not None: kw["num_test"] = args.num_test
        train_records, test_records = load_nemotron_safety_data(tokenizer, **kw)
    elif args.task in ("cot_repetitive", "cot_branched"):
        if args.data is None:
            raise ValueError("--data (labeled rollout jsonl) is required for cot_* tasks")
        kw = {"max_length": args.max_length}
        if args.num_train is not None: kw["num_train"] = args.num_train
        if args.num_test is not None: kw["num_test"] = args.num_test
        train_records, test_records = load_cot_data(args.data, tokenizer, **kw)

    # Tokenize
    if args.task in ("adversarial_if", "nemotron_safety", "cot_repetitive", "cot_branched"):
        train_ids = [r["_token_ids"] for r in train_records]
        train_labels = [r["label"] for r in train_records]
        test_ids = [r["_token_ids"] for r in test_records]
        test_labels = [r["label"] for r in test_records]
    else:
        print("Tokenizing train...")
        train_ids, train_labels = tokenize_records(train_records, tokenizer, args.task, args.max_length)
        print("Tokenizing test...")
        test_ids, test_labels = tokenize_records(test_records, tokenizer, args.task, args.max_length)
    print(f"  Train: {len(train_ids)}, Test: {len(test_ids)}")

    all_ids = train_ids + test_ids
    all_labels = train_labels + test_labels

    print(f"\nLoading {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()
    device = str(model.lm_head.weight.device)
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    num_train = len(train_ids)
    num_samples = len(all_ids)
    total_tokens = sum(len(ids) for ids in all_ids)
    hidden_dim = model.config.hidden_size
    n_e3 = len(EAGLE3_LAYER_IDS)

    print(f"hidden={hidden_dim}, num_samples={num_samples}, total_tokens={total_tokens}")

    base_meta = {
        "model": args.base_model,
        "task": args.task,
        "num_train": num_train,
        "num_test": len(test_ids),
        "token_ids": all_ids,
        "labels": all_labels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)

    # Streaming writers to disk. The raw file-backed storages avoid building
    # huge Python lists in RAM, which OOMs the system for cot/branched caches
    # (5k samples × ~3800 tokens × 3 layers ~ 600 GB if list-accumulated).
    last_path = args.output_prefix + ".pt"
    mid_path = args.output_prefix + "_mid.pt"
    eagle3_path = args.output_prefix + "_eagle3.pt"

    w_last = StreamingCacheWriter(last_path, total_tokens=total_tokens,
                                  hidden_dim=hidden_dim, num_samples=num_samples,
                                  dtype=torch.bfloat16, backing="disk")
    w_last.set_meta({**base_meta, "layer": -1})

    w_mid = StreamingCacheWriter(mid_path, total_tokens=total_tokens,
                                 hidden_dim=hidden_dim, num_samples=num_samples,
                                 dtype=torch.bfloat16, backing="disk")
    w_mid.set_meta({**base_meta, "layer": MID_LAYER})

    w_e3 = StreamingCacheWriter(eagle3_path, total_tokens=total_tokens,
                                hidden_dim=hidden_dim * n_e3, num_samples=num_samples,
                                dtype=torch.bfloat16, backing="disk")
    w_e3.set_meta({
        **base_meta,
        "eagle3_layer_ids": EAGLE3_LAYER_IDS,
        "eagle3_num_layers": n_e3,
        "eagle3_per_layer_dim": hidden_dim,
    })

    pad_id = tokenizer.pad_token_id or 0
    print(f"Precomputing ({num_samples} samples, bs={args.batch_size})...")
    for i in tqdm(range(0, num_samples, args.batch_size)):
        batch_ids = all_ids[i:i + args.batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True, use_cache=False)
            last = out.hidden_states[-1]
            mid = out.hidden_states[MID_LAYER]
            e3 = [out.hidden_states[lid] for lid in EAGLE3_LAYER_IDS]

        for j, ids in enumerate(batch_ids):
            seq_len = len(ids)
            last_j = last[j, :seq_len].detach().cpu().to(torch.bfloat16)
            mid_j = mid[j, :seq_len].detach().cpu().to(torch.bfloat16)
            # Concat 3 eagle3 layers along feature dim -> [seq_len, 3*hidden_dim].
            e3_j = torch.cat([layer[j, :seq_len].detach().cpu().to(torch.bfloat16)
                              for layer in e3], dim=-1)
            w_last.append(last_j)
            w_mid.append(mid_j)
            w_e3.append(e3_j)

    del model
    torch.cuda.empty_cache()

    w_last.close()
    print(f"Saved {last_path} (mmap_v1)")
    w_mid.close()
    print(f"Saved {mid_path} (mmap_v1)")
    w_e3.close()
    print(f"Saved {eagle3_path} (mmap_v1, concatenated {n_e3} layers along feature dim)")


if __name__ == "__main__":
    main()
