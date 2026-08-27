#!/usr/bin/env python3
"""
Precompute base model hidden states and input embeddings using vLLM encode.
Saves cached tensors to disk so training only needs the MTP layer.

Usage (vllm env):
  python fast_softprompt/precompute_hidden_states.py \
    --model Qwen/Qwen3.5-9B \
    --task wildchat \
    --output fast_softprompt/cache_wildchat_9b.pt
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

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def load_wildchat_data(num_train=3800, num_test=200, max_length=2048):
    with open(os.path.join(ROOT, "wildchat", "labeled_data_gpt54.json")) as f:
        data = json.load(f)
    all_samples = data["train"] + data["test"]
    pos = [r for r in all_samples if r["label"] == 0]
    neg = [r for r in all_samples if r["label"] == 1]
    random.shuffle(pos)
    random.shuffle(neg)
    half = min(len(pos), len(neg))
    balanced = pos[:half] + neg[:half]
    random.shuffle(balanced)
    train = balanced[:num_train]
    test = balanced[num_train:num_train + num_test]
    print(f"Balanced: {half*2} total, using {len(train)} train + {len(test)} test")
    return train, test


def load_verih_data():
    data_dir = os.path.join(ROOT, "verih", "data")
    with open(os.path.join(data_dir, "train.json")) as f:
        train_raw = json.load(f)
    with open(os.path.join(data_dir, "test.json")) as f:
        test_raw = json.load(f)

    def parse(records):
        parsed = []
        for r in records:
            t = r.get("type", "")
            if "conflict" in t:
                label = 1
            elif "aligned" in t:
                label = 0
            else:
                continue
            parsed.append({"sys_prompt": r.get("sys_prompt", ""), "user_prompt": r.get("user_prompt", ""), "label": label})
        return parsed

    train_parsed = parse(train_raw)
    test_parsed = parse(test_raw)
    train_pos = [r for r in train_parsed if r["label"] == 1]
    train_neg = [r for r in train_parsed if r["label"] == 0]
    random.shuffle(train_pos)
    random.shuffle(train_neg)
    half = min(len(train_pos), len(train_neg))
    train_sampled = train_pos[:half] + train_neg[:half]
    random.shuffle(train_sampled)

    test_pos = [r for r in test_parsed if r["label"] == 1]
    test_neg = [r for r in test_parsed if r["label"] == 0]
    random.shuffle(test_pos)
    random.shuffle(test_neg)
    test_half = min(len(test_pos), len(test_neg), 327)
    test_sampled = test_pos[:test_half] + test_neg[:test_half]
    random.shuffle(test_sampled)
    return train_sampled, test_sampled


def load_adversarial_if_data(tokenizer, num_train=1000, num_test=100, max_length=8192):
    data_path = os.path.join(ROOT, "adversarial_if", "classifier_dataset.jsonl")
    records = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            label = 1 if r["label"] == "incorrect" else 0
            ids = tokenizer.encode(r["text"], add_special_tokens=False)
            if len(ids) > max_length:
                skipped += 1
                continue
            records.append({"_token_ids": ids, "label": label})
    print(f"Loaded {len(records)} samples ({skipped} skipped > {max_length} tokens)")

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train_half = num_train // 2
    test_half = num_test // 2
    train = pos[:train_half] + neg[:train_half]
    test = pos[train_half:train_half + test_half] + neg[train_half:train_half + test_half]
    random.shuffle(train)
    random.shuffle(test)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    return train, test


def load_pii_data(tokenizer, num_train=1000, num_test=100, max_length=2048):
    data_path = os.path.join(ROOT, "pii", "classifier_dataset.jsonl")
    records = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            label = 1 if r["label"] == "pii" else 0
            ids = tokenizer.encode(r["text"], add_special_tokens=True)
            if len(ids) > max_length:
                skipped += 1
                continue
            records.append({"_token_ids": ids, "label": label})
    print(f"Loaded {len(records)} samples ({skipped} skipped > {max_length} tokens)")

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train_half = num_train // 2
    test_half = num_test // 2
    train = pos[:train_half] + neg[:train_half]
    test = pos[train_half:train_half + test_half] + neg[train_half:train_half + test_half]
    random.shuffle(train)
    random.shuffle(test)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    return train, test


def load_nemotron_safety_data(tokenizer, num_train=10000, num_test=1000, max_length=2048):
    data_path = os.path.join(ROOT, "nemotron_safety", "classifier_dataset.jsonl")
    records = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            label = 1 if r["label"] == "unsafe" else 0
            ids = tokenizer.encode(r["text"], add_special_tokens=True)
            if len(ids) > max_length:
                skipped += 1
                continue
            records.append({"_token_ids": ids, "label": label})
    print(f"Loaded {len(records)} samples ({skipped} skipped > {max_length} tokens)")

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train_half = num_train // 2
    test_half = num_test // 2
    train = pos[:train_half] + neg[:train_half]
    test = pos[train_half:train_half + test_half] + neg[train_half:train_half + test_half]
    random.shuffle(train)
    random.shuffle(test)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    return train, test


def load_longsafety_data(tokenizer, num_train=1000, num_test=200, max_length=8192):
    data_path = os.path.join(ROOT, "longsafety", "classifier_dataset.jsonl")
    records = []
    skipped = 0
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            label = 1 if r["label"] == "unsafe" else 0
            ids = tokenizer.encode(r["text"], add_special_tokens=True)
            if len(ids) > max_length:
                skipped += 1
                continue
            records.append({"_token_ids": ids, "label": label})
    print(f"Loaded {len(records)} samples ({skipped} skipped > {max_length} tokens)")

    pos = [r for r in records if r["label"] == 1]
    neg = [r for r in records if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train_half = num_train // 2
    test_half = num_test // 2
    train = pos[:train_half] + neg[:train_half]
    test = pos[train_half:train_half + test_half] + neg[train_half:train_half + test_half]
    random.shuffle(train)
    random.shuffle(test)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    return train, test


def tokenize_records(records, tokenizer, task, max_length=2048):
    """Tokenize records and return token id lists + labels, filtering out too-long samples."""
    all_token_ids = []
    all_labels = []
    skipped = 0
    for r in records:
        if task == "wildchat":
            ids = tokenizer.apply_chat_template(r["messages"], add_generation_prompt=False, tokenize=True, return_dict=False)
        elif task == "verih":
            msgs = [{"role": "system", "content": r["sys_prompt"]}, {"role": "user", "content": r["user_prompt"]}]
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=False, tokenize=True, return_dict=False)
        else:
            raise ValueError(f"Unknown task: {task}")

        if len(ids) > max_length:
            skipped += 1
            continue
        all_token_ids.append(ids)
        all_labels.append(r["label"])

    if skipped:
        print(f"  Skipped {skipped} samples over {max_length} tokens")
    return all_token_ids, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--task", choices=["wildchat", "verih", "longsafety", "pii", "adversarial_if", "nemotron_safety"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train", type=int, default=None,
                        help="Override loader default num_train (longsafety/pii/adversarial_if)")
    parser.add_argument("--num_test", type=int, default=None,
                        help="Override loader default num_test (longsafety/pii/adversarial_if)")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer's hidden state to save (-1 = last). "
                             "Uses output_hidden_states; index 0 = embeddings, N = last block.")
    parser.add_argument("--mid_layer", type=int, default=None,
                        help="If set, also save this layer's hidden states and write a second "
                             "cache to --output with '_mid' inserted before the .pt extension.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load data
    if args.task == "wildchat":
        train_records, test_records = load_wildchat_data()
    elif args.task == "verih":
        train_records, test_records = load_verih_data()
    elif args.task == "longsafety":
        kw = {"max_length": args.max_length}
        if args.num_train is not None: kw["num_train"] = args.num_train
        if args.num_test is not None: kw["num_test"] = args.num_test
        train_records, test_records = load_longsafety_data(tokenizer, **kw)
    elif args.task == "pii":
        kw = {"max_length": args.max_length}
        if args.num_train is not None: kw["num_train"] = args.num_train
        if args.num_test is not None: kw["num_test"] = args.num_test
        train_records, test_records = load_pii_data(tokenizer, **kw)
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

    # Tokenize
    if args.task in ("longsafety", "pii", "adversarial_if", "nemotron_safety"):
        # Already tokenized in load_*_data
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

    # Use transformers to get per-token hidden states
    print(f"\nLoading {args.model} with transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()
    device = str(model.lm_head.weight.device)
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    save_mid = args.mid_layer is not None
    need_all_layers = save_mid or args.layer != -1

    # Batch encode to get hidden states
    print(f"Computing hidden states for {len(all_ids)} samples (batch_size={args.batch_size})...")
    all_hidden_states = []
    all_hidden_states_mid = [] if save_mid else None
    for i in tqdm(range(0, len(all_ids), args.batch_size)):
        batch_ids = all_ids[i:i + args.batch_size]
        # Pad batch
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            if need_all_layers:
                out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
                hidden = out.hidden_states[args.layer]
                if save_mid:
                    hidden_mid = out.hidden_states[args.mid_layer]
            else:
                out = model.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state

        # Extract per-sample hidden states (only valid tokens)
        for j, ids in enumerate(batch_ids):
            seq_len = len(ids)
            all_hidden_states.append(hidden[j, :seq_len].cpu().to(torch.bfloat16))
            if save_mid:
                all_hidden_states_mid.append(hidden_mid[j, :seq_len].cpu().to(torch.bfloat16))

    del model
    torch.cuda.empty_cache()

    num_train = len(train_ids)
    base_cache = {
        "model": args.model,
        "task": args.task,
        "num_train": num_train,
        "num_test": len(test_ids),
        "token_ids": all_ids,
        "labels": all_labels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from cache_io import save_cache_mmap
    cache = {**base_cache, "layer": args.layer, "hidden_states": all_hidden_states}
    save_cache_mmap(args.output, cache)
    print(f"\nSaved cache to {args.output} (mmap_v1, + {args.output}.hidden)")
    print(f"  Train: {num_train}, Test: {len(test_ids)}")
    print(f"  Hidden state shape example: {all_hidden_states[0].shape}")

    if save_mid:
        mid_output = args.output.replace(".pt", "_mid.pt")
        cache_mid = {**base_cache, "layer": args.mid_layer, "hidden_states": all_hidden_states_mid}
        save_cache_mmap(mid_output, cache_mid)
        print(f"Saved mid-layer cache to {mid_output}")
        print(f"  Hidden state shape example: {all_hidden_states_mid[0].shape}")


if __name__ == "__main__":
    main()
