#!/usr/bin/env python3
"""
Precompute base model hidden states for CoT repetitive thinking classification.

Usage:
  python fast_softprompt/precompute_hidden_states_cot.py \
    --model Qwen/Qwen3.5-9B \
    --data cot_inconsistency/data/generated_webinstruct_verified_train_qwen3_5_plus_labeled_repetitive_thinking.jsonl \
    --output fast_softprompt/cache_cot_repetitive_9b.pt \
    --num_train 1000 --num_test 200
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


def load_cot_data(data_path, tokenizer, num_train=1000, num_test=200, max_length=8192):
    """Load data, filter by token length first, then balanced-sample."""
    records = []
    with open(data_path) as f:
        for line in f:
            records.append(json.loads(line))

    # Filter by token length before sampling
    filtered = []
    skipped = 0
    for r in records:
        messages = [
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": "<think>\n" + r["cot"] + "\n</think>"},
        ]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True, return_dict=False
        )
        if len(ids) > max_length:
            skipped += 1
            continue
        r["_token_ids"] = ids
        filtered.append(r)

    print(f"Filtered: {skipped} skipped (>{max_length} tokens), {len(filtered)} kept")

    pos = [r for r in filtered if r["label"] == 1]
    neg = [r for r in filtered if r["label"] == 0]
    random.shuffle(pos)
    random.shuffle(neg)

    train_half = num_train // 2
    test_half = num_test // 2

    train = pos[:train_half] + neg[:train_half]
    remaining_pos = pos[train_half:]
    remaining_neg = neg[train_half:]
    test = remaining_pos[:test_half] + remaining_neg[:test_half]

    random.shuffle(train)
    random.shuffle(test)

    print(f"Data: {len(pos)} pos, {len(neg)} neg total (after length filter)")
    print(f"Train: {len(train)} ({train_half} pos + {train_half} neg)")
    print(f"Test: {len(test)} ({test_half} pos + {test_half} neg)")
    return train, test


def extract_tokenized(records):
    """Extract pre-tokenized ids and labels from records."""
    all_token_ids = [r["_token_ids"] for r in records]
    all_labels = [r["label"] for r in records]
    return all_token_ids, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which layer's hidden state to save (-1 = last). "
                             "Uses output_hidden_states; index 0 = embeddings, N = last block.")
    parser.add_argument("--mid_layer", type=int, default=None,
                        help="If set, also save this layer's hidden states to <output>_mid.pt.")
    parser.add_argument("--task", type=str, default="cot_repetitive",
                        help="Task name stored in the cache metadata.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    train_records, test_records = load_cot_data(
        args.data, tokenizer, args.num_train, args.num_test, args.max_length
    )

    train_ids, train_labels = extract_tokenized(train_records)
    print(f"Train: {len(train_ids)} samples")

    test_ids, test_labels = extract_tokenized(test_records)
    print(f"Test: {len(test_ids)} samples")

    all_ids = train_ids + test_ids
    all_labels = train_labels + test_labels

    print(f"\nLoading {args.model} with transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )
    model.eval()
    device = str(model.lm_head.weight.device)
    print(f"  Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    save_mid = args.mid_layer is not None
    num_train = len(train_ids)
    num_samples = len(all_ids)
    total_tokens = sum(len(ids) for ids in all_ids)
    # Probe hidden_dim by loading model config (cheap).
    hidden_dim = model.config.hidden_size

    print(f"Computing hidden states for {num_samples} samples "
          f"(total_tokens={total_tokens}, hidden_dim={hidden_dim}, "
          f"batch_size={args.batch_size}, mid_layer={args.mid_layer if save_mid else 'off'})...")

    base_meta = {
        "model": args.model,
        "task": args.task,
        "num_train": num_train,
        "num_test": len(test_ids),
        "token_ids": all_ids,
        "labels": all_labels,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    from cache_io import StreamingCacheWriter

    writer = StreamingCacheWriter(args.output, total_tokens=total_tokens,
                                  hidden_dim=hidden_dim, num_samples=num_samples,
                                  dtype=torch.bfloat16, backing="disk")
    writer.set_meta({**base_meta, "layer": args.layer})

    writer_mid = None
    if save_mid:
        mid_output = args.output.replace(".pt", "_mid.pt")
        writer_mid = StreamingCacheWriter(mid_output, total_tokens=total_tokens,
                                          hidden_dim=hidden_dim, num_samples=num_samples,
                                          dtype=torch.bfloat16, backing="disk")
        writer_mid.set_meta({**base_meta, "layer": args.mid_layer})

    need_hidden_list = save_mid or args.layer != -1
    for i in tqdm(range(0, num_samples, args.batch_size)):
        batch_ids = all_ids[i:i + args.batch_size]
        max_len = max(len(ids) for ids in batch_ids)
        pad_id = tokenizer.pad_token_id or 0
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)

        with torch.no_grad():
            if not need_hidden_list:
                out = model.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state
                hidden_mid = None
            else:
                out = model.model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
                hidden = out.hidden_states[args.layer] if args.layer != -1 else out.last_hidden_state
                hidden_mid = out.hidden_states[args.mid_layer] if save_mid else None

        for j, ids in enumerate(batch_ids):
            seq_len = len(ids)
            writer.append(hidden[j, :seq_len].detach().cpu())
            if save_mid:
                writer_mid.append(hidden_mid[j, :seq_len].detach().cpu())

    del model
    torch.cuda.empty_cache()

    writer.close()
    print(f"\nSaved cache to {args.output} (mmap_v1)")
    print(f"  Train: {num_train}, Test: {len(test_ids)}")
    if writer_mid is not None:
        writer_mid.close()
        print(f"Saved mid-layer cache to {writer_mid.output_path}")


if __name__ == "__main__":
    main()
