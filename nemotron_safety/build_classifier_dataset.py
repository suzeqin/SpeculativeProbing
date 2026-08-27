#!/usr/bin/env python3
"""Build classifier_dataset.jsonl for Nemotron-Safety-Guard-Dataset-v3.

Binary task: prompt_label (safe -> 0, unsafe -> 1). Samples N balanced prompts
from the full multilingual dataset via reservoir sampling so we don't need to
load the entire dataset into memory.
"""
import json
import os
import random

from datasets import load_dataset

random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "classifier_dataset.jsonl")
NUM_PER_CLASS = 5500   # 5000 train + 500 val + 500 test per class -> 10000 train + 1000 val + 1000 test

def main():
    ds = load_dataset("nvidia/Nemotron-Safety-Guard-Dataset-v3", split="train", streaming=True)
    pool = {"safe": [], "unsafe": []}
    seen = {"safe": 0, "unsafe": 0}
    # Reservoir sampling per class
    for ex in ds:
        label = ex.get("prompt_label")
        prompt = ex.get("prompt")
        if label not in ("safe", "unsafe") or not prompt:
            continue
        seen[label] += 1
        rec = {"label": label, "text": prompt}
        if len(pool[label]) < NUM_PER_CLASS:
            pool[label].append(rec)
        else:
            j = random.randint(0, seen[label] - 1)
            if j < NUM_PER_CLASS:
                pool[label][j] = rec
        if len(pool["safe"]) >= NUM_PER_CLASS and len(pool["unsafe"]) >= NUM_PER_CLASS:
            # Stop once both buckets are full. With reservoir sampling we'd keep
            # updating forever; NUM_PER_CLASS samples from the start of the stream
            # is sufficient (dataset is already shuffled by HF).
            if seen["safe"] >= 5 * NUM_PER_CLASS and seen["unsafe"] >= 5 * NUM_PER_CLASS:
                break

    print(f"seen: safe={seen['safe']}, unsafe={seen['unsafe']}")
    print(f"pool: safe={len(pool['safe'])}, unsafe={len(pool['unsafe'])}")
    records = pool["safe"] + pool["unsafe"]
    random.shuffle(records)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} records to {OUT}")

if __name__ == "__main__":
    main()
