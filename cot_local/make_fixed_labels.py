#!/usr/bin/env python3
"""Build clean *_labeled_<task>_fixed_all.jsonl files.

Keeps only judge outputs that are an exact single verdict word. The original
labelling run truncated ~20% of judge responses mid-reasoning (no </think>
marker); those long "Thinking Process:" essays were mis-parsed by substring
matching and carried spurious positive labels. Dropping them is what the
original label_traces.py intended to do for unparseable outputs (label == -1).

No re-judging: forcing enable_thinking=False biases this judge to "no"/"linear"
(measured 0/300 positives), and letting thinking run to completion is far too
slow (~400 tok/s decode). The short verdicts are genuine thinking-on outputs
correctly parsed from after </think>, so they are the trustworthy subset.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(
    ROOT, "cot_inconsistency/data",
    "generated_webinstruct_verified_train_Qwen3_5_122B_A10B_GPTQ_Int4",
)
TASKS = {
    "repetitive_thinking": {"yes": 1, "no": 0},
    "branched_thinking": {"branched": 1, "linear": 0},
}

for task, table in TASKS.items():
    src = f"{BASE}_labeled_{task}.jsonl"
    dst = f"{BASE}_labeled_{task}_fixed_all.jsonl"
    kept, dropped = [], 0
    with open(src, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            raw = r["judge_output"].get("raw", "").strip().lower()
            if raw in table and r["label"] == table[raw]:
                kept.append(r)
            else:
                dropped += 1
    with open(dst, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pos = sum(r["label"] == 1 for r in kept)
    print(f"{task}: kept={len(kept)} pos={pos} ({100*pos/max(1,len(kept)):.1f}%) "
          f"neg={len(kept)-pos} dropped={dropped} -> {dst}")
