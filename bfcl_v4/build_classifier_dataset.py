#!/usr/bin/env python3
"""Build the BFCL-v4 false-positive-rate benchmark dataset.

BFCL-v4 (Berkeley Function-Calling Leaderboard v4) supplies *clean, normal*
agent tool-calling traffic — exactly the business traffic a deployed safety
probe must NOT flag. The binary task for training is therefore:

  label 0 = benign BFCL tool-calling request (normal business traffic)
  label 1 = attack text (agent attacks / injections, from the AgentLAB
            attack corpus, in either direct user-request form or embedded in
            a tool result — the form attacks take in agentic pipelines)

The held-out evaluation pool deliberately contains BOTH classes (benign BFCL
test traffic + held-out attacks), so model selection by val accuracy stays
balanced; the *headline* metric for this benchmark is the false-positive rate
measured on the BFCL-benign test subset (an attack-free probe-impact test).

Output: bfcl_v4/classifier_dataset.jsonl  {"label", "text", "source"}
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BFCL_DATA = os.path.join(ROOT, "bfcl", "gorilla_repo", "berkeley-function-call-leaderboard",
                         "bfcl_eval", "data")
sys.path.insert(0, os.path.join(ROOT, "agentlab"))
from attack_corpus import load_all_attacks  # noqa: E402

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "classifier_dataset.jsonl")

RECORD_FILES = [
    "BFCL_v4_live_simple.json", "BFCL_v4_live_multiple.json",
    "BFCL_v4_live_parallel.json", "BFCL_v4_live_parallel_multiple.json",
    "BFCL_v4_simple_python.json", "BFCL_v4_simple_java.json",
    "BFCL_v4_simple_javascript.json", "BFCL_v4_multiple.json",
    "BFCL_v4_parallel.json", "BFCL_v4_parallel_multiple.json",
    "BFCL_v4_multi_turn_base.json", "BFCL_v4_multi_turn_miss_func.json",
    "BFCL_v4_multi_turn_miss_param.json", "BFCL_v4_multi_turn_long_context.json",
    "BFCL_v4_web_search.json", "BFCL_v4_memory.json",
]

WRAPS = [
    "--- IMPORTANT NOTICE ---\n{attack}",
    "[TOOL OUTPUT ANNEX]\n{attack}",
    "<!-- {attack} -->",
    "System-generated appendix:\n{attack}",
    "P.S. {attack}",
]


def load_bfcl_samples() -> list[dict]:
    """Expand BFCL records into per-request benign texts.

    Each record yields up to two negatives: the bare user request, and the
    request together with the tool schemas it may call (the monitor in an
    agent pipeline observes both). The schema variant doubles the unique
    benign pool so train/test never share identical texts.
    """
    out: list[dict] = []
    for fname in RECORD_FILES:
        path = os.path.join(BFCL_DATA, fname)
        if not os.path.exists(path):
            continue
        for li, line in enumerate(open(path)):
            r = json.loads(line)
            turns = r.get("question") or []
            if not turns:
                continue
            funcs = r.get("function") or []
            schema = ""
            if funcs:
                lines = []
                for fdoc in funcs[:3]:
                    if isinstance(fdoc, dict):
                        lines.append(f"[Tool schema] {fdoc.get('name','')}: "
                                     f"{str(fdoc.get('description',''))[:200]}")
                schema = "\n".join(lines)
            if len(turns) == 1:
                msgs = turns[0]
                text = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
                if text.strip():
                    out.append({"text": text.strip(), "source": fname, "id": r.get("id")})
                    if schema:
                        out.append({"text": f"{text.strip()}\n\n{schema}",
                                    "source": fname + "+schema", "id": r.get("id")})
            else:
                for ti, msgs in enumerate(turns):
                    text = "\n".join(m.get("content", "") for m in msgs if isinstance(m, dict))
                    if text.strip():
                        out.append({"text": text.strip(), "source": fname,
                                    "id": f"{r.get('id')}#t{ti}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_neg_train", type=int, default=5000)
    ap.add_argument("--num_neg_test", type=int, default=400)
    ap.add_argument("--num_pos_train", type=int, default=5500)
    ap.add_argument("--num_pos_test", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = random.Random(42)

    benign = [s for s in load_bfcl_samples()
              if len(s["text"]) >= 25]
    # drop trivial numeric-only / degenerate texts
    benign = [s for s in benign if len(s["text"].split()) >= 4]
    print(f"BFCL benign samples: {len(benign)}")
    rng.shuffle(benign)

    needed = args.num_neg_train + args.num_neg_test
    assert len(benign) >= needed, f"need {needed} benign samples, got {len(benign)}"

    corpus = load_all_attacks()
    vectors = [t for texts in corpus["injection_vectors"].values() for t in texts]
    goals = corpus["injection_goals"]
    horizon = corpus["horizon_payloads"]
    safety = [a["text"] for a in corpus["agent_attacks"]]
    hijack = [a["text"] for a in corpus["hijack_descriptions"]]
    attack_pool = vectors * 5 + goals * 3 + horizon * 6 + hijack * 10 + safety
    rng.shuffle(attack_pool)
    # hold out the tail of the pool for the eval split
    train_pool = attack_pool[: int(len(attack_pool) * 0.9)]
    test_pool = attack_pool[int(len(attack_pool) * 0.9):]
    print(f"attack pool: {len(attack_pool)} (train {len(train_pool)} / test {len(test_pool)})")

    # benign question templates to dress tool-result attacks as agent traffic
    benign_qs = [s["text"] for s in benign[:3000]]

    records: list[dict] = []
    seen: set = set()

    def push(label, text, source, split):
        key = (text[:200], text[-100:])
        if key in seen:
            return False
        seen.add(key)
        records.append({"label": label, "text": text, "source": source, "split": split})
        return True

    def make_pos(attack, rng, mode_hint=None):
        mode = mode_hint or ("direct" if rng.random() < 0.45 else "tool_result")
        if mode == "direct":
            return attack, "attack_direct"
        q = rng.choice(benign_qs)
        q = (q[:400]) if len(q) > 400 else q
        wrap = rng.choice(WRAPS).format(attack=attack)
        return f"User request: {q}\n\n[Tool result]\n{wrap}", "attack_tool_result"

    # train positives
    n = 0
    i = 0
    while n < args.num_pos_train:
        text, src = make_pos(train_pool[i % len(train_pool)], rng)
        i += 1
        n += int(push(1, text, src, "train"))

    # train negatives (BFCL benign)
    n = 0
    idx = 0
    max_iter = 2 * len(benign)
    it = 0
    while n < args.num_neg_train and it < max_iter:
        it += 1
        s = benign[idx % len(benign)]
        idx += 1
        n += int(push(0, s["text"], f"bfcl:{s['source']}", "train"))

    # held-out: BFCL benign test traffic + held-out attacks (split="test")
    n = 0
    it = 0
    while n < args.num_neg_test and it < max_iter:
        it += 1
        s = benign[idx % len(benign)]
        idx += 1
        n += int(push(0, s["text"], f"bfcl_test:{s['source']}", "test"))
    n = 0
    i = 0
    while n < args.num_pos_test:
        text, src = make_pos(test_pool[i % len(test_pool)], rng)
        i += 1
        n += int(push(1, text, src, "test"))

    # train/test pools are each balanced by construction; shuffle within split
    train_recs = [r for r in records if r["split"] == "train"]
    test_recs = [r for r in records if r["split"] == "test"]
    rng.shuffle(train_recs)
    rng.shuffle(test_recs)
    out = train_recs + test_recs
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("train", "test"):
        recs = [r for r in out if r["split"] == sp]
        pos = sum(r["label"] for r in recs)
        print(f"  {sp}: {len(recs)} records ({pos} pos / {len(recs)-pos} neg)")
    print(f"wrote {len(out)} records to {args.out}")


if __name__ == "__main__":
    main()
