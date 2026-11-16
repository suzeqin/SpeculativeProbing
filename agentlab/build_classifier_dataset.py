#!/usr/bin/env python3
"""Build the AgentLAB indirect-injection classifier dataset.

Binary task: ``label`` 0 = benign agent-environment content, 1 = indirect
prompt-injection / agent attack. Positives follow the real AgentLAB threat
model (RAG / tool-output token pollution): benign documents (email bodies,
cloud files, web pages, tool listings) with an injection payload embedded, plus
a minority of direct attack instructions phrased as agent requests.

Sources (all bundled in the AgentLAB / AgentDojo repos):
  benign   - agentdojo environment documents (emails, files, calendar, slack,
             web pages, reviews, transactions) + benign user-task prompts
  attacks  - agentdojo injection vectors & injection-task goals, AgentLAB
             long-horizon payloads, Agent-SafetyBench instructions, Intent
             Hijacking descriptions

Output: agentlab/classifier_dataset.jsonl with {"label", "text", "source"}
records, shuffled with seed 42. Scaling follows the nemotron_safety protocol:
--num_per_class 5500  -> 10000 train + ~500 val + ~500 test (target_eval 800).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack_corpus import load_all_attacks  # noqa: E402
from agentdojo_env import load_benign_documents  # noqa: E402

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "classifier_dataset.jsonl")

# Carriers: benign documents into which payloads get embedded. Multi-doc
# contexts keep positives/negatives length-matched so the probe must read
# content, not count lines.
CONTEXT_HEADERS = {
    "email": "----- Forwarded message -----",
    "file": "----- Document content -----",
    "web_page": "----- Retrieved web page -----",
    "slack_message": "----- Slack message -----",
    "review": "----- Listing with reviews -----",
    "calendar": "----- Calendar entry -----",
    "transaction": "----- Transaction record -----",
    "flight": "----- Flight listing -----",
}

# Embedding styles for the payload (mimicking real poisoning patterns).
WRAPS = [
    "--- IMPORTANT NOTICE ---\n{attack}",
    "<!-- {attack} -->",
    "[SYSTEM AUTO-GENERATED MESSAGE]\n{attack}",
    "P.S. {attack}",
    "Forwarded note from the maintainer:\n{attack}",
    "###\n{attack}\n###",
]


def _mk_context(rng, docs, n_docs=None):
    """Join 1-4 benign docs into one tool-output context."""
    k = n_docs or rng.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.2, 0.1])[0]
    chosen = rng.sample(range(len(docs)), min(k, len(docs)))
    parts = []
    for i in chosen:
        d = docs[i]
        header = CONTEXT_HEADERS.get(d["kind"], "--- Document ---")
        parts.append(f"{header}\n{d['text']}")
    return "\n\n".join(parts)


def _embed(ctx: str, attack: str, rng) -> str:
    wrap = rng.choice(WRAPS).format(attack=attack)
    style = rng.random()
    if style < 0.4:  # appended (poisoned email reply / file tail)
        return f"{ctx}\n\n{wrap}"
    if style < 0.7:  # prepended (poisoned RAG chunk header)
        return f"{wrap}\n\n{ctx}"
    # spliced in the middle (poisoned document body)
    lines = ctx.split("\n\n")
    if len(lines) > 1:
        i = rng.randrange(1, len(lines))
        return "\n\n".join(lines[:i] + [wrap] + lines[i:])
    return f"{ctx}\n{wrap}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_per_class", type=int, default=5500)
    ap.add_argument("--num_test_per_class", type=int, default=400,
                    help="held-out records per class (split='test')")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = random.Random(42)

    corpus = load_all_attacks()
    docs = load_benign_documents()
    prompts = [p for p in corpus["user_prompts"] if len(p) < 600]
    print(f"benign docs: {len(docs)}, benign prompts: {len(prompts)}")

    # --- attack pool -------------------------------------------------------
    vectors = [t for texts in corpus["injection_vectors"].values() for t in texts]
    goals = corpus["injection_goals"]
    horizon = corpus["horizon_payloads"]
    safety = [a["text"] for a in corpus["agent_attacks"]]
    hijack = [a["text"] for a in corpus["hijack_descriptions"]]
    print(f"attacks: vectors={len(vectors)} goals={len(goals)} horizon={len(horizon)} "
          f"safety={len(safety)} hijack={len(hijack)}")

    # weighting: Agent-SafetyBench + hijack descriptions give attack diversity;
    # vectors/goals/horizon are the canonical indirect-injection texts.
    attack_pool = (
        vectors * 6 + goals * 3 + horizon * 6 + hijack * 8 + safety
    )
    rng.shuffle(attack_pool)
    print(f"weighted attack pool: {len(attack_pool)}")

    n = args.num_per_class
    records: list[dict] = []

    # --- negatives ---------------------------------------------------------
    neg_units = ([d["text"] for d in docs] + prompts)
    seen: set[str] = set()

    def gen_negative():
        text = rng.choice(neg_units)  # single doc / prompt
        if rng.random() >= 0.25:
            text = _mk_context(rng, docs)  # multi-doc tool-output context
            if rng.random() < 0.35:
                text = f"{text}\n\n----- User task -----\n{rng.choice(prompts)}"
        return text

    for split, count in (("train", n), ("test", args.num_test_per_class)):
        got, tries = 0, 0
        while got < count and tries < 40000:
            tries += 1
            text = gen_negative()
            key = text[:300]
            if key in seen:
                continue
            tries = 0
            seen.add(key)
            records.append({"label": 0, "text": text, "source": "agentdojo_env",
                            "split": split})
            got += 1

    # --- positives ---------------------------------------------------------
    # 80% indirect (embedded into benign context), 20% direct instructions.
    # The attack pool is partitioned 90/10 into train/test payloads so the
    # held-out attacks are never seen in training.
    split_pool = attack_pool[: int(len(attack_pool) * 0.9)]
    test_pool = attack_pool[int(len(attack_pool) * 0.9):]

    def gen_positive(pool):
        attack = rng.choice(pool)
        if len(attack) < 30:
            return None
        if rng.random() < 0.8:
            ctx = _mk_context(rng, docs, n_docs=rng.choices([1, 2], weights=[0.5, 0.5])[0])
            text = _embed(ctx, attack, rng)
            mode = "indirect"
        else:
            text = attack
            mode = "direct"
        return text, mode

    for split, pool, count in (("train", split_pool, n),
                               ("test", test_pool, args.num_test_per_class)):
        got, tries = 0, 0
        while got < count and tries < 60000:
            tries += 1
            out = gen_positive(pool)
            if out is None:
                continue
            text, mode = out
            key = (text[:200], text[-100:])
            if key in seen:
                continue
            tries = 0
            seen.add(key)
            records.append({"label": 1, "text": text, "source": f"attack_{mode}",
                            "split": split})
            got += 1

    train_recs = [r for r in records if r["split"] == "train"]
    test_recs = [r for r in records if r["split"] == "test"]
    rng.shuffle(train_recs)
    rng.shuffle(test_recs)
    out_records = train_recs + test_recs
    with open(args.out, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for sp in ("train", "test"):
        recs = [r for r in out_records if r["split"] == sp]
        pos = sum(r["label"] for r in recs)
        print(f"  {sp}: {len(recs)} records ({pos} pos / {len(recs)-pos} neg)")
    print(f"wrote {len(out_records)} records to {args.out}")


if __name__ == "__main__":
    main()
