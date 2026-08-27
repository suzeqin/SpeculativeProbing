#!/usr/bin/env python3
"""Zero-shot classification benchmark using the OpenAI API.

Reads test texts from one of our trained caches, builds a prompt per task
from `prompts/<task>.txt`, queries the OpenAI model, parses yes/no, and
writes metrics + predictions to disk.

API key is read from OPENAI_API_KEY env var. Do NOT hardcode it in source.

Usage:
  OPENAI_API_KEY=sk-... python benchmark_gpt_zeroshot.py \
      --task nemotron_safety --cache fast_softprompt/cache_nemotron_safety_9b.pt \
      --output fast_softprompt/results_nemotron_safety_gpt/gpt.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm


ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(ROOT, "prompts")

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"
MAX_CHARS = 12000  # truncate very long samples so a request fits in the context


def load_test_texts(cache_path):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    num_train = cache["num_train"]
    num_test = cache["num_test"]
    all_ids = cache["token_ids"]
    all_labels = cache["labels"]
    model_name = cache["model"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    test_ids = all_ids[num_train:]
    test_labels = all_labels[num_train:]
    texts = [tok.decode(ids, skip_special_tokens=True) for ids in test_ids]
    return texts, test_labels


def load_prompt(task):
    p = os.path.join(PROMPTS_DIR, f"{task}.txt")
    with open(p) as f:
        return f.read()


def build_prompt(template, text):
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n[... truncated ...]"
    return template.replace("{{TEXT}}", text)


def parse_prediction(completion_text):
    """Return 1 if yes, 0 if no, -1 if unparseable.

    Strategy: the prompt asks for a single-word answer but the model may emit
    reasoning before it. Look at the *last* yes/no word in the response.
    """
    if not completion_text:
        return -1
    s = completion_text.strip().lower()
    # Find all yes/no tokens (word-bounded)
    matches = re.findall(r"\b(yes|no)\b", s)
    if matches:
        final = matches[-1]
        return 1 if final == "yes" else 0
    # Fallback: plain first word
    first = re.match(r"[a-z]+", s)
    if first:
        w = first.group(0)
        if w.startswith("yes"):
            return 1
        if w.startswith("no"):
            return 0
    return -1


def call_openai(client, model, prompt, max_retries=3):
    """One-shot chat completion, retries on transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1024,
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            # Retry with backoff on rate limits / transient errors
            msg = str(e).lower()
            if "rate" in msg or "timeout" in msg or "connection" in msg or "server" in msg:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True,
                   choices=["adversarial_if", "wildchat", "verih",
                            "cot_repetitive", "branched_thinking", "nemotron_safety"])
    p.add_argument("--cache", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel API calls (stay polite: <= 16)")
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: set OPENAI_API_KEY env var", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    print(f"Loading test texts from {args.cache}...")
    texts, labels = load_test_texts(args.cache)
    print(f"  {len(texts)} samples, labels: {sum(labels)} pos / {len(labels) - sum(labels)} neg")

    template = load_prompt(args.task)
    prompts = [build_prompt(template, t) for t in texts]
    print(f"Loaded prompt template: prompts/{args.task}.txt ({len(template)} chars)")

    print(f"Querying {args.model} with {args.workers} parallel workers...")
    preds = [-1] * len(prompts)
    raw = [""] * len(prompts)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_openai, client, args.model, pr): i for i, pr in enumerate(prompts)}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            i = futures[fut]
            try:
                text_out = fut.result()
                raw[i] = text_out
                preds[i] = parse_prediction(text_out)
            except Exception as e:
                raw[i] = f"ERROR: {e}"
                preds[i] = -1
    dt = time.time() - t0

    # Handle unparseable: count separately and use 0 (safer default) for metrics
    unparseable = sum(1 for p in preds if p == -1)
    preds_for_metrics = [0 if p == -1 else p for p in preds]
    acc = accuracy_score(labels, preds_for_metrics)
    pr, rc, f1, _ = precision_recall_fscore_support(
        labels, preds_for_metrics, average="binary", zero_division=0)
    print(f"Done in {dt:.1f}s ({dt/len(texts):.2f}s/sample, {unparseable} unparseable)")
    print(f"Accuracy: {acc:.4f} | Precision: {pr:.4f} | Recall: {rc:.4f} | F1: {f1:.4f}")
    print(classification_report(labels, preds_for_metrics, zero_division=0))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    json.dump({
        "model": args.model,
        "task": args.task,
        "cache": args.cache,
        "prompt_file": f"prompts/{args.task}.txt",
        "num_test": len(labels),
        "unparseable": unparseable,
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "predictions": preds,
        "labels": labels,
        "raw_responses": raw,
        "runtime_sec": dt,
    }, open(args.output, "w"), indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
