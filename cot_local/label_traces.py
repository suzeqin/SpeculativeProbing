#!/usr/bin/env python3
"""Label generated CoT traces with a locally served judge model (chat.completions).

Adapted from cot_inconsistency/label_generated_simpleqa_gpt_repetitive_thinking.py:
the original calls the OpenAI Responses API (gpt-5.4); we keep the same judge
prompts, label schemes, parsing and resume-safe JSONL writing, but call the
OpenAI-compatible chat API of the local vLLM server instead. Thinking is turned
off via the Qwen chat template so the judge answers with the bare label.
"""
import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cot_inconsistency"))

from openai import OpenAI  # noqa: E402
from label_generated_simpleqa_gpt_repetitive_thinking import (  # noqa: E402
    JUDGE_PROMPTS, LABEL_SCHEMES, DEFAULT_LABEL_SCHEME, load_samples,
)


def log(msg):
    print(msg, flush=True)


def label_one(client, sample, model_name, judge_prompt, task, tokenizer=None, context=9216):
    prompt = (
        f"{judge_prompt}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Chain-of-thought reasoning:\n{sample['cot']}\n\n"
        f"Final response:\n{sample['response']}"
    )
    scheme = LABEL_SCHEMES.get(task, DEFAULT_LABEL_SCHEME)
    # Leave room for thinking + verdict inside the server's context window.
    if tokenizer is not None:
        n_in = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_in + 1024 > context:
            # The <think> section alone exceeds the precompute --max_length 8192
            # filter anyway; skip without wasting a server round trip.
            return -1, {"answer": "skipped_too_long", "raw": ""}
        max_tokens = max(512, context - n_in - 64)
    else:
        max_tokens = 4096
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        # With thinking enabled the chat template inserts the <think> opener in
        # the prompt; only the text after </think> is the actual verdict.
        if "</think>" in raw:
            raw_answer = raw.split("</think>", 1)[1].strip()
        else:
            raw_answer = raw
        answer = raw_answer.lower()
        if "```" in answer:
            match = re.search(r"```(?:text)?\s*(.*?)```", answer, re.DOTALL)
            answer = match.group(1).strip().lower() if match else answer
        # Strict verdict parsing. Substring matching (the previous behaviour)
        # silently turned truncated "Thinking Process:" essays into positives:
        # "yes"/"branched" appears somewhere in almost any long reasoning text.
        # Require a short answer beginning with an exact allowed label word.
        low = answer.strip(" \t\n\r\"'`*.,:;!?()[]{}")
        m = re.match(r"^(yes|no|linear|branched)\b", low) if len(low) <= 64 else None
        if m is not None and m.group(1) in (scheme["positive"], scheme["negative"]):
            word = m.group(1)
            label = 1 if word == scheme["positive"] else 0
            parsed = {"answer": word, "raw": raw_answer}
        else:
            label = -1
            log(f"  Unparseable judge output ({len(raw_answer)} chars): {raw_answer[:120]!r}")
            parsed = {"answer": low[:120], "raw": raw_answer}
        return label, parsed
    except Exception as exc:
        log(f"  Label error: {exc}")
        return -1, {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task", required=True, choices=list(JUDGE_PROMPTS.keys()))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tokenizer", default="/data1/suzeqin/SpeculativeProbing/models/Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--context", type=int, default=9216)
    args = parser.parse_args()

    judge_prompt = JUDGE_PROMPTS[args.task]
    samples = load_samples(args.input, args.limit)
    log(f"=== Labeling {len(samples)} samples with {args.model} for {args.task} ===")
    log(f"  Input: {args.input}")
    output_path = args.input.replace(".jsonl", f"_labeled_{args.task}.jsonl")
    log(f"  Output: {output_path}")

    already_done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    rec = json.loads(line)
                    already_done.add(rec["question"])
                except Exception:
                    continue
        if already_done:
            log(f"  Resume: {len(already_done)} samples already labeled; skipping them.")

    samples = [s for s in samples if s["question"] not in already_done]
    log(f"  To label: {len(samples)} new samples")
    if not samples:
        log("  Nothing to do.")
        return

    client = OpenAI(api_key="local-vllm", base_url=args.base_url)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    lock = threading.Lock()
    counters = {"done": 0, "errors": 0, "positive": 0, "negative": 0}

    with open(output_path, "a", encoding="utf-8") as f:
        def process(sample):
            label, judge_out = label_one(client, sample, args.model, judge_prompt, args.task,
                                         tokenizer=tokenizer, context=args.context)
            with lock:
                counters["done"] += 1
                if label == -1:
                    counters["errors"] += 1
                elif label == 1:
                    counters["positive"] += 1
                else:
                    counters["negative"] += 1

                if label != -1:
                    record = {
                        "text": f"<think>{sample['cot']}</think>\n{sample['response']}",
                        "question": sample["question"],
                        "gold_answer": sample.get("gold_answer", ""),
                        "cot": sample["cot"],
                        "response": sample["response"],
                        "label": label,
                        "judge_output": judge_out,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()

                if counters["done"] % 200 == 0 or counters["done"] == len(samples):
                    log(
                        f"  Labeled {counters['done']}/{len(samples)} "
                        f"(positive: {counters['positive']}, "
                        f"negative: {counters['negative']}, errors: {counters['errors']})"
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(process, samples))

    log(
        f"Done: positive={counters['positive']}, "
        f"negative={counters['negative']}, errors={counters['errors']}"
    )


if __name__ == "__main__":
    main()
