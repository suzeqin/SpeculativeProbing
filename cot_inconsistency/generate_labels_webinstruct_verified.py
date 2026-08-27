#!/usr/bin/env python3
"""
Generate reasoning traces from TIGER-Lab/WebInstruct-verified using Bailian.

1. Download TIGER-Lab/WebInstruct-verified
2. Randomly sample questions from the requested split
3. Use a Bailian model to generate thinking + response

Writes progress to JSONL files for easy monitoring.
Data files go into data/ subdirectory, named by dataset split and model to avoid
overwrites.
"""

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading

from openai import OpenAI

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

NUM_SAMPLES = 100

DATASET_NAME = "TIGER-Lab/WebInstruct-verified"
DATASET_SPLIT = os.environ.get("WEBINSTRUCT_SPLIT", "train")

BAILIAN_MODEL = os.environ.get("BAILIAN_MODEL", "qwen3.5-plus")
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _slug(text):
    return text.replace("/", "_").replace(".", "_").replace("-", "_")


def _model_slug():
    return _slug(BAILIAN_MODEL)


def _split_slug(dataset_split):
    return _slug(dataset_split)


def gen_jsonl_path(dataset_split, suffix=""):
    base = f"generated_webinstruct_verified_{_split_slug(dataset_split)}_{_model_slug()}"
    if suffix:
        base = f"{base}_{suffix}"
    return os.path.join(DATA_DIR, f"{base}.jsonl")


def log(msg):
    print(msg, flush=True)


def _first_present(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def download_webinstruct_verified(dataset_split, num_samples):
    log(f"Loading {DATASET_NAME} ({dataset_split})...")
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=dataset_split)
    total_rows = len(ds)
    selected_count = min(num_samples, total_rows)

    if selected_count < num_samples:
        log(f"  Requested {num_samples} samples but split only has {total_rows}; using {selected_count}")

    ds = ds.shuffle(seed=RANDOM_SEED).select(range(selected_count))

    questions = []
    skipped = 0
    for row in ds:
        question = _first_present(row, "original_question", "question")
        answer = _first_present(row, "short_answer", "answer")
        if not question or answer is None:
            skipped += 1
            continue
        questions.append(
            {
                "task_id": row.get("task_id"),
                "subject": row.get("subject"),
                "difficulty": row.get("difficulty"),
                "answer_type": row.get("answer_type"),
                "question": question,
                "gold_answer": answer,
            }
        )

    log(f"  Loaded {len(questions)} usable questions (skipped {skipped})")
    return questions


def extract_cot_and_response(text):
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if not match:
        return None, None
    cot = match.group(1).strip()
    response = match.group(2).strip()
    return (cot, response) if cot and response else (None, None)


def generate_one(client, question):
    try:
        resp = client.chat.completions.create(
            model=BAILIAN_MODEL,
            messages=[{"role": "user", "content": question}],
            max_completion_tokens=4096,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            return reasoning, content
        cot, response = extract_cot_and_response(content)
        return cot, response
    except Exception as e:
        log(f"  Generation error: {e}")
        return None, None


def generate_samples(questions, dataset_split, workers, suffix=""):
    """Generate responses, appending each valid sample to JSONL.

    Resume-safe: if the output file already has records for some questions, skip them.
    """
    gen_path = gen_jsonl_path(dataset_split, suffix=suffix)
    log(
        f"\n=== Generating {len(questions)} responses via Bailian ({BAILIAN_MODEL}) "
        f"from {DATASET_NAME}:{dataset_split} ==="
    )
    log(f"  Output: {gen_path}")
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("BAILIAN_API_KEY")
    if not api_key:
        raise RuntimeError("Set DASHSCOPE_API_KEY or BAILIAN_API_KEY")
    client = OpenAI(api_key=api_key, base_url=BAILIAN_BASE_URL)

    # Resume: skip questions already in the output file.
    already_done = set()
    if os.path.exists(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    rec = json.loads(line)
                    already_done.add(rec["question"])
                except Exception:
                    continue
        if already_done:
            log(f"  Resume: {len(already_done)} questions already have outputs; they will be skipped.")

    questions = [q for q in questions if q["question"] not in already_done]
    log(f"  To generate: {len(questions)} new questions")
    if not questions:
        log("  Nothing to do.")
        return

    lock = threading.Lock()
    counters = {"done": 0, "valid": 0, "errors": 0}

    with open(gen_path, "a", encoding="utf-8") as f:
        def process(q):
            cot, response = generate_one(client, q["question"])
            with lock:
                counters["done"] += 1
                if cot and response:
                    counters["valid"] += 1
                    record = {
                        "task_id": q["task_id"],
                        "subject": q["subject"],
                        "difficulty": q["difficulty"],
                        "answer_type": q["answer_type"],
                        "question": q["question"],
                        "gold_answer": q["gold_answer"],
                        "cot": cot,
                        "response": response,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                else:
                    counters["errors"] += 1
                if counters["done"] % 50 == 0:
                    log(
                        f"  Generated {counters['done']}/{len(questions)} "
                        f"(valid: {counters['valid']}, errors: {counters['errors']})"
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(process, questions))

    log(f"  Done: {counters['valid']} valid, {counters['errors']} errors")
    log(f"  Saved to {gen_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--dataset-split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--suffix", type=str, default="",
                        help="Appended to the output filename (keeps new rollouts in a separate file)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    questions = download_webinstruct_verified(
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
    )
    generate_samples(
        questions=questions,
        dataset_split=args.dataset_split,
        workers=args.workers,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
