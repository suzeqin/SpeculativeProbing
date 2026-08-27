#!/usr/bin/env python3
"""
Label generated SimpleQA reasoning traces for repetitive thinking with GPT.

Reads a generated JSONL file with records like:
    {"question": ..., "gold_answer": ..., "cot": ..., "response": ...}

Writes a labeled JSONL file with:
    {"text": ..., "question": ..., "gold_answer": ..., "cot": ..., "response": ...,
     "label": 0|1, "judge_output": {"answer": "yes"|"no", "raw": ...}}
"""

import concurrent.futures
import json
import os
import re
import threading

from openai import OpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
DEFAULT_INPUT_PATH = os.path.join(DATA_DIR, "generated_qwen3_5-plus_filtered.jsonl")

JUDGE_PROMPTS = {
    "repetitive_thinking": """You are evaluating a chain-of-thought (COT) for a specific failure mode: meaningless repetitive thinking.
This means the model reaches an answer, then keeps revisiting the same question and the same reasoning in loops without introducing any new information — repeatedly saying things like "wait, let me check again," "actually, is it X instead?" and arriving at the same conclusion it already had.
This is NOT about length or complexity. A long COT that genuinely works through hard reasoning is fine. You are only looking for obvious cases where the model is stuck in a loop, retreading the same ground over and over.
Given the COT inside <cot> tags, respond with only yes or no.

yes = obviously meaningless repetitive thinking
no = everything else (productive reasoning, brief COT, mild hesitation, etc.)""",

    "irrelevant_thinking": """You are evaluating a chain-of-thought (COT) for a specific failure mode: irrelevant content in the reasoning.
This means the model's thinking contains substantial content that is unrelated to the user's question — such as discussing a completely different topic, going on tangents about unrelated facts, inserting random or off-topic reasoning, or reasoning about something the user never asked about.
Brief mentions or minor asides are fine. You are only looking for cases where a significant portion of the thinking is clearly irrelevant to the question being asked.
Given the question and COT, respond with only yes or no.

yes = the thinking contains substantial irrelevant content unrelated to the user's question
no = the thinking stays on topic (even if it's long, repetitive, or wrong — as long as it's about the right question)""",

    "answer_indecision": """You are evaluating a chain-of-thought (COT) for a specific failure mode: excessive indecision between multiple candidate answers.
This means the model considers two or more possible answers and keeps going back and forth between them — repeatedly switching its preferred answer, second-guessing itself, arguing for one answer then the other, without converging on a clear conclusion through new evidence or reasoning.
This is NOT about briefly considering alternatives before settling on an answer. That is normal, productive reasoning. You are looking for cases where the model is genuinely stuck oscillating between candidates, e.g. "so the answer is A... but wait, maybe it's B... actually A makes more sense... hmm but B could also work..." going in circles.
Given the question and COT, respond with only yes or no.

yes = the model is excessively indecisive, oscillating between multiple candidate answers without resolving
no = everything else (considers alternatives briefly then decides, confident reasoning, single answer path, etc.)""",

    "branched_thinking": """You are classifying a chain-of-thought (COT) reasoning style as either linear or branched.

LINEAR thinking: The model follows a single logical thread from start to finish. It works through the problem step by step in one direction, building on each previous step. There may be minor corrections, but the overall structure is a straight path from problem to solution.

BRANCHED thinking: The model explicitly explores multiple approaches, considers alternative solution paths, backtracks to try different strategies, or systematically evaluates different cases/scenarios. The reasoning has a tree-like structure where the model goes down one path, then comes back to explore another.

This is about reasoning STRUCTURE, not quality. A linear solution can be correct or wrong; a branched solution can be correct or wrong. You are only classifying the shape of the reasoning.

Given the question and COT, respond with only "linear" or "branched".

linear = single-path, step-by-step reasoning
branched = multi-path, explores alternatives or backtracks to try different approaches""",
}


def log(msg):
    print(msg, flush=True)


LABEL_SCHEMES = {
    "branched_thinking": {"positive": "branched", "negative": "linear"},
}
DEFAULT_LABEL_SCHEME = {"positive": "yes", "negative": "no"}


def label_one(client, sample, model_name, judge_prompt, task=None):
    prompt = (
        f"{judge_prompt}\n\n"
        f"Question: {sample['question']}\n\n"
        f"Chain-of-thought reasoning:\n{sample['cot']}\n\n"
        f"Final response:\n{sample['response']}"
    )
    scheme = LABEL_SCHEMES.get(task, DEFAULT_LABEL_SCHEME)
    try:
        resp = client.responses.create(
            model=model_name,
            reasoning={"effort": "low"},
            input=[{"role": "user", "content": prompt}],
            # max_output_tokens=20,
        )
        # log(resp)
        raw = resp.output_text.strip()
        answer = raw.lower()
        if "```" in answer:
            match = re.search(r"```(?:text)?\s*(.*?)```", answer, re.DOTALL)
            answer = match.group(1).strip().lower() if match else answer
        if scheme["positive"] in answer:
            label = 1
            parsed = {"answer": scheme["positive"], "raw": raw}
        elif scheme["negative"] in answer:
            label = 0
            parsed = {"answer": scheme["negative"], "raw": raw}
        else:
            label = -1
            parsed = {"answer": answer, "raw": raw}
        return label, parsed
    except Exception as exc:
        log(f"  Label error: {exc}")
        return -1, {}


def load_samples(input_path, limit=None):
    samples = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--task", type=str, default="repetitive_thinking",
                        choices=list(JUDGE_PROMPTS.keys()),
                        help="Which failure mode to label for")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    judge_prompt = JUDGE_PROMPTS[args.task]
    samples = load_samples(args.input, args.limit)
    log(f"=== Labeling {len(samples)} samples with {args.model} for {args.task} ===")
    log(f"  Input: {args.input}")
    output_path = args.input.replace(".jsonl", f"_labeled_{args.task}.jsonl")
    log(f"  Output: {output_path}")

    # Resume: skip samples already in the output file (dedup on question text).
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

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    lock = threading.Lock()
    counters = {"done": 0, "errors": 0, "positive": 0, "negative": 0}

    with open(output_path, "a", encoding="utf-8") as f:
        def process(sample):
            label, judge_out = label_one(client, sample, args.model, judge_prompt, task=args.task)
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

                if counters["done"] % 50 == 0 or counters["done"] == len(samples):
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
