#!/usr/bin/env python3
"""Generate WebInstruct-verified CoT traces via a locally served OpenAI-compatible model.

Thin wrapper around cot_inconsistency/generate_labels_webinstruct_verified.py.
The original hardcodes the DashScope (Bailian) endpoint for qwen3.5-plus; here we
point the same generation logic at the local vLLM server so the traces come from
a locally deployed large model instead.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cot_inconsistency"))

os.environ.setdefault("DASHSCOPE_API_KEY", "local-vllm")

import generate_labels_webinstruct_verified as g  # noqa: E402
import re  # noqa: E402
import httpx  # noqa: E402
import functools  # noqa: E402
from openai import OpenAI  # noqa: E402

# Long generations under heavy server concurrency can exceed the OpenAI client's
# default 600s timeout; give local requests a generous budget.
g.OpenAI = functools.partial(
    OpenAI,
    timeout=httpx.Timeout(3600.0, connect=60.0),
    max_retries=3,
)


def _extract_cot_and_response_robust(text):
    # A locally served model may emit the reasoning without the opening
    # "<think>" tag (the chat template can insert it into the prompt instead);
    # normalize so the original regex still works.
    if "<think>" not in text and "</think>" in text:
        text = "<think>" + text
    match = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if not match:
        return None, None
    cot = match.group(1).strip()
    response = match.group(2).strip()
    return (cot, response) if cot and response else (None, None)


def _generate_one(client, question):
    # Same as the original generate_one but with an 8192-token budget: the
    # downstream pipeline keeps only traces whose <think> section fits in 8192
    # tokens (precompute --max_length 8192), and with the default 4096 cap most
    # local traces were truncated mid-thinking and discarded.
    try:
        resp = client.chat.completions.create(
            model=g.BAILIAN_MODEL,
            messages=[{"role": "user", "content": question}],
            max_completion_tokens=8192,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            return reasoning, content
        return g.extract_cot_and_response(content)
    except Exception as e:
        g.log(f"  Generation error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen3.5-122B-A10B-GPTQ-Int4")
    parser.add_argument("--num-samples", type=int, default=18000)
    parser.add_argument("--dataset-split", default=os.environ.get("WEBINSTRUCT_SPLIT", "train"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    g.BAILIAN_BASE_URL = args.base_url
    g.BAILIAN_MODEL = args.model
    g.extract_cot_and_response = _extract_cot_and_response_robust
    g.generate_one = _generate_one

    os.makedirs(g.DATA_DIR, exist_ok=True)
    questions = g.download_webinstruct_verified(
        dataset_split=args.dataset_split,
        num_samples=args.num_samples,
    )
    g.generate_samples(
        questions=questions,
        dataset_split=args.dataset_split,
        workers=args.workers,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()
