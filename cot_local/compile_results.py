#!/usr/bin/env python3
"""Compile the reproduced WebInstruct-verified (CoT) benchmark numbers and
compare them with the paper's Tables 3 (CoT Repetition) and 4 (CoT Reasoning
Strategy). Reads <results_dir>/<task>_<model>/<method>/final_results.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASKS = ["cot_repetitive", "cot_branched"]
TASK_TITLES = {
    "cot_repetitive": "Table 3 - CoT Repetition (WebInstruct-verified traces)",
    "cot_branched": "Table 4 - CoT Reasoning Strategy / branched (WebInstruct-verified traces)",
}
MODELS = [("4b", "Qwen3.5-4B"), ("9b", "Qwen3.5-9B"), ("mcpm", "MiniCPM4.1-8B")]
METHODS = ["mlp", "multimax", "sp1", "sp2", "sp5"]
METHOD_TITLES = {"mlp": "MLP", "multimax": "MM", "sp1": "SP-1", "sp2": "SP-2", "sp5": "SP-5"}

# Paper reported test accuracies (Tables 3 & 4 of "Speculative Probing").
PAPER = {
    "cot_repetitive": {
        "4b":  {"mlp": 87.5, "multimax": 87.0, "sp1": 88.0, "sp2": 87.5, "sp5": 89.0},
        "9b":  {"mlp": 89.5, "multimax": 87.5, "sp1": 91.5, "sp2": 89.5, "sp5": 89.5},
        "mcpm": {"mlp": 83.5, "multimax": 82.5, "sp1": 86.0, "sp2": 85.0, "sp5": 87.0},
    },
    "cot_branched": {
        "4b":  {"mlp": 73.0, "multimax": 85.5, "sp1": 83.0, "sp2": 81.5, "sp5": 82.5},
        "9b":  {"mlp": 76.5, "multimax": 87.0, "sp1": 84.5, "sp2": 83.0, "sp5": 83.5},
        "mcpm": {"mlp": 68.0, "multimax": 78.5, "sp1": 66.5, "sp2": 65.5, "sp5": 64.5},
    },
}


def read_acc(results_dir, task, model, method):
    path = os.path.join(results_dir, f"{task}_{model}", method, "final_results.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        data = json.load(f)
    # "metrics" holds the test metrics at the best-val epoch (headline number).
    acc = (data.get("metrics") or {}).get("accuracy", data.get("test_accuracy"))
    return (None if acc is None else round(100 * acc, 2) if acc <= 1 else round(acc, 2)), path


def main(results_dir="results"):
    for task in TASKS:
        print(f"\n=== {TASK_TITLES[task]} ===")
        print(f"{'Model':<16} {'':>2} {'MLP':>12} {'MM':>12} {'SP-1':>12} {'SP-2':>12} {'SP-5':>12}")
        for model, mname in MODELS:
            row = []
            for method in METHODS:
                acc, path = read_acc(os.path.join(ROOT, results_dir), task, model, method)
                paper = PAPER[task][model][method]
                if acc is None:
                    row.append(f"{'--':>5}/{paper:<5.1f}")
                else:
                    row.append(f"{acc:>5.1f}/{paper:<5.1f}")
            print(f"{mname:<16}   " + " ".join(f"{c:>12}" for c in row))
        print("  (cell format: reproduced/paper)")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
