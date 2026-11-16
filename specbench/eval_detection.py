#!/usr/bin/env python3
"""Detection metrics for the new-benchmark probes.

Re-runs every trained probe over the cached test split (the identical split
the training scripts used: eval_split.make_val_test_split, seed 1337) and
computes the full confusion matrix per (benchmark, model, config):

  * accuracy / balanced accuracy
  * FNR  (漏报率)  = FN / P   - attack samples classified benign
  * FPR  (误报率)  = FP / N   - benign samples classified attack
  * precision / recall / F1 / AUROC

For BFCL-v4 the headline number is FPR over the benign (BFCL traffic) subset.

Usage:
  python specbench/eval_detection.py --task agentlab --sizes 4b,9b,mcpm \
      --probe-root results_new_benchmark --out specbench/results/detection_agentlab.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "fast_softprompt"), os.path.join(ROOT, "specbench")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cache_io import load_cache  # noqa: E402
from eval_split import make_val_test_split  # noqa: E402

DEVICE = "cuda"
BF16 = torch.bfloat16

SPECS = {
    "4b": {"kind": "mtp", "base": "models/Qwen3.5-4B", "head": "models/Qwen3.5-4B-mtp",
           "hidden": 2560, "mid_cache_suffix": "_4b_mid8"},
    "9b": {"kind": "mtp", "base": "models/Qwen3.5-9B", "head": "models/Qwen3.5-9B-mtp",
           "hidden": 4096, "mid_cache_suffix": "_9b_mid"},
    "mcpm": {"kind": "eagle3", "base": "models/MiniCPM4.1-8B",
             "head": "models/MiniCPM4.1-8B-Eagle3/MiniCPM4_1-8B-Eagle3-bf16",
             "hidden": 4096, "mid_cache_suffix": "_mcpm_mid"},
}


def _label_token_ids(base_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    pos = tok.encode("yes", add_special_tokens=False)[-1]
    neg = tok.encode("no", add_special_tokens=False)[-1]
    return pos, neg


@torch.no_grad()
def evaluate_probe(task: str, size: str, method: str, probe_root: str,
                   target_eval: int, limit: int = 0) -> dict:
    spec = SPECS[size]
    cache_path = os.path.join(ROOT, f"cache_{task}_{size}.pt")
    if size == "mcpm":
        cache_path = os.path.join(ROOT, f"cache_{task}_mcpm.pt")
    cache = load_cache(cache_path, mmap=True)
    split = make_val_test_split(
        cache["hidden_states"], cache["token_ids"], cache["labels"],
        num_train_cached=cache["num_train"], num_test_cached=cache["num_test"],
        target_eval=target_eval,
    )
    if limit:
        split = {k: (v[:limit] if isinstance(v, list) else v)
                 for k, v in split.items()}
    test_labels = split["test_labels"]
    test_idx = split["test_indices"]

    ckpt_path = os.path.join(probe_root, f"{task}_{size}", method, "best.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    preds = []

    if method == "mlp":
        from train_mlp import MLPClassifier
        clf = MLPClassifier(spec["hidden"], num_labels=2, mlp_hidden=2048, dropout=0.1)
        clf.load_state_dict(ck["state_dict"], strict=True)
        clf = clf.to(DEVICE).eval()
        h = torch.stack([s.float()[-1] for s in split["test_hidden"]]).to(DEVICE)
        bs = 64
        for i in range(0, h.size(0), bs):
            logits, _ = clf(h[i:i + bs])
            preds += logits.argmax(-1).tolist()
    elif method == "multimax":
        mid_path = os.path.join(ROOT, f"cache_{task}{spec['mid_cache_suffix']}.pt")
        mid = load_cache(mid_path, mmap=True)
        from train_multimax import MultiMaxClassifier
        clf = MultiMaxClassifier(spec["hidden"], num_heads=10, mlp_width=100, dropout=0.1)
        clf.load_state_dict(ck["state_dict"], strict=True)
        clf = clf.to(DEVICE).eval()
        n_test = len(test_labels)
        for i in range(n_test):
            # the classifier was trained on the mid-layer cache; index the same
            # held-out sample from that cache (not the last-layer test_hidden)
            hs = mid["hidden_states"][test_idx[i]].unsqueeze(0).to(DEVICE)  # [1, L, d]
            mask = torch.ones(1, hs.size(1), device=DEVICE)
            logits, _ = clf(hs, mask)
            preds.append(int(logits[0].argmax().item()))
    elif method.startswith("sp"):
        k = int(method[2:])
        if spec["kind"] == "mtp":
            from qwen35_mtp import Qwen3_5MTP
            from train_softprompt import RecursiveSoftPromptClassifier, _load_embed_weights
            mtp = Qwen3_5MTP.from_mtp_checkpoint(os.path.join(ROOT, spec["head"]),
                                                 os.path.join(ROOT, spec["base"]))
            mtp = mtp.to(DEVICE, BF16).eval()
            embed_w, lm_w = _load_embed_weights(os.path.join(ROOT, spec["base"]))
            with torch.no_grad():
                mtp.lm_head.weight = torch.nn.Parameter(lm_w.to(DEVICE, BF16))
                mtp.model.embed_tokens.weight = torch.nn.Parameter(embed_w.to(DEVICE, BF16))
            embed_w = embed_w.to(DEVICE, BF16)
            pos_id, neg_id = _label_token_ids(os.path.join(ROOT, spec["base"]))
            clf = RecursiveSoftPromptClassifier(
                mtp_model=mtp, embed_weights=embed_w, pos_token_id=pos_id,
                neg_token_id=neg_id, num_recursive_steps=k, device=DEVICE,
                classifier_head="linear", dropout=0.2,
            )
            clf.load_state_dict(ck["state_dict"], strict=False)
            clf = clf.to(DEVICE).eval()
            for i in range(len(test_labels)):
                h = split["test_hidden"][i].unsqueeze(0).to(DEVICE)
                ids = torch.tensor([split["test_ids"][i]], device=DEVICE)
                mask = torch.ones_like(ids)
                logits, _ = clf(h, ids, mask, None)
                preds.append(int(logits.argmax(-1).item()))
            del mtp, clf
            torch.cuda.empty_cache()
        else:
            import minicpm_compat  # noqa: F401
            from train_eagle3_minicpm import Eagle3SoftPromptClassifier, _load_embed_from_safetensors
            from minicpm_eagle3_wrapper import from_minicpm_eagle3_checkpoint
            e3 = from_minicpm_eagle3_checkpoint(os.path.join(ROOT, spec["head"]))
            e3 = e3.to(DEVICE, BF16).eval()
            embed_w = _load_embed_from_safetensors(os.path.join(ROOT, spec["base"])).to(DEVICE, BF16)
            clf = Eagle3SoftPromptClassifier(eagle3=e3, embed_weights=embed_w,
                                             soft_prompt_length=k, device=DEVICE,
                                             dropout=0.2)
            clf.load_state_dict(ck["state_dict"], strict=False)
            clf = clf.to(DEVICE).eval()
            e3_cache = load_cache(os.path.join(ROOT, f"cache_{task}_mcpm_eagle3.pt"), mmap=True)
            d_layer = int(e3_cache["eagle3_per_layer_dim"])
            for i in range(len(test_labels)):
                # index the same held-out sample in the Eagle3 cache
                hf = e3_cache["hidden_states"][test_idx[i]]
                h3 = tuple(hf[:, j * d_layer:(j + 1) * d_layer].unsqueeze(0).to(DEVICE)
                           for j in range(3))
                ids = torch.tensor([split["test_ids"][i]], device=DEVICE)
                mask = torch.ones_like(ids)
                logits, _ = clf(h3, ids, mask, None)
                preds.append(int(logits.argmax(-1).item()))
            del e3, clf
            torch.cuda.empty_cache()
    else:
        raise ValueError(method)

    tp = sum(1 for p, y in zip(preds, test_labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(preds, test_labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(preds, test_labels) if p == 0 and y == 1)
    tn = sum(1 for p, y in zip(preds, test_labels) if p == 0 and y == 0)
    P, N = tp + fn, fp + tn
    acc = (tp + tn) / max(len(test_labels), 1)
    fnr = fn / max(P, 1)
    fpr = fp / max(N, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(P, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "n_test": len(test_labels), "n_pos": P, "n_neg": N,
        "accuracy": acc, "balanced_accuracy": (rec + (tn / max(N, 1))) / 2,
        "FNR": fnr, "FPR": fpr, "precision": prec, "recall": rec, "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--sizes", default="4b,9b,mcpm")
    ap.add_argument("--methods", default="sp1,sp2,sp5,mlp,multimax")
    ap.add_argument("--probe-root", default="results_new_benchmark")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N test samples (smoke test)")
    args = ap.parse_args()

    targets = {"agentlab": 800, "bfcl_v4": 800, "livecodebench": 600}
    results = {}
    for size in args.sizes.split(","):
        for method in args.methods.split(","):
            label = f"{args.task}_{size}/{method}"
            try:
                r = evaluate_probe(args.task, size, method, args.probe_root,
                                   targets[args.task], args.limit)
                results[f"{size}/{method}"] = r
                print(f"{label}: acc={r['accuracy']:.4f} FNR={r['FNR']:.4f} FPR={r['FPR']:.4f} f1={r['f1']:.4f}")
            except Exception as e:
                import traceback
                print(f"{label}: SKIP ({type(e).__name__}: {e})")
                traceback.print_exc()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
