#!/usr/bin/env python3
"""
Fast MLP probe training using precomputed hidden states.
Only trains a small MLP classifier on cached last hidden states.

Usage:
  python fast_softprompt/train_mlp.py \
    --cache fast_softprompt/cache_wildchat_9b.pt \
    --output_dir wildchat_output_mlp_9b_fast \
    --lr 5e-4 --epochs 10
"""

from __future__ import annotations

import argparse
import math
import os
import json
import random
import sys
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANDOM_SEED = 42


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class MLPClassifier(nn.Module):
    def __init__(self, hidden_dim, num_labels=2, mlp_hidden=1024, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_labels),
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, hidden, labels=None):
        logits = self.mlp(hidden.float())
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--target_eval", type=int, default=200)
    parser.add_argument("--val_fraction", type=float, default=0.5)
    parser.add_argument("--save_epoch_ckpts", action="store_true")
    parser.add_argument("--mmap_cache", action="store_true",
                        help="mmap hidden_states (requires mmap_v1 cache format).")
    args = parser.parse_args()

    set_seed()

    print(f"Loading cache from {args.cache} (mmap={args.mmap_cache})...")
    from cache_io import load_cache
    cache = load_cache(args.cache, mmap=args.mmap_cache)
    all_ids = cache["token_ids"]
    all_labels = cache["labels"]
    all_hidden = cache["hidden_states"]
    base_model_name = cache["model"]
    task = cache["task"]

    from eval_split import make_val_test_split
    split = make_val_test_split(
        all_hidden, all_ids, all_labels,
        num_train_cached=cache["num_train"],
        num_test_cached=cache["num_test"],
        target_eval=args.target_eval,
        val_fraction=args.val_fraction,
    )
    num_train = split["num_train_used"]
    num_val = split["num_val"]
    num_test = split["num_test"]
    print(f"  Task: {task}, Model: {base_model_name}")
    print(f"  Train: {num_train}, Val: {num_val}, Test: {num_test} "
          f"(borrowed {split['borrowed_from_train']} from train -> eval)")

    # Extract last-token hidden state for each sample
    device = "cuda"
    print("Extracting last-token hidden states...")
    def last_tokens(arr):
        return torch.stack([h[-1] for h in arr])
    train_features = last_tokens(split["train_hidden"]).to(device)
    train_labels = torch.tensor(split["train_labels"], dtype=torch.long, device=device)
    val_features = last_tokens(split["val_hidden"]).to(device)
    val_labels = torch.tensor(split["val_labels"], dtype=torch.long, device=device)
    test_features = last_tokens(split["test_hidden"]).to(device)
    test_labels = torch.tensor(split["test_labels"], dtype=torch.long, device=device)

    hidden_dim = train_features.size(1)
    print(f"  Feature dim: {hidden_dim}, Train: {train_features.size(0)}, Val: {val_features.size(0)}, Test: {test_features.size(0)}")

    target_names = {
        "wildchat": ["same_topic", "different_topic"],
        "verih": ["aligned", "conflict"],
        "longsafety": ["safe", "unsafe"],
        "pii": ["no_pii", "pii"],
        "adversarial_if": ["correct", "incorrect"],
        "nemotron_safety": ["safe", "unsafe"],
        "cot_repetitive": ["stay", "repetitive"],
        "cot_branched": ["stay", "branch"],
    }.get(task, ["class_0", "class_1"])

    classifier = MLPClassifier(hidden_dim, num_labels=2, mlp_hidden=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)

    best_eval_acc = -1.0
    best_epoch_metrics = None
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        classifier.train()
        perm = torch.randperm(num_train, device=device)
        train_f = train_features[perm]
        train_l = train_labels[perm]

        running_loss, correct, seen = 0.0, 0, 0
        num_batches = math.ceil(num_train / args.batch_size)
        recent_correct = deque(maxlen=20)

        pbar = tqdm(range(0, num_train, args.batch_size), total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}")
        for s in pbar:
            e = min(s + args.batch_size, num_train)
            optimizer.zero_grad()
            logits, loss = classifier(train_f[s:e], train_l[s:e])
            if not torch.isfinite(loss):
                # A handful of cached 27B hidden states are non-finite (NaN).
                # Skip the batch so one bad sample cannot poison every weight.
                continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                batch_matches = (preds == train_l[s:e])
                correct += batch_matches.sum().item()
                seen += (e - s)
                recent_correct.extend(batch_matches.cpu().tolist())
            rolling_acc = (sum(recent_correct) / len(recent_correct)) if recent_correct else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{rolling_acc:.3f}"})

        # Val eval (for model selection)
        def eval_on(feat, lbl):
            classifier.eval()
            with torch.no_grad():
                lg, _ = classifier(feat, None)
                pr = torch.argmax(lg, dim=-1).cpu().tolist()
            gd = lbl.cpu().tolist()
            a = accuracy_score(gd, pr)
            p, r, f, _ = precision_recall_fscore_support(gd, pr, average="binary", zero_division=0)
            return pr, gd, a, float(p), float(r), float(f)

        _, _, val_acc, val_pr, val_rc, val_f1 = eval_on(val_features, val_labels)

        row = {"epoch": epoch + 1, "train_loss": running_loss / max(num_batches, 1), "train_acc": correct / max(seen, 1), "val_acc": val_acc}
        history.append(row)

        if args.output_dir and args.save_epoch_ckpts:
            ckpt_dir = os.path.join(args.output_dir, "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch + 1, "val_acc": val_acc,
                "state_dict": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
            }, os.path.join(ckpt_dir, f"epoch_{epoch+1:03d}.pt"))

        marker = ""
        if val_acc > best_eval_acc:
            best_eval_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in classifier.state_dict().items()}
            marker = " *best*"
        print(f"Epoch {epoch+1}: loss={row['train_loss']:.4f}, train_acc={row['train_acc']:.4f}, val_acc={val_acc:.4f}{marker}")

    # Final test eval
    if best_epoch > 0:
        classifier.load_state_dict(best_state)
    tp, tg, test_acc, test_pr, test_rc, test_f1 = eval_on(test_features, test_labels)
    vp, vg, vA, vP, vR, vF = eval_on(val_features, val_labels)
    print(f"\nBest epoch (by val): {best_epoch} val_acc={best_eval_acc:.4f}")
    print(f"Test  | Acc: {test_acc:.4f} | P: {test_pr:.4f} | R: {test_rc:.4f} | F1: {test_f1:.4f}")
    print(classification_report(tg, tp, target_names=target_names,
                                labels=list(range(len(target_names))), zero_division=0))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "config": {
                "task": task, "arch": "mlp_cached", "base_model": base_model_name,
                "num_epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.lr, "mlp_hidden": args.hidden_dim,
                "dropout": args.dropout, "best_epoch": best_epoch,
                "num_train": num_train, "num_val": num_val, "num_test": num_test,
            },
            "training_history": history,
            "val_metrics": {"accuracy": vA, "precision": vP, "recall": vR, "f1": vF},
            "metrics": {"accuracy": test_acc, "precision": test_pr, "recall": test_rc, "f1": test_f1},
        }
        out_path = os.path.join(args.output_dir, "final_results.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved results to: {out_path}")


if __name__ == "__main__":
    main()
