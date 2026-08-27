#!/usr/bin/env python3
"""
MultiMax probe training using precomputed hidden states.
Based on Kramár et al. 2025, "Building Production-Ready Probes For Gemini".

Unlike MLP probes (last-token only), MultiMax uses the full sequence
of hidden states with hard-max aggregation per head.

Usage:
  python fast_softprompt/train_multimax.py \
    --cache fast_softprompt/cache_cot_repetitive_9b.pt \
    --output_dir results/multimax_branched_9b \
    --num_heads 10 --mlp_width 100 --epochs 10
"""

from __future__ import annotations

import argparse
import math
import os
import json
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm

RANDOM_SEED = 42


def set_seed():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


class MultiMaxClassifier(nn.Module):
    """
    MultiMax probe: MLP transform + H learned value vectors + hard-max aggregation.

    For each head h, computes max_j(v_h^T y_j) over all token positions j,
    then feeds the H-dim aggregated vector through a classification head.
    """

    def __init__(self, hidden_dim, num_heads=10, mlp_width=100, num_labels=2, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.mlp_width = mlp_width

        # MLP transform: hidden_dim -> mlp_width -> mlp_width
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
        )

        # H learned value vectors: (num_heads, mlp_width)
        self.value_vectors = nn.Parameter(
            torch.randn(num_heads, mlp_width) / (mlp_width ** 0.5)
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_heads, num_labels),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, hidden_states, attention_mask, labels=None):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len) - 1 for valid, 0 for pad
            labels: (batch,) - optional
        """
        # MLP transform: (batch, seq_len, hidden_dim) -> (batch, seq_len, mlp_width)
        y = self.mlp(hidden_states.float())

        # Compute scores: (batch, seq_len, num_heads) via einsum
        scores = torch.einsum('bsd,hd->bsh', y, self.value_vectors)

        # Mask invalid positions
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
            scores = scores + (1.0 - mask) * (-1e9)

        # Hard max per head: (batch, num_heads)
        aggregated = scores.max(dim=1).values

        # Classification
        logits = self.head(aggregated)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return logits, loss


def pad_hidden_states(batch_hidden, batch_labels):
    """Pad variable-length hidden state sequences into tensors."""
    max_len = max(h.size(0) for h in batch_hidden)
    hidden_dim = batch_hidden[0].size(1)
    B = len(batch_hidden)

    hidden_padded = torch.zeros(B, max_len, hidden_dim, dtype=batch_hidden[0].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.long)

    for i, h in enumerate(batch_hidden):
        seq_len = h.size(0)
        hidden_padded[i, :seq_len] = h
        mask[i, :seq_len] = 1

    labels = torch.tensor(batch_labels, dtype=torch.long)
    return hidden_padded, mask, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=10)
    parser.add_argument("--mlp_width", type=int, default=100)
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
    all_labels = cache["labels"]
    all_hidden = cache["hidden_states"]
    all_ids = cache.get("token_ids", all_labels)  # multimax doesn't need token_ids, use a placeholder
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

    train_hidden = split["train_hidden"]
    train_labels = split["train_labels"]
    val_hidden = split["val_hidden"]
    val_labels = split["val_labels"]
    test_hidden = split["test_hidden"]
    test_labels = split["test_labels"]

    hidden_dim = train_hidden[0].size(1)
    print(f"  Hidden dim: {hidden_dim}")

    device = "cuda"
    classifier = MultiMaxClassifier(
        hidden_dim, num_heads=args.num_heads, mlp_width=args.mlp_width, dropout=args.dropout
    ).to(device)
    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"  MultiMax params: {total_params:,} (heads={args.num_heads}, width={args.mlp_width})")

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=3e-3)

    target_names = ["class_0", "class_1"]

    best_eval_acc = -1.0
    best_epoch_metrics = None
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        classifier.train()
        perm = list(range(num_train))
        random.shuffle(perm)

        running_loss, correct, seen = 0.0, 0, 0
        num_batches = math.ceil(num_train / args.batch_size)
        recent_correct = deque(maxlen=20)

        pbar = tqdm(range(0, num_train, args.batch_size), total=num_batches, desc=f"Epoch {epoch+1}/{args.epochs}")
        for s in pbar:
            e = min(s + args.batch_size, num_train)
            batch_idx = perm[s:e]
            batch_h = [train_hidden[i] for i in batch_idx]
            batch_l = [train_labels[i] for i in batch_idx]

            hidden_padded, mask, labels = pad_hidden_states(batch_h, batch_l)
            hidden_padded = hidden_padded.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, loss = classifier(hidden_padded, mask, labels)
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                batch_matches = (preds == labels)
                correct += batch_matches.sum().item()
                seen += labels.numel()
                recent_correct.extend(batch_matches.cpu().tolist())
            rolling_acc = (sum(recent_correct) / len(recent_correct)) if recent_correct else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{rolling_acc:.3f}"})

        def run_eval(hiddens, lbls):
            classifier.eval()
            prs, gds = [], []
            with torch.no_grad():
                for s in range(0, len(lbls), args.batch_size):
                    e = min(s + args.batch_size, len(lbls))
                    hp, mk, lb = pad_hidden_states(hiddens[s:e], lbls[s:e])
                    hp = hp.to(device); mk = mk.to(device)
                    lg, _ = classifier(hp, mk, None)
                    prs.extend(torch.argmax(lg, dim=-1).cpu().tolist())
                    gds.extend(lb.tolist())
            a = accuracy_score(gds, prs)
            p, r, f, _ = precision_recall_fscore_support(gds, prs, average="binary", zero_division=0)
            return prs, gds, a, float(p), float(r), float(f)

        _, _, val_acc, val_pr, val_rc, val_f1 = run_eval(val_hidden, val_labels)

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

    if best_epoch > 0:
        classifier.load_state_dict(best_state)
    tp, tg, test_acc, test_pr, test_rc, test_f1 = run_eval(test_hidden, test_labels)
    vp, vg, vA, vP, vR, vF = run_eval(val_hidden, val_labels)
    print(f"\nBest epoch (by val): {best_epoch} val_acc={best_eval_acc:.4f}")
    print(f"Test  | Acc: {test_acc:.4f} | P: {test_pr:.4f} | R: {test_rc:.4f} | F1: {test_f1:.4f}")
    print(classification_report(tg, tp, target_names=target_names,
                                labels=list(range(len(target_names))), zero_division=0))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out = {
            "config": {
                "task": task, "arch": "multimax", "base_model": base_model_name,
                "num_epochs": args.epochs, "batch_size": args.batch_size,
                "learning_rate": args.lr, "num_heads": args.num_heads,
                "mlp_width": args.mlp_width, "dropout": args.dropout,
                "best_epoch": best_epoch,
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
