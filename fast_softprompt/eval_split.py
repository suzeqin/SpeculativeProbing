"""Shared val/test split logic.

Cache files store `num_train` / `num_test` where `num_test` held-out samples
live at the end of the `token_ids` / `labels` / `hidden_states` arrays. We:

1. Use the cached held-out samples as the evaluation pool (never borrow from
   the train bucket, which would unbalance a 50/50 balanced dataset).
2. Deterministically split the pool into val + test with a stratified shuffle
   so each split stays class-balanced (default 50/50).

Training uses `train_hidden[: num_train_used]`, model selection uses `val_*`,
and the final reported metric is computed on `test_*`.
"""
from __future__ import annotations

import random
from typing import Any, List, Tuple


def make_val_test_split(
    all_hidden: List[Any],
    all_ids: List[Any],
    all_labels: List[int],
    num_train_cached: int,
    num_test_cached: int,
    target_eval: int = 200,
    val_fraction: float = 0.5,
    seed: int = 1337,
):
    """Split the cached data into (train, val, test) slices.

    Uses only the cached held-out samples as the eval pool (never borrows from
    train); val/test are stratified so both stay class-balanced.

    Returns a dict with keys:
        train_hidden, train_ids, train_labels  (lists of length `num_train_used`)
        val_hidden,   val_ids,   val_labels
        test_hidden,  test_ids,  test_labels
        num_train_used, num_val, num_test
    """
    total = len(all_labels)
    assert total == num_train_cached + num_test_cached, \
        f"cache sizes inconsistent: {total} vs {num_train_cached}+{num_test_cached}"

    # Use only the cached held-out samples (the balanced sampler guarantees
    # this pool is class-balanced); never borrow from train, which would break
    # the balance when fewer positives are available.
    eval_start = num_train_cached
    eval_indices = list(range(eval_start, total))

    # Stratified deterministic val/test split so both stay class-balanced.
    rng = random.Random(seed)
    pos = [i for i in eval_indices if all_labels[i] == 1]
    neg = [i for i in eval_indices if all_labels[i] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pair = min(len(pos), len(neg))
    n_val_pair = int(round(val_fraction * n_pair))
    val_idx = pos[:n_val_pair] + neg[:n_val_pair]
    test_idx = pos[n_val_pair:n_pair] + neg[n_val_pair:n_pair]
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_idx = list(range(eval_start))
    needed_from_train = 0

    def pick(arr, idxs):
        return [arr[i] for i in idxs]

    out = dict(
        train_hidden=pick(all_hidden, train_idx),
        train_ids=pick(all_ids, train_idx),
        train_labels=pick(all_labels, train_idx),
        val_hidden=pick(all_hidden, val_idx),
        val_ids=pick(all_ids, val_idx),
        val_labels=pick(all_labels, val_idx),
        test_hidden=pick(all_hidden, test_idx),
        test_ids=pick(all_ids, test_idx),
        test_labels=pick(all_labels, test_idx),
        num_train_used=len(train_idx),
        num_val=len(val_idx),
        num_test=len(test_idx),
        borrowed_from_train=needed_from_train,
    )
    return out
