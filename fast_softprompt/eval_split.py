"""Shared val/test split logic.

Cache files store `num_train` / `num_test` where `num_test` samples live at
the end of the `token_ids` / `labels` / `hidden_states` arrays. We:

1. Start with the cached held-out samples as the evaluation pool.
2. If the pool has fewer than `target_eval` samples, shift the next samples
   from the tail of the train bucket into eval (keeps class balance because
   the cache was shuffled deterministically before splitting).
3. Deterministically split the evaluation pool into val + test (default 50/50).

Training then uses `train_hidden[: num_train_used]`, model selection uses
`val_*`, and the final reported metric is computed on `test_*`.
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

    Borrows the *last* samples of the cached train bucket if the cached test
    bucket is smaller than `target_eval`. Keeps the earlier train samples as
    the actual training set.

    Returns a dict with keys:
        train_hidden, train_ids, train_labels  (lists of length `num_train_used`)
        val_hidden,   val_ids,   val_labels
        test_hidden,  test_ids,  test_labels
        num_train_used, num_val, num_test
    """
    total = len(all_labels)
    assert total == num_train_cached + num_test_cached, \
        f"cache sizes inconsistent: {total} vs {num_train_cached}+{num_test_cached}"

    needed_from_train = max(0, target_eval - num_test_cached)
    if needed_from_train > num_train_cached:
        needed_from_train = num_train_cached  # cap

    # Eval pool index range: [num_train_cached - needed_from_train, total)
    eval_start = num_train_cached - needed_from_train
    eval_indices = list(range(eval_start, total))

    # Deterministic shuffle of the eval pool, then val/test split
    rng = random.Random(seed)
    rng.shuffle(eval_indices)
    n_eval = len(eval_indices)
    n_val = int(round(val_fraction * n_eval))
    val_idx = eval_indices[:n_val]
    test_idx = eval_indices[n_val:]

    train_idx = list(range(eval_start))

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
