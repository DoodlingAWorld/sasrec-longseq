"""Component D: throughput profiler.

A simple, honest measure of how fast the pipeline moves: time a `step_fn` over a
sequence of batches and report items/second. Used to quantify the speedup from the
vectorized sampler (Component A) and to compare loss variants (Component B).
"""

from __future__ import annotations

import time
from typing import Callable

import torch


def _count_valid_items(batch: torch.Tensor) -> int:
    """Default item counter: number of non-pad entries in an int64 [B, L] tensor."""
    return int((batch != 0).sum().item())


def profile_throughput(
    step_fn: Callable[[torch.Tensor], None],
    batches: list[torch.Tensor],
    count_items: Callable[[torch.Tensor], int] = _count_valid_items,
) -> dict:
    """Time `step_fn` over `batches` and report throughput.

    Parameters
    ----------
    step_fn : callable
        The work to time, e.g. a forward+backward pass. Called as `step_fn(batch)`
        once per batch. Its return value is ignored.
    batches : list[torch.Tensor]
        Pre-built batches (each typically a [B, L] int64 id tensor). Build these
        BEFORE timing so data loading isn't counted.
    count_items : callable
        Maps a batch to its item count (default: non-pad entries).

    Returns
    -------
    metrics : dict with keys:
        "num_batches"     : int    number of batches timed
        "total_sec"       : float  wall-clock seconds across all step_fn calls
        "sec_per_batch"   : float  total_sec / num_batches
        "batches_per_sec" : float  num_batches / total_sec
        "items_per_sec"   : float  (sum of count_items over batches) / total_sec

    What to implement
    -----------------
    1. Sum the item counts across all batches (use `count_items`).
    2. Record `time.perf_counter()`, call `step_fn(batch)` for each batch, record end.
    3. Compute and return the metrics dict above. Guard against an empty `batches`
       list (return zeros rather than dividing by zero).

    Note
    ----
    For a fair number, the caller should "warm up" by running `step_fn` once before
    profiling (the scripts do this), and on CPU keep `torch.set_grad_enabled` matching
    what you're measuring (forward-only vs forward+backward).
    """
    # COMPONENT D (see EXERCISES.md): implement this. Spec: tests/test_profiling.py
    raise NotImplementedError("Component D: implement profile_throughput")
