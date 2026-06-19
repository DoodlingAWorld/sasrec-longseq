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
    3. Compute and return the metrics dict above. If `batches` is empty, return all
       five fields as zero (no division by zero):
         {"num_batches": 0, "total_sec": 0.0, "sec_per_batch": 0.0,
          "batches_per_sec": 0.0, "items_per_sec": 0.0}

    Note
    ----
    For a fair number, the caller should "warm up" by running `step_fn` once before
    profiling (the scripts do this), and on CPU keep `torch.set_grad_enabled` matching
    what you're measuring (forward-only vs forward+backward).
    """
    if not batches:
        return {
            "num_batches": 0,
            "total_sec": 0.0,
            "sec_per_batch": 0.0,
            "batches_per_sec": 0.0,
            "items_per_sec": 0.0,
        }

    total_items = sum(count_items(batches[i]) for i in range(len(batches)))

    start = time.perf_counter()
    for batch in batches:
        step_fn(batch)
    total_sec = time.perf_counter() - start

    num_batches = len(batches)
    return {
        "num_batches": num_batches,
        "total_sec": total_sec,
        "sec_per_batch": total_sec / num_batches,
        "batches_per_sec": num_batches / total_sec,
        "items_per_sec": total_items / total_sec,
    }
