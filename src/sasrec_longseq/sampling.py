"""Component A: vectorized negative sampling.

The Phase 1 dataset (`core/data.py::SASRecTrainDataset`) samples one negative per
position with a Python `while` loop calling `_random_neq` for every single
(user, time-step). That is O(B * L) Python-level work per batch and is a real
bottleneck on a CPU-bound pipeline.

Your job: sample the whole batch's negatives in a few vectorized tensor ops instead.
This is the canonical "feed data faster / do more with the same capacity" win.
"""

from __future__ import annotations

import torch


def sample_negatives_vectorized(
    pos: torch.Tensor, num_items: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Sample one negative item per valid position, vectorized over the whole batch.

    Parameters
    ----------
    pos : torch.Tensor
        Int64 tensor of shape [B, L] of positive (target) item ids. Padding
        positions are 0.
    num_items : int
        Number of real items. Valid item ids are 1..num_items (0 is padding).
    generator : torch.Generator, optional
        Seeded RNG for reproducibility. If None, use the global default RNG.

    Returns
    -------
    neg : torch.Tensor
        Int64 tensor of shape [B, L] where, for every VALID position (`pos != 0`):
          * `1 <= neg <= num_items`
          * `neg != pos` at that position
        and every PAD position (`pos == 0`) has `neg == 0`.

    Requirements
    ------------
    * Must be vectorized: NO Python loop over the B*L elements. (A small bounded
      loop that re-samples only the *still-colliding* entries is fine and expected,
      since you can't guarantee `neg != pos` in a single draw.)
    * Use `torch.randint(..., generator=generator)` so results are reproducible.

    Hint (the standard pattern)
    ---------------------------
    1. Draw `neg` uniformly in [1, num_items] with the same shape as `pos`.
    2. Find collisions: valid positions where `neg == pos`. While any remain,
       re-draw ONLY those entries (boolean-mask assignment).
    3. Zero out the pad positions at the end.
    """
    # COMPONENT A (see EXERCISES.md): implement this. Spec: tests/test_sampling.py
    raise NotImplementedError("Component A: implement sample_negatives_vectorized")
