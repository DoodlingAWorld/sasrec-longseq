"""Component C: sequence-length scaling harness (reproduces the paper's Table V).

The paper shows training cost grows with the maximum sequence length n while NDCG
saturates around n=500. This harness sweeps n, trains the model for a fixed number
of epochs at each, and records cost (seconds per epoch) vs. quality (val NDCG).
"""

from __future__ import annotations

import time
from typing import Callable

from .core.config import SASRecConfig
from .core.train import train as core_train


def measure_scaling(
    n_values: list[int],
    epochs: int = 20,
    seed: int = 42,
    hidden_dim: int = 50,
    num_blocks: int = 2,
    train_fn: Callable[[SASRecConfig], dict] | None = None,
) -> list[dict]:
    """Train the model at several max sequence lengths and record cost vs. quality.

    Parameters
    ----------
    n_values : list[int]
        Maximum sequence lengths to sweep, e.g. [10, 50, 100, 200, 300, 400, 500, 600].
    epochs : int
        Epochs to train at each n (keep modest; this is a relative comparison).
    seed, hidden_dim, num_blocks : int
        Fixed model/training settings held constant across the sweep.
    train_fn : callable, optional
        Injected for testing. `train_fn(cfg) -> dict` must return a dict containing
        at least "best_val_ndcg". Defaults to the vendored core training routine.

    Returns
    -------
    results : list[dict]
        One dict per n, IN THE SAME ORDER as `n_values`, each with keys:
          {"n": int, "sec_per_epoch": float, "val_ndcg": float}

    What to implement
    -----------------
    For each n in n_values:
      1. Build a `SASRecConfig(max_len=n, num_epochs=epochs, hidden_dim=hidden_dim,
         num_blocks=num_blocks, seed=seed)`.
      2. Time how long `train_fn(cfg)` takes (use `time.perf_counter()`), and divide
         by `epochs` to get `sec_per_epoch`.
      3. Pull `best_val_ndcg` from the returned dict as `val_ndcg`.
      4. Append the result dict.
    Return the list.

    Expected shape of the result (the Table V story): `sec_per_epoch` rises with n
    (roughly with the O(n^2) attention term), while `val_ndcg` improves then plateaus.
    """
    # COMPONENT C (see EXERCISES.md): implement this. Spec: tests/test_scaling.py
    raise NotImplementedError("Component C: implement measure_scaling")
