"""Component B: full-vocabulary softmax loss.

The Phase 1 objective (`core/losses.py`) is binary cross-entropy over (positive,
ONE sampled negative) per step: cheap (O(1) negatives) but an approximation of the
true next-item distribution. The classic alternative is a **full softmax**: score
EVERY item and apply cross-entropy. Exact, but O(|I|) compute and memory per step.

Implementing both lets us compare them head-to-head (quality vs. speed vs. memory),
which is the experiment this repo is built around.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def full_softmax_loss(
    feats: torch.Tensor, pos: torch.Tensor, item_embedding: nn.Embedding
) -> torch.Tensor:
    """Cross-entropy over the full item vocabulary.

    Parameters
    ----------
    feats : torch.Tensor
        [B, L, D] sequence representations (the output of `SASRec.seq_repr`).
    pos : torch.Tensor
        [B, L] int64 positive next-item ids. Padding positions are 0 and must be
        IGNORED in the loss.
    item_embedding : nn.Embedding
        The model's (tied) item embedding table, weight shape [num_items + 1, D].
        Row 0 is the padding embedding.

    Returns
    -------
    loss : torch.Tensor
        A scalar. Cross-entropy averaged over the valid (non-pad) positions.

    What to compute
    ---------------
    1. Score every item at every position:
         logits = feats @ item_embedding.weight.T        # [B, L, num_items + 1]
    2. Flatten to [B*L, num_items + 1] and `pos` to [B*L], then use
       `F.cross_entropy(..., ignore_index=0)` so pad targets (id 0) are dropped
       AND item id 0 is never a valid class to predict.

    Why it matters
    --------------
    This is exact (no sampling noise) but the logits tensor is [B, L, |I|+1], which
    is why full softmax is memory/compute-heavy for large catalogs — the very reason
    sampled losses exist. You'll quantify that tradeoff in the experiments.
    """
    # COMPONENT B (see EXERCISES.md): implement this. Spec: tests/test_losses_full.py
    raise NotImplementedError("Component B: implement full_softmax_loss")
