"""Spec for Component B: full_softmax_loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sasrec_longseq.losses_full import full_softmax_loss


def _setup(B=2, L=3, D=4, num_items=5, seed=0):
    torch.manual_seed(seed)
    emb = nn.Embedding(num_items + 1, D, padding_idx=0)
    feats = torch.randn(B, L, D, requires_grad=True)
    pos = torch.randint(1, num_items + 1, (B, L))
    return feats, pos, emb


def test_returns_scalar():
    feats, pos, emb = _setup()
    loss = full_softmax_loss(feats, pos, emb)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gradient_flows():
    feats, pos, emb = _setup()
    loss = full_softmax_loss(feats, pos, emb)
    loss.backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()


def test_only_valid_positions_contribute():
    """ignore_index=0 + logits=feats@W.T + mean over valid positions."""
    torch.manual_seed(2)
    D, num_items = 4, 5
    emb = nn.Embedding(num_items + 1, D, padding_idx=0)
    feats = torch.randn(1, 3, D)
    pos = torch.tensor([[0, 2, 0]])  # only position 1 is a real target (item 2)

    loss = full_softmax_loss(feats, pos, emb)

    logits = feats @ emb.weight.T  # [1, 3, num_items+1]
    manual = F.cross_entropy(logits[0, 1].unsqueeze(0), torch.tensor([2]))
    assert torch.allclose(loss, manual, atol=1e-5)
