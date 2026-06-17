"""Spec for Component A: sample_negatives_vectorized."""

import torch

from sasrec_longseq.sampling import sample_negatives_vectorized


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_shape_and_dtype():
    pos = torch.randint(1, 50, (4, 7))
    neg = sample_negatives_vectorized(pos, num_items=49, generator=_gen())
    assert neg.shape == pos.shape
    assert neg.dtype == pos.dtype  # int64


def test_pad_positions_are_zero():
    pos = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
    neg = sample_negatives_vectorized(pos, num_items=10, generator=_gen())
    assert (neg[pos == 0] == 0).all()


def test_valid_positions_in_range_and_differ_from_pos():
    pos = torch.randint(1, 100, (16, 20))
    neg = sample_negatives_vectorized(pos, num_items=99, generator=_gen(1))
    valid = pos != 0
    assert (neg[valid] >= 1).all()
    assert (neg[valid] <= 99).all()
    assert (neg[valid] != pos[valid]).all()


def test_reproducible_with_seeded_generator():
    pos = torch.randint(1, 50, (8, 8))
    a = sample_negatives_vectorized(pos, 49, generator=_gen(123))
    b = sample_negatives_vectorized(pos, 49, generator=_gen(123))
    assert torch.equal(a, b)


def test_resamples_when_only_one_valid_negative_exists():
    # num_items=2 and every target is 1 => the only valid negative is 2.
    pos = torch.ones(3, 3, dtype=torch.long)
    neg = sample_negatives_vectorized(pos, num_items=2, generator=_gen(7))
    assert (neg == 2).all()
