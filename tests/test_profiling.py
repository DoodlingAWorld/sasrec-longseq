"""Spec for Component D: profile_throughput."""

import torch

from sasrec_longseq.profiling import profile_throughput


def test_metrics_structure_and_math():
    batches = [
        torch.ones(2, 5, dtype=torch.long) for _ in range(3)
    ]  # 10 items each -> 30
    metrics = profile_throughput(lambda b: None, batches)

    assert metrics["num_batches"] == 3
    assert metrics["total_sec"] >= 0.0
    # internal consistency
    assert abs(metrics["sec_per_batch"] * 3 - metrics["total_sec"]) < 1e-6
    if metrics["total_sec"] > 0:
        assert abs(metrics["items_per_sec"] * metrics["total_sec"] - 30) < 1e-3


def test_step_fn_called_once_per_batch():
    calls = []
    batches = [torch.ones(1, 3, dtype=torch.long) for _ in range(4)]
    profile_throughput(lambda b: calls.append(1), batches)
    assert len(calls) == 4


def test_empty_batches_no_zero_division():
    metrics = profile_throughput(lambda b: None, [])
    assert metrics["num_batches"] == 0
    assert metrics["total_sec"] == 0.0
    assert metrics["sec_per_batch"] == 0.0
    assert metrics["items_per_sec"] == 0.0
    assert metrics["batches_per_sec"] == 0.0
