#!/usr/bin/env python3
"""Throughput comparison: loop-based vs vectorized negative sampling.

Builds real ML-1M training batches and times (a) the forward+backward step and
(b) the two negative-sampling strategies, reporting items/second for each.

    python scripts/profile_throughput.py --batches 20
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec_longseq.core.config import SASRecConfig  # noqa: E402
from sasrec_longseq.core.data import (  # noqa: E402
    SASRecTrainDataset,
    _random_neq,
)
from sasrec_longseq.core.model import SASRec  # noqa: E402
from sasrec_longseq.core.train import load_dataset, set_seed  # noqa: E402
from sasrec_longseq.profiling import profile_throughput  # noqa: E402
from sasrec_longseq.sampling import sample_negatives_vectorized  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = SASRecConfig(seed=args.seed)
    set_seed(cfg.seed)
    user_train, _, _, _, num_items = load_dataset(cfg)
    ds = SASRecTrainDataset(user_train, num_items, cfg.max_len, seed=cfg.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    # Pre-build a few (seq, pos) batches so data loading isn't timed.
    batches = []
    for i, (seq, pos, _neg) in enumerate(dl):
        batches.append((seq, pos))
        if i + 1 >= args.batches:
            break

    pos_batches = [pos for _seq, pos in batches]
    rng = torch.Generator().manual_seed(cfg.seed)

    # (1) vectorized sampler throughput
    vec = profile_throughput(
        lambda pos: sample_negatives_vectorized(pos, num_items, generator=rng),
        pos_batches,
    )

    # (2) loop sampler throughput (the Phase-1 per-position approach)
    import numpy as np

    np_rng = np.random.default_rng(cfg.seed)

    def loop_sampler(pos):
        arr = pos.numpy()
        out = np.zeros_like(arr)
        for b in range(arr.shape[0]):
            interacted = set(arr[b].tolist())
            for t in range(arr.shape[1]):
                if arr[b, t] != 0:
                    out[b, t] = _random_neq(1, num_items + 1, interacted, np_rng)
        return torch.from_numpy(out)

    loop = profile_throughput(loop_sampler, pos_batches)

    # (3) model forward+backward throughput
    model = SASRec(num_items, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    from sasrec_longseq.core.losses import sasrec_bce_loss

    def train_step(args_tuple):
        seq, pos = args_tuple
        neg = sample_negatives_vectorized(pos, num_items, generator=rng)
        pl, nl = model(seq, pos, neg)
        loss = sasrec_bce_loss(pl, nl, pos)
        opt.zero_grad()
        loss.backward()
        opt.step()

    train_step(batches[0])  # warm up
    fb = profile_throughput(
        train_step, batches, count_items=lambda t: int((t[1] != 0).sum().item())
    )

    print(f"\nNegative sampling throughput (items/sec):")
    print(f"  vectorized : {vec['items_per_sec']:>12,.0f}")
    print(f"  loop       : {loop['items_per_sec']:>12,.0f}")
    if loop["items_per_sec"] > 0:
        print(f"  speedup    : {vec['items_per_sec'] / loop['items_per_sec']:>11.1f}x")
    print(
        f"\nForward+backward step: {fb['items_per_sec']:>12,.0f} items/sec "
        f"({fb['sec_per_batch']*1000:.1f} ms/batch)"
    )


if __name__ == "__main__":
    main()
