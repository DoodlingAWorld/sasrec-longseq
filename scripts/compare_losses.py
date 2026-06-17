#!/usr/bin/env python3
"""Compare the sampled-BCE objective (Phase 1) vs the full-softmax objective.

Trains the same model for a fixed number of epochs under each loss and reports
seconds/epoch and validation NDCG@10, so you can see the quality-vs-cost tradeoff.

    python scripts/compare_losses.py --epochs 20
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec_longseq.core.config import SASRecConfig  # noqa: E402
from sasrec_longseq.core.data import SASRecTrainDataset  # noqa: E402
from sasrec_longseq.core.eval import evaluate  # noqa: E402
from sasrec_longseq.core.losses import sasrec_bce_loss  # noqa: E402
from sasrec_longseq.core.model import SASRec  # noqa: E402
from sasrec_longseq.core.train import load_dataset, set_seed  # noqa: E402
from sasrec_longseq.losses_full import full_softmax_loss  # noqa: E402
from sasrec_longseq.sampling import sample_negatives_vectorized  # noqa: E402


def train_one(loss_name: str, cfg: SASRecConfig):
    set_seed(cfg.seed)
    user_train, user_valid, _user_test, _nu, num_items = load_dataset(cfg)
    ds = SASRecTrainDataset(user_train, num_items, cfg.max_len, seed=cfg.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    model = SASRec(num_items, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98))
    gen = torch.Generator().manual_seed(cfg.seed)

    t0 = time.perf_counter()
    for _epoch in range(cfg.num_epochs):
        model.train()
        for seq, pos, _neg in dl:
            if loss_name == "bce":
                neg = sample_negatives_vectorized(pos, num_items, generator=gen)
                pl, nl = model(seq, pos, neg)
                loss = sasrec_bce_loss(pl, nl, pos)
            else:  # full softmax
                feats = model.seq_repr(seq)
                loss = full_softmax_loss(feats, pos, model.item_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    sec_per_epoch = (time.perf_counter() - t0) / cfg.num_epochs

    ndcg, hit = evaluate(
        model,
        user_train,
        user_valid,
        num_items,
        cfg.max_len,
        num_neg=cfg.num_neg_eval,
        topk=cfg.topk,
        seed=cfg.seed,
    )
    return sec_per_epoch, ndcg, hit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"{'loss':>12} {'sec/epoch':>12} {'val NDCG@10':>12} {'val Hit@10':>12}")
    for name in ("bce", "full_softmax"):
        cfg = SASRecConfig(seed=args.seed, num_epochs=args.epochs)
        spe, ndcg, hit = train_one(name, cfg)
        print(f"{name:>12} {spe:>12.2f} {ndcg:>12.4f} {hit:>12.4f}")


if __name__ == "__main__":
    main()
