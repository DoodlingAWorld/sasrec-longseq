#!/usr/bin/env python3
"""Sequence-length scaling study (reproduces the paper's Table V).

Sweeps max sequence length, records seconds/epoch and validation NDCG, prints a
table, and (if matplotlib is available) saves a cost-vs-quality plot.

    python scripts/run_scaling.py                       # default sweep
    python scripts/run_scaling.py --n 10 50 100 200 --epochs 20
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sasrec_longseq.scaling import measure_scaling  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n", type=int, nargs="+", default=[10, 50, 100, 200, 300, 400, 500, 600]
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="experiments/runs")
    args = p.parse_args()

    results = measure_scaling(args.n, epochs=args.epochs, seed=args.seed)

    print(f"\n{'n':>6} {'sec/epoch':>12} {'val NDCG@10':>12}")
    for r in results:
        print(f"{r['n']:>6} {r['sec_per_epoch']:>12.2f} {r['val_ndcg']:>12.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "scaling.json"), "w") as f:
        json.dump(results, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ns = [r["n"] for r in results]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(
            ns,
            [r["sec_per_epoch"] for r in results],
            "o-",
            color="tab:red",
            label="sec/epoch",
        )
        ax1.set_xlabel("max sequence length n")
        ax1.set_ylabel("seconds / epoch", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(
            ns,
            [r["val_ndcg"] for r in results],
            "s-",
            color="tab:blue",
            label="val NDCG@10",
        )
        ax2.set_ylabel("val NDCG@10", color="tab:blue")
        fig.tight_layout()
        out_png = os.path.join(args.out_dir, "scaling.png")
        fig.savefig(out_png, dpi=120)
        print(f"\nSaved plot -> {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
