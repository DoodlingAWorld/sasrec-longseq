# sasrec-longseq: efficiency and longer sequences for SASRec

An efficiency-focused extension of [SASRec](https://arxiv.org/abs/1808.09781), built on top of
the faithful reproduction in [`sasrec-pytorch`](https://github.com/DoodlingAWorld/sasrec-pytorch).
Sequential recommenders are moving toward feeding longer and longer user histories, so this repo
studies the cost side of that: how training scales with sequence length, how to feed data faster,
and how the choice of training objective trades quality for compute.

## What's here

1. **Vectorized negative sampling** (`sampling.py`): replaces SASRec's per-position Python
   sampling loop with a batched tensor implementation, and measures the speedup.
2. **Full-softmax vs sampled-BCE loss** (`losses_full.py`): the exact next-item objective vs the
   one-negative approximation, compared on quality, speed, and memory.
3. **Sequence-length scaling** (`scaling.py`): reproduces the paper's Table V (cost grows with the
   max length `n`; quality saturates around `n = 500`).
4. **Throughput profiling** (`profiling.py`): items/second for the data path and the training step.

The SASRec model, data pipeline, baseline loss, and evaluation are vendored under
`src/sasrec_longseq/core/` from the sibling repo so this project clones and runs standalone.

## Results

Measured on CPU with MovieLens-1M (default config: n=200, d=50, b=2).

| Study | Result |
|---|---|
| Negative sampling: loop vs vectorized | **74.2x faster** (31.4M vs 0.42M items/sec); see `scripts/profile_throughput.py` |
| Loss: sampled-BCE vs full-softmax (val NDCG@10 / sec per epoch) | _running (`scripts/compare_losses.py`)_ |
| Sequence-length scaling (Table V) | _running (`scripts/run_scaling.py`, plot in `experiments/runs/`)_ |

The vectorized sampler replaces a per-position Python loop with batched tensor ops, the
canonical "feed data faster" win. Forward+backward on the same machine runs at ~45K items/sec.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

pytest                                   # component specs (fail until implemented)
python scripts/run_scaling.py            # Table V sweep + plot
python scripts/compare_losses.py         # loss comparison
python scripts/profile_throughput.py     # sampler + step throughput
```

On a Meta devvm, prefix pip with `--proxy http://fwdproxy:8080`; the vendored dataset
downloader uses the proxy automatically.

## Layout

```
src/sasrec_longseq/
  core/             vendored SASRec (model, data, baseline loss, eval) from sasrec-pytorch
  sampling.py       [component A] vectorized negative sampling
  losses_full.py    [component B] full-vocabulary softmax loss
  scaling.py        [component C] sequence-length scaling harness (Table V)
  profiling.py      [component D] throughput profiler
scripts/            run_scaling.py, compare_losses.py, profile_throughput.py
tests/              one spec file per component
EXERCISES.md        what each component is and how to build it
```

## License

MIT. Builds on SASRec (Kang & McAuley, ICDM 2018).
