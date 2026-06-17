# Components to implement

Four modules are stubbed (they raise `NotImplementedError`). Each is a real, standalone
piece, not a one-liner. The test file next to each is the spec: make it pass. A suggested
order (warm-up to headline) is D, A, B, C, but they're independent.

Run one component's tests in a tight loop while you work:
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_sampling.py -q
```

The driver scripts (`scripts/`) and the results in the README only work once the
components they use are implemented.

## D. Throughput profiler (warm-up)
* File: `src/sasrec_longseq/profiling.py`  ·  Test: `tests/test_profiling.py`

Time a `step_fn` over a list of pre-built batches and report items/second, seconds/batch,
batches/second. Guard the empty-list case so you never divide by zero.

**Concept:** throughput is the honest currency of the Code-Modeller archetype. items/sec is
just total valid items divided by wall-clock seconds; the discipline is measuring the right
thing (pre-build batches so data loading isn't counted, warm up once before timing).

## A. Vectorized negative sampling
* File: `src/sasrec_longseq/sampling.py`  ·  Test: `tests/test_sampling.py`

Replace the Phase 1 per-position Python loop with one that samples the whole batch's
negatives in a few tensor ops. For each valid position you need a negative in `[1, num_items]`
that differs from the positive at that position; pad positions stay 0.

**Concept:** the standard pattern is "draw, then re-draw only the collisions." You cannot
guarantee `neg != pos` in a single uniform draw, so you loop, but the loop runs over the
shrinking set of *colliding entries* (a boolean mask), not over every element in Python. This
is the difference between O(B*L) interpreter overhead and a handful of vectorized ops. The
`test_resamples_when_only_one_valid_negative_exists` case (num_items=2, all targets=1) forces
your collision loop to actually converge.

## B. Full-vocabulary softmax loss
* File: `src/sasrec_longseq/losses_full.py`  ·  Test: `tests/test_losses_full.py`

Score every item at every position (`feats @ item_embedding.weight.T`) and apply cross-entropy
against the positive targets, ignoring pad positions (`ignore_index=0`).

**Concept:** this is the exact next-item objective, the opposite end of the spectrum from the
Phase 1 sampled BCE (one negative). Exact, but the logits tensor is `[B, L, |I|+1]`, so cost
and memory scale with the catalog size. Implementing both lets `scripts/compare_losses.py`
quantify the quality-vs-cost tradeoff that motivates sampled losses in the first place.

## C. Sequence-length scaling harness (headline result)
* File: `src/sasrec_longseq/scaling.py`  ·  Test: `tests/test_scaling.py`

Sweep the max sequence length `n`, train the model for a fixed number of epochs at each, and
record `sec_per_epoch` and `val_ndcg`. This reproduces the paper's Table V.

**Concept:** this is experiment design. The `train_fn` parameter is dependency injection: the
test passes a fake trainer so it can verify your orchestration (one config per `n`, correct
`max_len`/`epochs`, timing, result assembly) in milliseconds, while the real run uses the
vendored core trainer. Expect `sec_per_epoch` to rise with `n` (the O(n^2) attention term) and
`val_ndcg` to improve then plateau, just like the paper.

## After all four
Run the drivers to produce the repo's results:
```bash
python scripts/profile_throughput.py     # loop vs vectorized sampler speedup
python scripts/compare_losses.py         # sampled BCE vs full softmax
python scripts/run_scaling.py            # Table V + cost-vs-quality plot
```
Then we fill the README results section with your numbers.
