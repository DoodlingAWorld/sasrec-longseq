"""Spec for Component C: measure_scaling (uses an injected fake trainer — fast)."""

from sasrec_longseq.scaling import measure_scaling


def test_orchestration_and_passthrough():
    calls = []

    def fake_train(cfg):
        calls.append(cfg.max_len)
        return {
            "best_val_ndcg": 0.5 + 0.001 * cfg.max_len,
            "test_ndcg": 0.0,
            "test_hit": 0.0,
        }

    res = measure_scaling([10, 50, 100], epochs=2, train_fn=fake_train)

    assert [r["n"] for r in res] == [10, 50, 100]  # order preserved
    assert calls == [10, 50, 100]  # trained once per n, in order
    for r in res:
        assert set(r.keys()) >= {"n", "sec_per_epoch", "val_ndcg"}
        assert r["sec_per_epoch"] >= 0.0
    # val_ndcg is pulled from the trainer's best_val_ndcg
    assert abs(res[1]["val_ndcg"] - (0.5 + 0.001 * 50)) < 1e-9


def test_builds_config_per_n():
    cfgs = []

    def fake_train(cfg):
        cfgs.append(cfg)
        return {"best_val_ndcg": 0.5}

    measure_scaling(
        [10, 200], epochs=3, hidden_dim=16, num_blocks=1, train_fn=fake_train
    )

    assert [c.max_len for c in cfgs] == [10, 200]
    assert all(c.num_epochs == 3 for c in cfgs)
    assert all(c.hidden_dim == 16 and c.num_blocks == 1 for c in cfgs)
