"""Prompt 8: backtest and live retraining paths must never diverge."""
import numpy as np

from core.retraining import LiveRunner, RetrainConfig, RetrainEngine
from data.features import build_features


def test_live_and_backtest_retraining_identical(px, spec):
    px = px.iloc[:520]
    cfg = RetrainConfig(every_bars=40, min_train_bars=250, drift_lookback=10)
    feats = build_features(px, spec)
    r = np.log(px["close"]).diff()
    vp = r.rolling(21).std().fillna(r.expanding().std()).fillna(0).values
    bt = RetrainEngine(feats.values, r.shift(-1).values, vp, px.index, 3, cfg, n_init=1).run()
    live = LiveRunner(spec, 3, cfg, n_init=1)
    for t in range(len(px)):
        live.on_new_bar(px.iloc[: t + 1])
    lv = live.engine
    assert [x["bar"] for x in lv.retrain_log] == [x["bar"] for x in bt.retrain_log]
    assert [x["reason"] for x in lv.retrain_log] == [x["reason"] for x in bt.retrain_log]
    for t in bt.hist_probs:
        np.testing.assert_allclose(lv.hist_probs[t], bt.hist_probs[t], atol=1e-9)
    # playbooks identical: proves no future return leaked into live or backtest routing
    for a, b in zip(lv.versions, bt.versions):
        assert a.playbook == b.playbook


def test_drift_tripwire_fires_on_structural_break(px, spec):
    px = px.iloc[:600].copy()
    # inject a violent new regime after bar 450
    k = px.index[450:]
    shock = np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.08, len(k))))
    for c in ["open", "high", "low", "close"]:
        px.loc[k, c] = px.loc[k, c] * shock
    cfg = RetrainConfig(every_bars=10_000, min_train_bars=300, drift_lookback=10, drift_ll_z=-3)
    feats = build_features(px, spec)
    r = np.log(px["close"]).diff()
    e = RetrainEngine(feats.values, r.shift(-1).values, r.rolling(21).std().fillna(0).values,
                      px.index, 3, cfg, n_init=1).run()
    reasons = [x["reason"] for x in e.retrain_log]
    assert reasons[0] == "initial" and any(x.startswith("drift") for x in reasons[1:])


def test_rolling_window_respects_length(px, spec):
    cfg = RetrainConfig(every_bars=50, window="rolling", window_bars=300, min_train_bars=250, drift_enabled=False)
    feats = build_features(px, spec)
    r = np.log(px["close"]).diff()
    e = RetrainEngine(feats.values, r.shift(-1).values, r.rolling(21).std().fillna(0).values,
                      px.index, 3, cfg, n_init=1).run()
    for v in e.versions:
        n = px.index.get_loc(px.index[px.index.strftime("%Y-%m-%d") == v.train_end][0]) - \
            px.index.get_loc(px.index[px.index.strftime("%Y-%m-%d") == v.train_start][0]) + 1
        assert n <= 300
