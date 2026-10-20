"""THE test that matters most (CLAUDE.md rule 5). If any of these fail, future data
is leaking into present decisions."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import load_raw
from core.hmm_engine import RegimeModel, predict_regime_filtered
from core.retraining import RetrainConfig, RetrainEngine
from data.features import build_features, lookahead_check

ROOT = Path(__file__).resolve().parents[1]


def test_every_configured_feature_set_is_point_in_time(px):
    ext = pd.Series(np.linspace(12, 40, len(px)), index=px.index)
    for name, spec in load_raw()["feature_sets"].items():
        leaks = lookahead_check(px, spec, ext=ext, probes=15)
        assert not leaks, f"feature set {name} leaks future data via {leaks}"


def test_lookahead_check_catches_a_planted_leak(px, spec, monkeypatch):
    from data import features as F
    monkeypatch.setitem(F.BUILDERS, "ret", lambda p, w, **_: np.log(p["close"]).diff().shift(-1))
    assert F.lookahead_check(px, spec, probes=10), "the leak detector failed to detect a planted leak"


def test_filtered_probs_do_not_change_when_future_is_appended(px, spec):
    X = build_features(px, spec).dropna().values
    m = RegimeModel(3, n_init=1).fit(X[:500])
    full = predict_regime_filtered(m, X)
    for t in [120, 333, 501, len(X) - 2]:
        trunc = predict_regime_filtered(m, X[: t + 1])
        np.testing.assert_allclose(trunc[-1], full[t], atol=1e-12)


def test_filtered_differs_from_smoothed(px, spec):
    """Sanity: smoothed posteriors (forward-backward) are NOT what we trade on."""
    from hmmlearn.hmm import GaussianHMM
    X = build_features(px, spec).dropna().values
    m = RegimeModel(3, n_init=1).fit(X)
    h = GaussianHMM(3, "full")
    h.startprob_, h.transmat_, h.means_, h.covars_ = m.startprob, m.transmat, m.means, m.covars
    Z = m.scaler.transform(X)
    assert np.abs(h.predict_proba(Z)[:-1] - m.filter(X)[:-1]).max() > 1e-3
    np.testing.assert_allclose(h.predict_proba(Z)[-1], m.filter(X)[-1], atol=1e-8)  # equal only at the end


def test_scaler_fitted_on_training_rows_only(px, spec):
    X = build_features(px, spec).dropna().values
    m = RegimeModel(3, n_init=1).fit(X[:400])
    np.testing.assert_allclose(m.scaler.mu, X[:400].mean(0))


def test_no_smoothed_inference_calls_in_decision_code():
    """Static guard: no hmmlearn predict()/predict_proba()/decode() in decision paths."""
    bad = []
    allowed = {"tuning/detector_bakeoff.py"}  # GMM/k-means have no temporal structure: no look-ahead
    for f in list((ROOT / "core").rglob("*.py")) + list((ROOT / "backtest").rglob("*.py")) + \
            list((ROOT / "tuning").rglob("*.py")):
        rel = str(f.relative_to(ROOT))
        if rel in allowed:
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"\.(predict|predict_proba)\(|\.decode\(\s*\w", line) and not line.strip().startswith("#"):
                bad.append(f"{rel}:{i}: {line.strip()}")
    assert not bad, "smoothed/Viterbi inference found:\n" + "\n".join(bad)


def test_walk_forward_decisions_identical_on_truncated_history(px, spec):
    """End-to-end: the engine's probability at bar t is the same whether it saw the
    whole history or only bars <= t."""
    feats = build_features(px, spec)
    r = np.log(px["close"]).diff()
    vp = r.rolling(21).std().fillna(0).values
    cfg = RetrainConfig(every_bars=60, min_train_bars=250, drift_enabled=True, drift_lookback=10)

    def run(n):
        e = RetrainEngine(feats.values[:n], r.shift(-1).values[:n], vp[:n], px.index[:n], 3, cfg, n_init=1)
        return e.run()

    full = run(len(px))
    t = 700
    part = run(t + 1)
    np.testing.assert_allclose(part.hist_probs[t], full.hist_probs[t], atol=1e-10)
    assert part.retrain_log == [x for x in full.retrain_log if x["bar"] <= t]
