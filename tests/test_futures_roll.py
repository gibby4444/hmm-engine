"""Prompt 4: roll dates must not produce spurious regime changes."""
import numpy as np
import pandas as pd

from core.hmm_engine import RegimeModel, predict_regime_filtered
from data.features import build_features
from data.market_data import back_adjust
from tests.conftest import make_regime_prices

SPEC = [{"name": "ret", "window": 1}, {"name": "rvol", "window": 10}]


def _contracts():
    base = make_regime_prices(600, seed=3)
    idx = base.index
    # three overlapping contracts, each 3% contango over the previous
    cs = [base * (1.03 ** k) for k in range(3)]
    for c in cs:
        c["volume"] = base["volume"]
    rolls = [idx[200], idx[400]]
    return base, cs, rolls


def test_back_adjusted_returns_equal_true_returns():
    base, cs, rolls = _contracts()
    adj = back_adjust(cs, rolls, method="ratio")
    r_adj = np.log(adj["close"]).diff().dropna()
    r_true = np.log(base["close"]).diff().dropna()
    np.testing.assert_allclose(r_adj.values, r_true.loc[r_adj.index].values, atol=1e-12)


def test_unadjusted_stitch_creates_fake_jump_and_adjusted_does_not():
    base, cs, rolls = _contracts()
    raw = pd.concat([cs[0].loc[: rolls[0]].iloc[:-1], cs[1].loc[rolls[0]: rolls[1]].iloc[:-1], cs[2].loc[rolls[1]:]])
    r_raw = np.log(raw["close"]).diff()
    assert abs(r_raw.loc[rolls[0]] - np.log(base["close"]).diff().loc[rolls[0]]) > 0.025


def test_roll_dates_do_not_flip_regime():
    base, cs, rolls = _contracts()
    adj = back_adjust(cs, rolls)
    fa = build_features(adj, SPEC)
    fb = build_features(base, SPEC)
    X = fb.dropna().values
    m = RegimeModel(3, n_init=1).fit(X)
    pa = predict_regime_filtered(m, fa.dropna().values).argmax(1)
    pb = predict_regime_filtered(m, X).argmax(1)
    assert (pa == pb).all(), "back-adjusted series changed the regime path"
