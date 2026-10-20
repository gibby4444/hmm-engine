import numpy as np
import pandas as pd

from core.vol_integration import circuit_breaker
from data.market_data import fetch_external


def test_breaker_forces_defensive_above_level():
    e, log = circuit_breaker(np.ones(5), np.array([15, 31, 29, 45, 20.0]), 30)
    np.testing.assert_allclose(e, [1, 0, 1, 0, 1])
    assert len(log) == 2


def test_breaker_fails_safe_on_feed_gap():
    e, log = circuit_breaker(np.ones(3), np.array([15, np.nan, 20.0]), 30)
    assert e[1] == 0 and log["reason"].iloc[0] == "feed_gap"
    e2, _ = circuit_breaker(np.ones(3), np.array([15, np.nan, 20.0]), 30, on_gap="hold")
    assert e2[1] == 1


def test_external_feed_failure_degrades_gracefully(monkeypatch):
    import data.market_data as md
    monkeypatch.setattr(md, "fetch_prices", lambda *a, **k: (_ for _ in ()).throw(ConnectionError("down")))
    idx = pd.bdate_range("2024-01-01", periods=5)
    v, gap = md.fetch_external("VIX", idx)
    assert v.isna().all() and gap.all()


def test_external_series_lagged_one_bar(monkeypatch):
    import data.market_data as md
    idx = pd.bdate_range("2024-01-01", periods=4)
    monkeypatch.setattr(md, "fetch_prices", lambda *a, **k: pd.DataFrame({"close": [10., 20, 30, 40]}, index=idx))
    v, _ = md.fetch_external("VIX", idx)
    assert np.isnan(v.iloc[0]) and list(v.iloc[1:]) == [10, 20, 30]
