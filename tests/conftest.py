import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_regime_prices(n=900, seed=7):
    """Synthetic OHLCV with genuine regime switching (calm-up / volatile-down / chop)."""
    rng = np.random.default_rng(seed)
    mus, sds = [0.0006, -0.0015, 0.0], [0.006, 0.025, 0.012]
    P = np.array([[0.98, 0.01, 0.01], [0.03, 0.95, 0.02], [0.02, 0.02, 0.96]])
    s, r = 0, []
    for _ in range(n):
        s = rng.choice(3, p=P[s])
        r.append(rng.normal(mus[s], sds[s]))
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2015-01-01", periods=n)
    hi = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    lo = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    vol = rng.integers(1e6, 2e6, n).astype(float)
    return pd.DataFrame({"open": op, "high": np.maximum(hi, np.maximum(op, close)),
                         "low": np.minimum(lo, np.minimum(op, close)), "close": close, "volume": vol}, index=idx)


@pytest.fixture(scope="session")
def px():
    return make_regime_prices()


SPEC = [{"name": "ret", "window": 5}, {"name": "rvol", "window": 21}, {"name": "vol_ratio", "window": 21}]


@pytest.fixture(scope="session")
def spec():
    return SPEC
