"""External volatility circuit breaker (Prompt 7).

Lives OUTSIDE the model: when the indicator exceeds `level`, the defensive regime
is forced regardless of the HMM's read. Fails safe: a missing reading (feed gap)
trips the breaker unless on_gap="hold" is chosen explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def circuit_breaker(exposure: np.ndarray, ext: pd.Series | np.ndarray, level: float,
                    defensive_exposure: float = 0.0, on_gap: str = "defensive",
                    index=None) -> tuple[np.ndarray, pd.DataFrame]:
    e = np.asarray(exposure, dtype=float).copy()
    v = np.asarray(ext, dtype=float)
    gap = ~np.isfinite(v)
    trip = (v > level) & ~gap
    if on_gap == "defensive":
        trip |= gap
    e[trip] = np.minimum(e[trip], defensive_exposure)
    idx = index if index is not None else np.arange(len(e))
    log = pd.DataFrame({"time": np.asarray(idx)[trip], "value": v[trip],
                        "reason": np.where(gap[trip], "feed_gap", f"above_{level}")})
    return e, log
