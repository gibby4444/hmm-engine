"""Data layer: cached price/external-series fetching with point-in-time discipline,
plus back-adjusted continuous futures construction (Prompts 4 & 7)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CACHE = Path(__file__).resolve().parents[1] / ".cache"


def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def fetch_prices(symbol: str, start: str = "2005-01-01", refresh: bool = False) -> pd.DataFrame:
    """OHLCV, split/dividend adjusted. Cached to parquet/csv."""
    CACHE.mkdir(exist_ok=True)
    fp = CACHE / f"{symbol.replace('^', '_')}.csv"
    if fp.exists() and not refresh:
        return pd.read_csv(fp, index_col=0, parse_dates=True)
    import yfinance as yf

    df = _flatten(yf.download(symbol, start=start, progress=False, auto_adjust=True))
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df.to_csv(fp)
    return df


def fetch_external(series: str, index: pd.DatetimeIndex, prices: pd.DataFrame | None = None,
                   refresh: bool = False) -> tuple[pd.Series, pd.Series]:
    """External vol series aligned point-in-time to `index`.

    Returns (values, gap_flag). Value at bar t is the latest observation whose
    publication time is <= the close of bar t. For daily VIX the close prints at
    16:15 ET, after the 16:00 equity close, so a strict implementation lags it by
    one bar: the value usable for a decision at SPY's close on day t is VIX(t-1).
    That is what we do — conservative, never optimistic.

    Graceful degradation: on fetch failure returns all-NaN with gap_flag=True; the
    model then runs without the feature and the circuit breaker fails SAFE
    (see core/vol_integration.py).
    """
    try:
        if series == "rv_ratio":
            if prices is None:
                raise ValueError("rv_ratio needs prices")
            r = np.log(prices["close"]).diff()
            val = (r.rolling(7).std() / r.rolling(90).std()).reindex(index)
            # computed from closes up to and including t → available at t
            return val, val.isna()
        sym = {"VIX": "^VIX", "MOVE": "^MOVE", "VVIX": "^VVIX"}.get(series, series)
        raw = fetch_prices(sym, refresh=refresh)["close"]
        aligned = raw.shift(1).reindex(index, method="ffill")  # 1-bar publication lag
        return aligned, aligned.isna()
    except Exception as e:  # noqa: BLE001
        log.warning("External series %s unavailable (%s); running without it", series, e)
        empty = pd.Series(np.nan, index=index)
        return empty, pd.Series(True, index=index)


# ── Futures: back-adjusted continuous contracts ─────────────────────────────

def back_adjust(contracts: list[pd.DataFrame], roll_dates: list[pd.Timestamp],
                method: str = "ratio") -> pd.DataFrame:
    """Stitch contracts into a back-adjusted continuous series.

    contracts[i] is used until roll_dates[i] (exclusive), then contracts[i+1].
    At each roll the *older* history is shifted (difference) or scaled (ratio) so
    the price on the roll date is continuous. Feature computation must use this
    series; unadjusted stitching injects a fake return equal to the calendar
    spread at every roll, which an HMM happily reads as a regime change.
    Ratio adjustment keeps returns exact (preferred for log-return features).
    """
    assert len(roll_dates) == len(contracts) - 1
    cols = ["open", "high", "low", "close"]
    pieces = []
    for i, c in enumerate(contracts):
        lo = roll_dates[i - 1] if i > 0 else c.index.min()
        hi = roll_dates[i] if i < len(roll_dates) else c.index.max() + pd.Timedelta(days=1)
        pieces.append(c.loc[(c.index >= lo) & (c.index < hi)].copy())
    # walk backwards accumulating adjustment
    for i in range(len(pieces) - 2, -1, -1):
        rd = roll_dates[i]
        new_px = contracts[i + 1].loc[rd, "close"]
        old_px = contracts[i].loc[rd, "close"]
        for j in range(i + 1):
            if method == "ratio":
                pieces[j][cols] = pieces[j][cols] * (new_px / old_px)
            else:
                pieces[j][cols] = pieces[j][cols] + (new_px - old_px)
    out = pd.concat(pieces).sort_index()
    out["roll"] = out.index.isin(roll_dates)
    return out
