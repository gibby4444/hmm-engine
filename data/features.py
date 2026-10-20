"""Point-in-time feature construction (Prompt 3 hygiene).

Every feature at row t uses only information available at the close of bar t:
trailing windows only, no centered rolling, no bfill, no full-sample statistics.
Normalisation is NOT done here; it is fitted on training data only inside the
HMM engine (see core/hmm_engine.Scaler), which is where look-ahead usually sneaks in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BUILDERS = {}


def feature(name):
    def deco(fn):
        BUILDERS[name] = fn
        return fn
    return deco


def _logret(px: pd.DataFrame) -> pd.Series:
    return np.log(px["close"]).diff()


@feature("ret")
def f_ret(px, w, **_):
    """Trailing w-bar mean log return."""
    return _logret(px).rolling(w).mean()


@feature("rvol")
def f_rvol(px, w, **_):
    """Trailing w-bar realized vol (log of, to tame fat tails for a Gaussian HMM)."""
    return np.log(_logret(px).rolling(w).std() + 1e-8)


@feature("vol_ratio")
def f_vol_ratio(px, w, **_):
    """Short/long realized vol ratio: vol expanding vs contracting."""
    r = _logret(px)
    return np.log((r.rolling(max(2, w // 4)).std() + 1e-8) / (r.rolling(w * 3).std() + 1e-8))


@feature("rv_ratio")
def f_rv_ratio(px, w, **_):
    r = _logret(px)
    return np.log((r.rolling(w).std() + 1e-8) / (r.rolling(w * 8).std() + 1e-8))


@feature("atr")
def f_atr(px, w, **_):
    pc = px["close"].shift(1)
    tr = pd.concat([px["high"] - px["low"], (px["high"] - pc).abs(), (px["low"] - pc).abs()], axis=1).max(axis=1)
    return np.log(tr.rolling(w).mean() / px["close"] + 1e-8)


@feature("volume_ratio")
def f_volume_ratio(px, w, **_):
    v = px["volume"].replace(0, np.nan)
    return np.log(v.rolling(max(2, w // 4)).mean() / v.rolling(w).mean())


@feature("drawdown")
def f_drawdown(px, w, **_):
    """Distance from trailing w-bar high."""
    return np.log(px["close"] / px["close"].rolling(w).max())


@feature("ext_vol")
def f_ext_vol(px, w, ext=None, **_):
    if ext is None:
        raise ValueError("ext_vol feature requires an aligned external series")
    return np.log(ext.astype(float))


@feature("gap")
def f_gap(px, w, **_):
    """Session gap: open vs prior close (futures session feature)."""
    return np.log(px["open"] / px["close"].shift(1))


@feature("range_ratio")
def f_range_ratio(px, w, **_):
    rng = np.log(px["high"] / px["low"])
    return np.log((rng + 1e-8) / (rng.rolling(w).mean() + 1e-8))


def build_features(px: pd.DataFrame, spec: list[dict], ext: pd.Series | None = None) -> pd.DataFrame:
    cols = {}
    for f in spec:
        name, w = f["name"], int(f.get("window", 1))
        cols[f"{name}_{w}"] = BUILDERS[name](px, w, ext=ext)
    df = pd.DataFrame(cols, index=px.index)
    return df.replace([np.inf, -np.inf], np.nan)


def correlated_pairs(feats: pd.DataFrame, limit: float = 0.85) -> list[tuple[str, str, float]]:
    c = feats.dropna().corr()
    out = []
    for i, a in enumerate(c.columns):
        for b in c.columns[i + 1:]:
            if abs(c.loc[a, b]) >= limit:
                out.append((a, b, float(c.loc[a, b])))
    return out


def lookahead_check(px: pd.DataFrame, spec: list[dict], ext: pd.Series | None = None,
                    probes: int = 25, seed: int = 0) -> list[str]:
    """Empirical look-ahead test: computing features on data truncated at bar t
    must give exactly the same row t as computing them on the full history.
    Any feature that changes when the future is removed is peeking."""
    full = build_features(px, spec, ext)
    rng = np.random.default_rng(seed)
    valid = np.where(full.notna().all(axis=1))[0]
    bad = set()
    for t in rng.choice(valid[valid < len(px) - 1], size=min(probes, len(valid) - 1), replace=False):
        trunc = build_features(px.iloc[: t + 1], spec, None if ext is None else ext.iloc[: t + 1])
        a, b = trunc.iloc[-1].values.astype(float), full.iloc[t].values.astype(float)
        same = np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)  # NaN-vs-value counts as a leak
        bad |= set(trunc.columns[~same])
    return sorted(bad)
