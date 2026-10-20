"""Glue: profile → data → features → walk-forward engine → gated, sized exposure → backtest.

Everything downstream (validation, lag, bake-off, gating replay) reuses the cached
raw walk-forward output, so changing a gate or sizing dial never refits the HMM.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import engine as backtest
from core.config import profile_for
from core.confidence_sizing import SizingConfig, final_exposure
from data.market_data import fetch_external, fetch_prices
from data.features import build_features
from core.gating import GateConfig, apply_gates
from core.retraining import RetrainConfig, RetrainEngine
from core.vol_integration import circuit_breaker

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "wf"


@dataclass
class Inputs:
    symbol: str
    prof: dict
    px: pd.DataFrame
    ext: pd.Series
    ext_gap: pd.Series
    feats: pd.DataFrame
    X: np.ndarray
    fwd_ret: np.ndarray
    vol_proxy: np.ndarray


def load_inputs(symbol: str, overrides: dict | None = None, feature_spec: list | None = None) -> Inputs:
    prof = profile_for(symbol, overrides=overrides)
    px = fetch_prices(symbol)
    ext, gap = fetch_external(prof["external_vol"]["series"], px.index, prices=px)
    spec = feature_spec or prof["features"]
    feats = build_features(px, spec, ext=ext)
    r = np.log(px["close"]).diff()
    fwd = r.shift(-1).values
    vol_proxy = r.rolling(21).std().fillna(r.expanding().std()).fillna(0).values  # orders states only
    return Inputs(symbol, prof, px, ext, gap, feats, feats.values, fwd, vol_proxy)


def _key(symbol, spec, n_states, rcfg: RetrainConfig, seed, tag):
    blob = json.dumps([symbol, spec, n_states, rcfg.__dict__, seed, tag], sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def walk_forward(inp: Inputs, n_states: int | None = None, rcfg: RetrainConfig | None = None,
                 seed: int = 0, model_factory=None, tag: str = "hmm", use_cache: bool = True) -> RetrainEngine:
    n_states = n_states or inp.prof["n_states"]
    rcfg = rcfg or RetrainConfig.from_profile(inp.prof)
    spec = list(inp.feats.columns)
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{inp.symbol}_{tag}_{_key(inp.symbol, spec, n_states, rcfg, seed, tag)}.pkl"
    if use_cache and fp.exists():
        with open(fp, "rb") as f:
            return pickle.load(f)
    eng = RetrainEngine(inp.X, inp.fwd_ret, inp.vol_proxy, inp.px.index, n_states, rcfg,
                        bars_per_year=inp.prof["bars_per_year"], seed=seed, model_factory=model_factory)
    eng.run()
    eng.X = None  # don't pickle the matrix
    with open(fp, "wb") as f:
        pickle.dump(eng, f)
    eng.X = inp.X
    return eng


def system_exposure(eng: RetrainEngine, inp: Inputs, gate: GateConfig | None, sizing: SizingConfig | None,
                    breaker_level: float | None = None, neutral_exposure: float = 0.0):
    """Turn raw walk-forward probabilities into the final per-bar exposure."""
    fr = eng.frame()
    P = fr[[c for c in fr.columns if c.startswith("prob_")]].values
    g = apply_gates(P, gate or GateConfig(0.0, 1), index=fr.index)
    base = eng.exposure_for(g.regime, fr["version"].values, neutral_exposure)
    sz = sizing or SizingConfig(mode="hard")
    exp = final_exposure(base, g.confidence, sz, inp.prof["risk"])
    blog = pd.DataFrame()
    if breaker_level is not None:
        exp, blog = circuit_breaker(exp, inp.ext.reindex(fr.index).values, breaker_level,
                                    on_gap="hold", index=fr.index)  # historical gaps = pre-series start
    full = pd.Series(0.0, index=inp.px.index)
    full.loc[fr.index] = exp
    return full, g, blog, fr


def evaluate(exposure: pd.Series, inp: Inputs, start=None, end=None) -> dict:
    close = inp.px["close"]
    if start is not None or end is not None:
        sl = slice(start, end)
        close, exposure = close.loc[sl], exposure.loc[sl]
    return backtest.run(close, exposure, inp.prof["costs_bps"], inp.prof["bars_per_year"])
