"""Cost-aware vectorised backtester. Position decided at the close of bar t earns
the return of bar t+1. Costs are charged on turnover at the time of the change."""
from __future__ import annotations

import numpy as np
import pandas as pd


def run(close: pd.Series, exposure: np.ndarray | pd.Series, costs_bps: float,
        bars_per_year: int = 252) -> dict:
    close = close.astype(float)
    pos = pd.Series(np.asarray(exposure, dtype=float), index=close.index).fillna(0.0)
    r_next = np.log(close).diff().shift(-1).fillna(0.0)
    simple_next = np.expm1(r_next)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * costs_bps / 1e4
    strat = pos * simple_next - cost
    eq = (1 + strat).cumprod()
    return {"returns": strat, "equity": eq, "position": pos, "turnover": turnover, "cost": cost,
            **metrics(strat, bars_per_year, pos, cost)}


def metrics(r: pd.Series, bpy: int, pos: pd.Series | None = None, cost: pd.Series | None = None) -> dict:
    r = r.dropna()
    if len(r) < 2:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / bpy
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(bpy)
    sharpe = r.mean() / r.std() * np.sqrt(bpy) if r.std() > 0 else 0.0
    dn = r[r < 0].std() * np.sqrt(bpy)
    dd = (eq / eq.cummax() - 1).min()
    out = {"cagr": cagr, "vol": vol, "sharpe": sharpe,
           "sortino": r.mean() * bpy / dn if dn > 0 else np.nan,
           "max_dd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan,
           "skew": float(r.skew()), "kurt": float(r.kurt() + 3), "n_bars": len(r)}
    if pos is not None:
        p = pos.loc[r.index]
        out["exposure_avg"] = float(p.abs().mean())
        out["trades"] = int((p.diff().abs() > 1e-9).sum())
    if cost is not None:
        out["cost_drag_ann"] = float(cost.loc[r.index].sum() / yrs)
    return out


def ma_filter(close: pd.Series, window: int = 200) -> np.ndarray:
    """Naive regime filter baseline: long above the N-bar MA, flat below."""
    ma = close.rolling(window).mean()
    return (close > ma).astype(float).where(ma.notna(), 0.0).values


def vol_target(close: pd.Series, target: float = 0.15, lookback: int = 20, cap: float = 1.0,
               bpy: int = 252) -> np.ndarray:
    rv = np.log(close).diff().rolling(lookback).std() * np.sqrt(bpy)
    return (target / rv).clip(upper=cap).fillna(0.0).values
