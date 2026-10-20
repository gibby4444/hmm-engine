"""Prompt 9 — Regime validation harness. The gate for every tuning change.

    python -m tuning.regime_validation --symbol SPY

Three systems on identical data, costs and risk rules:
  (a) strategies + tuned HMM regime switch (walk-forward, scheduled + drift refits)
  (b) same strategies + naive N-day moving-average filter
  (c) best single strategy, no regime layer (buy-and-hold or vol-target,
      chosen on the FULL-period Sharpe — a choice made with hindsight, which
      favours the baseline, i.e. makes the HMM's job harder, not easier)
The HMM earns its place only if it beats BOTH baselines out-of-sample.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest import engine as bt
from backtest.walk_forward import Inputs, load_inputs, system_exposure, walk_forward
from core.confidence_sizing import SizingConfig
from core.config import settings
from core.gating import GateConfig, apply_gates
from core.hmm_engine import RegimeModel, predict_regime_filtered, state_playbook
from core.risk_manager import cap_exposure
from tuning import report as R

KEYS = ["cagr", "vol", "sharpe", "sortino", "max_dd", "calmar", "exposure_avg", "trades", "cost_drag_ann"]


def in_sample_hmm(inp: Inputs, gate: GateConfig, n_states: int) -> tuple[pd.Series, int]:
    """Fit once on ALL data and trade it: the optimistic, overfit number."""
    valid = np.isfinite(inp.X).all(1)
    m = RegimeModel(n_states).fit(inp.X[valid], vol_proxy=inp.vol_proxy[valid])
    P = predict_regime_filtered(m, inp.X[valid])
    pb = state_playbook(P.argmax(1), inp.fwd_ret[valid], n_states, inp.prof["bars_per_year"])
    g = apply_gates(P, gate)
    exp = np.array([0.0 if s < 0 else pb[int(s)]["exposure"] for s in g.regime])
    out = pd.Series(0.0, index=inp.px.index)
    out.loc[inp.px.index[valid]] = cap_exposure(exp, inp.prof["risk"])
    return out, g.confirmed_switches


def validate(symbol: str | None = None, inp: Inputs | None = None, eng=None, gate: GateConfig | None = None,
             sizing: SizingConfig | None = None, breaker: float | None = None, label: str = "HMM regime") -> dict:
    t = settings()["tuning"]
    inp = inp or load_inputs(symbol)
    symbol = inp.symbol
    eng = eng or walk_forward(inp)
    gate = gate or GateConfig.from_profile(inp.prof)
    sizing = sizing or SizingConfig.from_profile(inp.prof)
    bpy, cost = inp.prof["bars_per_year"], inp.prof["costs_bps"]
    close = inp.px["close"]

    hmm_exp, g, blog, fr = system_exposure(eng, inp, gate, sizing, breaker)
    oos0 = fr.index[0]
    ma = pd.Series(bt.ma_filter(close, t["ma_window"]), index=close.index)
    bh = pd.Series(1.0, index=close.index)
    vt = pd.Series(bt.vol_target(close, t["vol_target"], t["vol_target_lookback"], 1.0, bpy), index=close.index)
    is_hmm, is_sw = in_sample_hmm(inp, gate, eng.n_states)

    def perf(exp, start=None):
        c, e = (close.loc[start:], exp.loc[start:]) if start is not None else (close, exp)
        r = bt.run(c, e, cost, bpy)
        return {k: r.get(k) for k in KEYS}, r

    single = {"buy & hold": bh, f"vol target {t['vol_target']:.0%}": vt}
    first_valid = inp.px.index[np.isfinite(inp.X).all(1)][0]
    best_single = max(single, key=lambda k: perf(single[k], first_valid)[0]["sharpe"])

    systems = {label: (is_hmm, hmm_exp, g.confirmed_switches, is_sw),
               f"MA{t['ma_window']} filter": (ma, ma, int((ma.loc[oos0:].diff().abs() > 0).sum()), None),
               f"single: {best_single}": (single[best_single], single[best_single], 0, None)}
    for k, v in single.items():
        if k != best_single:
            systems[f"(ref) {k}"] = (v, v, 0, None)
    rows_is, rows_oos, curves = {}, {}, {}
    for name, (e_is, e_oos, sw, sw_is) in systems.items():
        rows_is[name] = {**perf(e_is, first_valid)[0], "switches": sw_is if sw_is is not None else np.nan}
        p, r = perf(e_oos, oos0)
        rows_oos[name] = {**p, "switches": sw}
        curves[name] = r["equity"]
    IS, OOS = pd.DataFrame(rows_is).T, pd.DataFrame(rows_oos).T
    gap = (IS["sharpe"] - OOS["sharpe"]).rename("IS−OOS sharpe")
    names = list(systems)
    h, ma_n, s_n = names[0], names[1], names[2]
    beats_ma = OOS.loc[h, "sharpe"] > OOS.loc[ma_n, "sharpe"]
    beats_single = OOS.loc[h, "sharpe"] > OOS.loc[s_n, "sharpe"]
    if beats_ma and beats_single:
        verdict = (f"PASS — {h} beats both baselines out-of-sample on Sharpe "
                   f"({OOS.loc[h, 'sharpe']:.2f} vs {OOS.loc[ma_n, 'sharpe']:.2f} / {OOS.loc[s_n, 'sharpe']:.2f}). "
                   "Keep, then discount for trial count (tuning log).")
        kind = "verdict"
    elif not beats_ma:
        verdict = (f"FAIL — the {ma_n} matches or beats the HMM out-of-sample "
                   f"({OOS.loc[ma_n, 'sharpe']:.2f} vs {OOS.loc[h, 'sharpe']:.2f} Sharpe). "
                   "Recommendation: use the simpler moving-average filter.")
        kind = "bad"
    else:
        verdict = (f"FAIL — the HMM beats the MA filter but not the best single strategy "
                   f"({s_n}: {OOS.loc[s_n, 'sharpe']:.2f} vs {OOS.loc[h, 'sharpe']:.2f}). "
                   "The regime layer is not earning its complexity; run the single strategy.")
        kind = "bad"
    return {"symbol": symbol, "IS": IS, "OOS": OOS, "gap": gap, "verdict": verdict, "pass": kind == "verdict",
            "kind": kind, "oos_start": oos0, "curves": curves, "gate": g, "breaker_log": blog,
            "hmm_oos_sharpe": float(OOS.loc[h, "sharpe"]), "hmm_returns": perf(hmm_exp, oos0)[1]["returns"],
            "n_refits": len(eng.versions), "refit_reasons": pd.Series([r["reason"].split("(")[0]
                                                                        for r in eng.retrain_log]).value_counts()}


def render(res: dict) -> str:
    fig, ax = R.styled_ax((11, 4))
    for i, (k, eq) in enumerate(res["curves"].items()):
        ax.plot(eq.index, eq.values, lw=1.4 if i == 0 else 1.0, label=k,
                color=R.STATE_COLORS[i % 6], alpha=1 if i < 3 else 0.5)
    ax.set_yscale("log")
    ax.legend(facecolor=R.PANEL, labelcolor=R.TEXT, edgecolor="#22325f", fontsize=9)
    ax.set_title(f"Out-of-sample equity from {res['oos_start'].date()} (log)")
    fmt = R.PERF_FMT
    oos = res["OOS"].join(res["gap"])
    body = (R.card(f"<b>{res['verdict']}</b>", res["kind"]) +
            R.card(f"Walk-forward from {res['oos_start'].date()}; {res['n_refits']} model fits "
                   f"({', '.join(f'{v} {k}' for k, v in res['refit_reasons'].items())}). "
                   "Every system pays the same costs and runs under the same risk caps.") +
            "<h2>Out-of-sample</h2>" + R.table(oos, {**fmt, "IS−OOS sharpe": "{:+.2f}"}, best_row=oos.index[0]) +
            "<h2>In-sample (full-period fit — optimistic by construction)</h2>" + R.table(res["IS"], fmt) +
            R.fig_to_img(fig))
    return R.write(f"regime_validation_{res['symbol']}.html", f"Regime validation — {res['symbol']}", body,
                   "Rule: complexity must beat the simpler alternative out-of-sample or it doesn't ship")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    res = validate(a.symbol)
    print(res["OOS"].round(3).to_string())
    print(res["verdict"])
    print(render(res))


if __name__ == "__main__":
    main()
