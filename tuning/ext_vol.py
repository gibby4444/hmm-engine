"""Prompt 7 — External volatility: A/B as a feature, and circuit-breaker sweep.

    python -m tuning.ext_vol --symbol SPY
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.walk_forward import evaluate, load_inputs, system_exposure, walk_forward
from core.confidence_sizing import SizingConfig
from core.gating import GateConfig
from tuning import feature_lab
from tuning import report as R


def breaker_sweep(inp, eng) -> pd.DataFrame:
    g, sz = GateConfig.from_profile(inp.prof), SizingConfig.from_profile(inp.prof)
    oos0 = eng.frame().index[0]
    ext = inp.ext.loc[oos0:]
    rows = {}
    base_exp, *_ = system_exposure(eng, inp, g, sz)
    b = evaluate(base_exp, inp, start=oos0)
    rows["no breaker"] = {"bars tripped": 0, "% of bars": 0.0, "episodes": 0,
                          **{k: b[k] for k in ["cagr", "sharpe", "max_dd", "calmar", "trades"]}}
    for lvl in inp.prof["external_vol"]["breaker_test_levels"]:
        exp = system_exposure(eng, inp, g, sz, breaker_level=lvl)[0]
        m = evaluate(exp, inp, start=oos0)
        tripped = (ext > lvl)
        episodes = int((tripped.astype(int).diff() == 1).sum())
        rows[f"breaker > {lvl}"] = {"bars tripped": int(tripped.sum()), "% of bars": float(tripped.mean()),
                                    "episodes": episodes, **{k: m[k] for k in ["cagr", "sharpe", "max_dd",
                                                                               "calmar", "trades"]}}
    return pd.DataFrame(rows).T


def run(symbol: str):
    inp = load_inputs(symbol)
    eng = walk_forward(inp)
    sweep = breaker_sweep(inp, eng)
    ab = None
    base_set = inp.prof["feature_set"]
    cand = {"equity_base": "equity_plus_vix"}.get(base_set)
    if cand:
        ab = feature_lab.ab(symbol, base_set, cand)
    return inp, sweep, ab


def render(symbol, inp, sweep, ab):
    s = inp.prof["external_vol"]["series"]
    body = R.card(f"External series: <b>{s}</b>. Point-in-time alignment: daily VIX settles after the equity "
                  "close, so the value used at bar t is the prior session's print (1-bar publication lag). "
                  "Feed failures degrade gracefully: the model runs without the feature and the live breaker "
                  "fails safe (trips to defensive) on a missing reading.")
    body += "<h2>Circuit breaker — historical frequency and impact (out-of-sample)</h2>" + R.table(
        sweep, {**R.PERF_FMT, "bars tripped": "{:.0f}", "% of bars": "{:.1%}", "episodes": "{:.0f}"})
    body += R.card("Read the sweep as a curve, not a menu. A level is robust only if its neighbours look "
                   "similar; picking the single best row is curve-fitting with one more parameter.", "warn")
    if ab:
        A, B = ab["A"], ab["B"]
        df = pd.DataFrame([A, B]).set_index("set")[["oos_return_logscore", "seed_ARI", "min_avg_visit"]]
        body += ("<h2>As a feature — states with vs without</h2>" +
                 R.table(df, {"oos_return_logscore": "{:.4f}", "seed_ARI": "{:.2f}", "min_avg_visit": "{:.1f}"}) +
                 R.card(("<b>KEEP candidate</b>" if ab["keep"] else "<b>REVERT</b>") + "<ul>" +
                        "".join(f"<li>{r}</li>" for r in ab["reasons"]) + "</ul>",
                        "verdict" if ab["keep"] else "warn"))
    elif s == "rv_ratio":
        body += R.card("This profile's external series is the short/long realised-vol ratio, which is already a "
                       "base feature (rv_ratio), so the feature A/B is not applicable. Free implied-vol history "
                       "for crypto (e.g. DVOL) isn't in the data layer; add a fetcher to test it.")
    return R.write(f"ext_vol_{symbol}.html", f"External volatility — {symbol}", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    inp, sw, ab = run(a.symbol)
    print(sw.round(3).to_string())
    print(render(a.symbol, inp, sw, ab))


if __name__ == "__main__":
    main()
