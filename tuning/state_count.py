"""Prompt 2 — State-count comparison harness. Presents evidence; never auto-selects.

    python -m tuning.state_count --symbol SPY
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from backtest.walk_forward import load_inputs
from core.config import settings
from core.hmm_engine import RegimeModel, predict_regime_filtered
from tuning import report as R
from tuning.state_report import inspect


def compare(symbol: str, lo: int | None = None, hi: int | None = None, seeds: int | None = None):
    t = settings()["tuning"]
    lo, hi = lo or t["state_count_range"][0], hi or t["state_count_range"][1]
    seeds = seeds or t["stability_seeds"]
    inp = load_inputs(symbol)
    valid = np.isfinite(inp.X).all(1)
    X, vp = inp.X[valid], inp.vol_proxy[valid]
    rows, details = [], {}
    for k in range(lo, hi + 1):
        fits = [RegimeModel(k, seed=s).fit(X, vol_proxy=vp) for s in range(seeds)]
        lls = [f.train_loglik_per_bar for f in fits]
        best = fits[int(np.argmax(lls))]
        bic, aic = best.bic_aic(X)
        labels = [predict_regime_filtered(f, X).argmax(1) for f in fits]
        ari = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
        ins = inspect(best, inp)
        tb = ins["table"]
        details[k] = ins
        rows.append({"states": k, "BIC": bic, "AIC": aic, "loglik/bar": best.train_loglik_per_bar,
                     "seed ARI mean": np.mean(ari), "seed ARI min": np.min(ari),
                     "min state share": tb["share"].min(), "min avg visit": tb["avg_visit_bars"].min(),
                     "redundant pairs": len(ins["redundant"]),
                     "names": " | ".join(tb["name"])})
    df = pd.DataFrame(rows).set_index("states")
    df["flags"] = [", ".join(f for f, c in [("redundant", r["redundant pairs"] > 0),
                                          ("unstable", r["seed ARI min"] < 0.7),
                                          ("tiny state", r["min state share"] < 0.05),
                                          ("flickery", r["min avg visit"] < 3)] if c) or "—"
                   for _, r in df.iterrows()]
    return df, details, inp


def render(symbol, df, details):
    bic_best, aic_best = df["BIC"].idxmin(), df["AIC"].idxmin()
    marked = df.copy()
    marked.index = [f"{k}{' ★BIC' if k == bic_best else ''}{' ★AIC' if k == aic_best else ''}" for k in df.index]
    body = R.card(f"BIC favours <b>{bic_best}</b> states, AIC favours <b>{aic_best}</b>. "
                  "These shortlist candidates; they don't decide. Information criteria on financial returns "
                  "keep rewarding extra states that carve up fat tails rather than represent distinct regimes.",
                  "verdict")
    body += "<h2>Comparison</h2>" + R.table(marked, {"BIC": "{:,.0f}", "AIC": "{:,.0f}", "loglik/bar": "{:.3f}",
                                                    "seed ARI mean": "{:.2f}", "seed ARI min": "{:.2f}",
                                                    "min state share": "{:.1%}", "min avg visit": "{:.1f}",
                                                    "redundant pairs": "{:.0f}"})
    body += "<h2>Per-count state summaries</h2>"
    for k, ins in details.items():
        tb = ins["table"][["name", "share", "ret_mean_ann", "realized_vol", "avg_visit_bars", "visits"]]
        body += f"<h3>{k} states</h3>" + R.table(tb, {"share": "{:.0%}", "ret_mean_ann": "{:+.1%}",
                                                   "realized_vol": "{:.1%}", "avg_visit_bars": "{:.1f}",
                                                   "visits": "{:.0f}"})
    body += "<h2>Decision checklist (yours, not the harness's)</h2>" + R.card("""<ul>
<li>Can every state be given a distinct, honest name from its statistics?</li>
<li>Does every state map to a <i>different</i> playbook? Two states that trade the same way are one state.</li>
<li>Is the count stable across seeds (ARI min ≥ 0.7)? Unstable states are fitting noise.</li>
<li>Are visits long enough to act on after gating (avg visit well above your persistence setting)?</li>
<li>Does the count survive the walk-forward validation harness (Prompt 9)? Only that settles it.</li></ul>""")
    return R.write(f"state_count_{symbol}.html", f"State-count selection — {symbol}", body,
                   "ARI = adjusted Rand index of filtered assignments between random-seed refits (1 = identical)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    df, det, _ = compare(a.symbol)
    print(df.to_string())
    print(render(a.symbol, df, det))


if __name__ == "__main__":
    main()
