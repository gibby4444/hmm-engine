"""Prompt 3 — Feature experiment framework. Feature sets live in config/settings.yaml.

    python -m tuning.feature_lab --symbol SPY --a equity_base --b equity_long

Comparability note (important): the HMM's own log-likelihood is NOT comparable
across feature sets — different dimensions and units give different densities.
The headline out-of-sample metric here is therefore the predictive log score of
the NEXT-BAR RETURN under each model's regime mixture, which is in the same units
for every feature set. The feature-space log-lik is shown for reference only.
"""
from __future__ import annotations

import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score

from backtest.walk_forward import load_inputs
from core.config import feature_set, settings
from core.hmm_engine import RegimeModel, predict_regime_filtered
from data.features import correlated_pairs, lookahead_check
from tuning import report as R
from tuning.state_report import visits


def evaluate_set(symbol: str, set_name: str, n_states: int | None = None) -> dict:
    t = settings()["tuning"]
    spec = feature_set(set_name)
    inp = load_inputs(symbol, feature_spec=spec)
    n_states = n_states or inp.prof["n_states"]
    valid = np.isfinite(inp.X).all(1)
    X, vp, fwd = inp.X[valid], inp.vol_proxy[valid], inp.fwd_ret[valid]
    split = int(len(X) * (1 - t["holdout_frac"]))
    Xtr, Xte = X[:split], X[split:]
    fits = [RegimeModel(n_states, seed=s).fit(Xtr, vol_proxy=vp[:split]) for s in range(t["stability_seeds"])]
    m = fits[int(np.argmax([f.train_loglik_per_bar for f in fits]))]
    labels = [predict_regime_filtered(f, Xtr).argmax(1) for f in fits]
    ari = float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]))

    Ptr = predict_regime_filtered(m, Xtr)
    st = Ptr.argmax(1)
    mu = np.array([np.nanmean(fwd[:split][st == k]) if (st == k).any() else 0 for k in range(n_states)])
    sd = np.array([np.nanstd(fwd[:split][st == k]) if (st == k).sum() > 1 else np.nanstd(fwd[:split])
                   for k in range(n_states)]) + 1e-6
    # out-of-sample: filter continues from training (warm), parameters frozen
    Pall = predict_regime_filtered(m, X)
    Pte = Pall[split:]
    y = fwd[split:]
    ok = np.isfinite(y)
    dens = (Pte[ok] * norm.pdf(y[ok][:, None], mu, sd)).sum(1)
    ret_logscore = float(np.mean(np.log(dens + 1e-300)))
    # baseline: single Gaussian fitted on training returns
    base = float(np.mean(norm.logpdf(y[ok], np.nanmean(fwd[:split]), np.nanstd(fwd[:split]))))
    # separation: dispersion of state vol relative to pooled
    v = visits(Pall.argmax(1))
    durs = [np.mean(v.get(k, [0])) for k in range(n_states)]
    leaks = lookahead_check(inp.px, spec, inp.ext)
    return {"set": set_name, "features": list(inp.feats.columns),
            "oos_return_logscore": ret_logscore, "oos_logscore_vs_iid": ret_logscore - base,
            "oos_feature_loglik": m.score_oos(Xte, Xtr), "seed_ARI": ari,
            "vol_separation": float(np.std(np.log(sd)) ), "min_avg_visit": float(np.min(durs)),
            "mean_avg_visit": float(np.mean(durs)), "lookahead_leaks": leaks,
            "corr_pairs": correlated_pairs(inp.feats, t["corr_warn"]), "holdout_start": str(inp.px.index[valid][split].date())}


def ab(symbol: str, a: str, b: str) -> dict:
    A, B = evaluate_set(symbol, a), evaluate_set(symbol, b)
    d_score = B["oos_return_logscore"] - A["oos_return_logscore"]
    reasons = []
    keep = True
    if B["lookahead_leaks"]:
        keep, reasons = False, [f"LOOK-AHEAD in {B['lookahead_leaks']} — invalid"]
    if d_score <= 0:
        keep = False
        reasons.append(f"OOS return log score did not improve ({d_score:+.4f}/bar)")
    else:
        reasons.append(f"OOS return log score improved {d_score:+.4f}/bar")
    if B["seed_ARI"] < A["seed_ARI"] - 0.10:
        keep = False
        reasons.append(f"stability dropped (ARI {A['seed_ARI']:.2f}→{B['seed_ARI']:.2f})")
    if B["min_avg_visit"] < 3:
        keep = False
        reasons.append(f"a state averages only {B['min_avg_visit']:.1f}-bar visits (flicker)")
    return {"A": A, "B": B, "keep": keep, "reasons": reasons, "delta": d_score}


def render(symbol: str, results: list[tuple[str, dict]]) -> str:
    body = R.card("Scaling is fitted on training rows only; every feature passed through the empirical "
                  "truncation test (features computed on data cut at bar t must equal the full-history value "
                  "at t). Headline metric: next-bar return log score on the held-out period — comparable "
                  "across feature sets. Any KEEP here is provisional until the walk-forward harness agrees.")
    for title, res in results:
        A, B = res["A"], res["B"]
        df = pd.DataFrame([A, B]).set_index("set")[
            ["oos_return_logscore", "oos_logscore_vs_iid", "seed_ARI", "vol_separation", "min_avg_visit",
             "mean_avg_visit", "oos_feature_loglik"]]
        verdict = "KEEP candidate" if res["keep"] else "REVERT (stay on current set)"
        warn = ""
        for X in (A, B):
            if X["corr_pairs"]:
                warn += f"<li>{X['set']}: correlated pairs " + ", ".join(
                    f"{p} ~ {q} ({c:+.2f})" for p, q, c in X["corr_pairs"]) + " — likely redundancy</li>"
            if X["lookahead_leaks"]:
                warn += f"<li><b>{X['set']}: LOOK-AHEAD detected in {X['lookahead_leaks']}</b></li>"
        body += (f"<h2>{title}</h2><div class='sub'>{A['set']}: {A['features']}<br>{B['set']}: {B['features']}"
                 f"<br>holdout from {A['holdout_start']}</div>" +
                 R.table(df, {"oos_return_logscore": "{:.4f}", "oos_logscore_vs_iid": "{:+.4f}",
                              "seed_ARI": "{:.2f}", "vol_separation": "{:.2f}", "min_avg_visit": "{:.1f}",
                              "mean_avg_visit": "{:.1f}", "oos_feature_loglik": "{:.3f}"}) +
                 R.card(f"<b>{verdict}</b><ul>" + "".join(f"<li>{r}</li>" for r in res["reasons"]) + "</ul>"
                        + (f"<b>Hygiene warnings</b><ul>{warn}</ul>" if warn else "Hygiene: clean."),
                        "verdict" if res["keep"] else "warn"))
    return R.write(f"feature_lab_{symbol}.html", f"Feature lab — {symbol}", body,
                   "oos_feature_loglik is in each set's own feature space: reference only, not comparable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    x = ap.parse_args()
    res = ab(x.symbol, x.a, x.b)
    print(res["keep"], res["reasons"])
    print(render(x.symbol, [(f"{x.a} vs {x.b}", res)]))


if __name__ == "__main__":
    main()
