"""Prompt 12 — Tuning log + deflated Sharpe. Every experiment is a trial.

    python -m tuning.tuning_log record --symbol SPY --change "persistence 3→5" \
        --hypothesis "fewer whipsaws in chop" --decision revert --reason "OOS Sharpe fell"
    python -m tuning.tuning_log report --symbol SPY

`record` runs the validation harness for you (one command), stores before/after,
and bumps the trial counter. Deflated Sharpe (Bailey & López de Prado, 2014)
discounts the current system's Sharpe for ALL trials logged.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from tuning import report as R

LOG = Path(__file__).resolve().parents[1] / "tuning" / "tuning_log.jsonl"
EULER = 0.5772156649


def deflated_sharpe(sr_per_bar: float, n_bars: int, skew: float, kurt: float, n_trials: int,
                    trial_sr_var: float) -> dict:
    """Probability the true Sharpe exceeds the max expected from n_trials of pure luck.
    Inputs are per-bar (non-annualised) Sharpe; kurt is raw (normal = 3)."""
    n_trials = max(1, n_trials)
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = np.sqrt(max(trial_sr_var, 1e-12)) * ((1 - EULER) * norm.ppf(1 - 1 / n_trials) +
                                                   EULER * norm.ppf(1 - 1 / (n_trials * np.e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr_per_bar + (kurt - 1) / 4 * sr_per_bar ** 2))
    dsr = norm.cdf((sr_per_bar - sr0) * np.sqrt(n_bars - 1) / denom)
    return {"dsr": float(dsr), "sr0_per_bar": float(sr0), "n_trials": n_trials}


def entries(symbol: str | None = None) -> list[dict]:
    if not LOG.exists():
        return []
    out = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    return [e for e in out if symbol is None or e["symbol"] == symbol]


def record(symbol: str, change: str, hypothesis: str, decision: str, reason: str,
           before: dict | None = None, after: dict | None = None, config: dict | None = None,
           returns: pd.Series | None = None) -> dict:
    """Append one experiment. before/after are metric dicts (e.g. validate()['OOS'].iloc[0])."""
    assert decision in ("keep", "revert", "baseline")
    n = len(entries(symbol)) + 1
    e = {"n": n, "date": datetime.now().isoformat(timespec="seconds"), "symbol": symbol, "change": change,
         "hypothesis": hypothesis, "before": before, "after": after, "decision": decision,
         "reason": reason, "config": config}
    if returns is not None:
        r = returns.dropna()
        e["after_stats"] = {"sr_per_bar": float(r.mean() / r.std()), "n_bars": len(r),
                            "skew": float(r.skew()), "kurt": float(r.kurt() + 3)}
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(e, default=float) + "\n")
    return e


def current_dsr(symbol: str) -> dict | None:
    es = entries(symbol)
    stats = [e for e in es if e.get("after_stats")]
    if not stats:
        return None
    srs = [e["after_stats"]["sr_per_bar"] for e in stats]
    kept = [e for e in stats if e["decision"] in ("keep", "baseline")]
    cur = (kept or stats)[-1]["after_stats"]
    # Variance of Sharpe across trials. With fewer than 3 recorded return series the
    # cross-trial variance is unknowable, so fall back to the sampling variance of the
    # Sharpe estimator itself (Mertens/Lo) — a floor, never zero. Zero would make
    # every system look significant regardless of how many trials were run.
    sr, T = cur["sr_per_bar"], cur["n_bars"]
    est_var = (1 - cur["skew"] * sr + (cur["kurt"] - 1) / 4 * sr ** 2) / (T - 1)
    var = max(float(np.var(srs, ddof=1)), est_var) if len(srs) >= 3 else est_var
    return deflated_sharpe(sr, T, cur["skew"], cur["kurt"], len(es), var)


def render(symbol: str, bpy: int = 252) -> Path:
    es = entries(symbol)
    rows = []
    for e in es:
        b, a = e.get("before") or {}, e.get("after") or {}
        rows.append({"#": e["n"], "date": e["date"][:10], "change": e["change"], "hypothesis": e["hypothesis"],
                     "Sharpe before": b.get("sharpe", np.nan), "Sharpe after": a.get("sharpe", np.nan),
                     "maxDD after": a.get("max_dd", np.nan), "decision": e["decision"].upper(),
                     "reason": e["reason"]})
    df = pd.DataFrame(rows).set_index("#") if rows else pd.DataFrame()
    d = current_dsr(symbol)
    if d:
        sr0_ann = d["sr0_per_bar"] * np.sqrt(bpy)
        kind = "verdict" if d["dsr"] >= 0.95 else ("warn" if d["dsr"] >= 0.5 else "bad")
        head = R.card(f"<b>{d['n_trials']} trials logged.</b> To be believed, the current system's Sharpe must "
                      f"beat ≈<b>{sr0_ann:.2f}</b> (annualised) — the best Sharpe you'd expect from "
                      f"{d['n_trials']} tries at pure noise. Deflated Sharpe probability: <b>{d['dsr']:.1%}</b> "
                      "(≥95% is the conventional bar). The more you tune, the higher this bar gets. "
                      "Note: DSR asks whether the Sharpe is distinguishable from luck, not whether it beats the "
                      "baselines; a system that fails the validation harness fails regardless of its DSR.", kind)
    else:
        head = R.card(f"<b>{len(es)} trials logged.</b> No return series recorded yet, so no deflated Sharpe.")
    cfg = next((e["config"] for e in reversed(es) if e["decision"] in ("keep", "baseline") and e.get("config")), None)
    body = head + "<h2>Experiment history</h2>" + (R.table(df, {"Sharpe before": "{:.2f}", "Sharpe after": "{:.2f}",
                                                                "maxDD after": "{:.1%}"}) if len(df) else "<p>empty</p>")
    if cfg:
        body += "<h2>Current configuration & evidence trail</h2>" + R.card(
            "<pre style='white-space:pre-wrap;color:#e6edf7'>" + json.dumps(cfg, indent=1, default=str) + "</pre>"
            + "Each setting above is backed by the KEEP/BASELINE row that introduced it in the history table.")
    return R.write(f"tuning_log_{symbol}.html", f"Tuning log — {symbol}", body)


def record_with_validation(symbol: str, change: str, hypothesis: str, decision: str, reason: str,
                           before_res: dict | None, after_res: dict, config: dict | None = None):
    """One-command path wired to the validation harness."""
    return record(symbol, change, hypothesis, decision, reason,
                  before=None if before_res is None else before_res["OOS"].iloc[0].to_dict(),
                  after=after_res["OOS"].iloc[0].to_dict(), config=config, returns=after_res["hmm_returns"])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--symbol", default="SPY")
    r.add_argument("--change", required=True)
    r.add_argument("--hypothesis", required=True)
    r.add_argument("--decision", choices=["keep", "revert", "baseline"], required=True)
    r.add_argument("--reason", required=True)
    p = sub.add_parser("report")
    p.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    if a.cmd == "record":
        from tuning.regime_validation import validate
        res = validate(a.symbol)   # validates the CURRENT settings.yaml
        prev = [e for e in entries(a.symbol) if e.get("after")]
        e = record(a.symbol, a.change, a.hypothesis, a.decision, a.reason,
                   before=prev[-1]["after"] if prev else None, after=res["OOS"].iloc[0].to_dict(),
                   returns=res["hmm_returns"])
        print(f"logged trial #{e['n']}: {res['verdict']}")
    print(render(a.symbol))


if __name__ == "__main__":
    main()
