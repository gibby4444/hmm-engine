"""Export walk-forward results + experiment tables to JSON for the web dashboard.

    python tools/export_dashboard.py            # writes web/data.json

The dashboard re-runs gating, sizing, breaker and the backtest in the browser from
the raw filtered probabilities exported here. Anything that needs a refit
(features, state count, retraining) is exported as precomputed tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest import engine as bt  # noqa: E402
from backtest.walk_forward import evaluate, load_inputs, system_exposure, walk_forward  # noqa: E402
from core.confidence_sizing import SizingConfig  # noqa: E402
from core.config import settings  # noqa: E402
from core.gating import GateConfig  # noqa: E402
from tuning import tuning_log  # noqa: E402
from tuning.state_report import inspect  # noqa: E402


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def table(df: pd.DataFrame) -> dict:
    return {"index": [str(i) for i in df.index], "columns": list(map(str, df.columns)),
            "rows": [[clean(v) for v in r] for r in df.itertuples(index=False)]}


def asset(symbol: str) -> dict:
    t = settings()["tuning"]
    inp = load_inputs(symbol)
    eng = walk_forward(inp)
    fr = eng.frame()
    P = fr[[c for c in fr.columns if c.startswith("prob_")]].values
    close = inp.px["close"]
    i0 = close.index.get_loc(fr.index[0])
    first_valid = int(np.argmax(np.isfinite(inp.X).all(1)))
    bpy = inp.prof["bars_per_year"]
    # benchmark single strategy: chosen exactly as the validation harness does
    vt = pd.Series(bt.vol_target(close, t["vol_target"], t["vol_target_lookback"], 1.0, bpy), index=close.index)
    bh = pd.Series(1.0, index=close.index)
    fv = close.index[first_valid]
    s_bh = bt.run(close.loc[fv:], bh.loc[fv:], inp.prof["costs_bps"], bpy)["sharpe"]
    s_vt = bt.run(close.loc[fv:], vt.loc[fv:], inp.prof["costs_bps"], bpy)["sharpe"]
    single = "vol_target" if s_vt > s_bh else "buy_hold"

    # reference metrics so the browser engine can be checked against Python
    refs = []
    for g, mode, brk in [(GateConfig.from_profile(inp.prof), "hard", None), (GateConfig(0.0, 1), "hard", None),
                         (GateConfig(0.6, 5), "both", None),
                         (GateConfig.from_profile(inp.prof), "hard",
                          inp.prof["external_vol"]["circuit_breaker_level"])]:
        exp, gr, _, _ = system_exposure(eng, inp, g, SizingConfig.from_profile(inp.prof, mode=mode), brk)
        m = evaluate(exp, inp, start=fr.index[0])
        refs.append({"threshold": g.confidence_threshold, "persistence": g.persistence_bars, "mode": mode,
                     "breaker": brk, "sharpe": m["sharpe"], "cagr": m["cagr"], "max_dd": m["max_dd"],
                     "switches": gr.confirmed_switches})

    ins = inspect(eng.model, inp)
    st = ins["table"]
    sizing = SizingConfig.from_profile(inp.prof)
    gate = GateConfig.from_profile(inp.prof)
    reasons = pd.Series([r["reason"].split("(")[0] for r in eng.retrain_log]).value_counts().to_dict()

    def report_tbl(name):
        try:
            return pd.read_html(ROOT / "reports" / name)
        except Exception:  # noqa: BLE001
            return []

    return clean({
        "symbol": symbol, "profile": inp.prof["profile_name"], "bpy": bpy,
        "costs_bps": inp.prof["costs_bps"], "n_states": eng.n_states,
        "ext_name": inp.prof["external_vol"]["series"],
        "breaker_default": inp.prof["external_vol"]["circuit_breaker_level"],
        "breaker_levels": inp.prof["external_vol"]["breaker_test_levels"],
        "gate": {"threshold": gate.confidence_threshold, "persistence": gate.persistence_bars,
                 "neutral": gate.neutral_on_low_confidence},
        "sizing": {"mode": sizing.mode, "floor": sizing.floor_size, "full": sizing.full_conf,
                   "mapping": sizing.mapping},
        "features": list(inp.feats.columns), "feature_set": inp.prof["feature_set"],
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "close": [round(float(c), 4) for c in close.values],
        "ext": [None if not np.isfinite(v) else round(float(v), 4) for v in inp.ext.values],
        "oos_start": int(i0), "first_valid": first_valid,
        "probs": [[int(round(p * 10000)) for p in row] for row in P],
        "version": fr["version"].astype(int).tolist(),
        "playbooks": [[v.playbook[s]["exposure"] for s in range(eng.n_states)] for v in eng.versions],
        "refits": {"total": len(eng.versions), "by_reason": reasons,
                   "log": [{"date": r["date"], "reason": r["reason"]} for r in eng.retrain_log]},
        "single": single, "ma_window": t["ma_window"], "vt_target": t["vol_target"],
        "vt_lookback": t["vol_target_lookback"], "refs": refs,
        "states": {"names": list(st["name"]), "table": table(st.drop(columns="name"))},
        "tuning_log": tuning_log.entries(symbol),
        "dsr": tuning_log.current_dsr(symbol),
        "reports": {k: [table(df) for df in report_tbl(f"{k}_{symbol}.html")]
                    for k in ["state_count", "feature_lab", "detector_bakeoff", "detection_lag", "ext_vol"]},
    })


def main():
    out = {"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
           "assets": {s: asset(s) for s in ["SPY", "BTC-USD"]}}
    d = ROOT / "web"
    d.mkdir(exist_ok=True)
    (d / "data.json").write_text(json.dumps(out, separators=(",", ":")))
    print(d / "data.json", round((d / "data.json").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
