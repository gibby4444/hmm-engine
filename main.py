"""Entry point for the tuning toolkit.

    python main.py tune --symbol SPY           # runs the whole loop, writes every report
    python main.py inspect --symbol SPY        # Prompt 1 only
    python main.py validate --symbol SPY       # Prompt 9 gate only

paper_trading is true in settings.yaml and nothing in this toolkit places orders.
"""
from __future__ import annotations

import argparse
import sys

from core.config import settings
from monitoring.logger import log_event


def tune(symbol: str):
    from backtest.walk_forward import load_inputs, walk_forward
    from tuning import (detection_lag, detector_bakeoff, ext_vol, feature_lab, gating_eval,
                        regime_validation, state_count, state_report, tuning_log)
    from core.gating import GateConfig

    inp = load_inputs(symbol)
    eng = walk_forward(inp)
    m = eng.model
    m.meta["train_span"] = f"{eng.versions[-1].train_start} → {eng.versions[-1].train_end}"
    out = {"state_report": state_report.render(symbol, m, inp, state_report.inspect(m, inp))}

    df, det, _ = state_count.compare(symbol)
    out["state_count"] = state_count.render(symbol, df, det)

    base = inp.prof["feature_set"]
    fam = base.split("_")[0]
    exps = [(f"(a) {base} vs {fam}_long — longer windows", feature_lab.ab(symbol, base, f"{fam}_long")),
            (f"(b) {base} vs {fam}_short — shorter windows", feature_lab.ab(symbol, base, f"{fam}_short")),
            (f"(c) {base} + trailing drawdown feature", feature_lab.ab(symbol, base, f"{fam}_plus_dd"))]
    out["feature_lab"] = feature_lab.render(symbol, exps)

    oo, logs = gating_eval.gating_on_off(inp, eng)
    out["gating_sizing"] = gating_eval.render(symbol, oo, logs, gating_eval.sizing_modes(inp, eng))

    i2, sweep, ab = ext_vol.run(symbol)
    out["ext_vol"] = ext_vol.render(symbol, i2, sweep, ab)

    val = regime_validation.validate(inp=inp, eng=eng)
    out["validation"] = regime_validation.render(val)

    res, fa = detection_lag.analyse(inp, eng, GateConfig.from_profile(inp.prof))
    out["detection_lag"] = detection_lag.render(symbol, res, detection_lag.summary(res), fa,
                                                detection_lag.tradeoff(inp, eng))

    table, rec, _ = detector_bakeoff.bakeoff(symbol)
    out["bakeoff"] = detector_bakeoff.render(symbol, table, rec)

    if not tuning_log.entries(symbol):
        cfg = {k: inp.prof[k] for k in ("feature_set", "n_states", "gating", "sizing", "retraining", "costs_bps")}
        tuning_log.record_with_validation(symbol, "baseline profile", "starting point", "baseline",
                                          val["verdict"][:60], None, val, cfg)
        # the feature-lab and detector comparisons above were trials too
        for title, r in exps:
            tuning_log.record(symbol, title, "feature-window/feature-add A/B", "keep" if r["keep"] else "revert",
                              "; ".join(r["reasons"]))
        tuning_log.record(symbol, "detector bake-off (GMM, k-means)", "simpler detector may match HMM", "revert",
                          rec[:120])
    out["tuning_log"] = tuning_log.render(symbol, inp.prof["bars_per_year"])
    log_event("tuning_run_complete", symbol=symbol, verdict=val["verdict"], reports={k: str(v) for k, v in out.items()})
    return out, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "inspect", "validate"])
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    assert settings()["paper_trading"] is True, "paper_trading must stay true for the tuning toolkit"
    if a.cmd == "tune":
        out, val = tune(a.symbol)
        for k, v in out.items():
            print(f"{k:14s} {v}")
        print(val["verdict"])
    elif a.cmd == "inspect":
        from tuning import state_report
        sys.argv = [sys.argv[0], "--symbol", a.symbol]
        state_report.main()
    else:
        from tuning import regime_validation
        sys.argv = [sys.argv[0], "--symbol", a.symbol]
        regime_validation.main()


if __name__ == "__main__":
    main()
