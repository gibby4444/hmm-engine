"""Prompt 5 (evaluation) + Prompt 6 (sizing comparison).

    python -m tuning.gating_eval --symbol SPY

Replays cached walk-forward probabilities with gating on vs off, and the three
sizing modes, through the same backtest engine. No refitting involved.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.walk_forward import evaluate, load_inputs, system_exposure, walk_forward
from core.confidence_sizing import SizingConfig
from core.gating import GateConfig
from tuning import report as R

COLS = ["cagr", "sharpe", "max_dd", "calmar", "trades", "cost_drag_ann"]


def gating_on_off(inp, eng):
    rows, logs = {}, {}
    oos0 = eng.frame().index[0]
    for name, g in {"gating OFF (raw argmax)": GateConfig(0.0, 1),
                    "gating ON (profile)": GateConfig.from_profile(inp.prof)}.items():
        exp, gr, _, _ = system_exposure(eng, inp, g, SizingConfig(mode="hard"))
        m = evaluate(exp, inp, start=oos0)
        rows[name] = {"raw flickers": gr.raw_flickers, "confirmed switches": gr.confirmed_switches,
                      "absorbed": gr.absorbed, **{k: m[k] for k in COLS}}
        logs[name] = gr.log
    return pd.DataFrame(rows).T, logs


def sizing_modes(inp, eng):
    g = GateConfig.from_profile(inp.prof)
    oos0 = eng.frame().index[0]
    rows = {}
    for label, gate, mode in [("hard gating alone", g, "hard"),
                              ("confidence scaling alone", GateConfig(0.0, 1), "confidence"),
                              ("both (gate → scale)", g, "both")]:
        sz = SizingConfig.from_profile(inp.prof, mode=mode)
        exp, *_ = system_exposure(eng, inp, gate, sz)
        m = evaluate(exp, inp, start=oos0)
        rows[label] = {k: m[k] for k in ["cagr", "vol", "sharpe", "sortino", "max_dd", "calmar",
                                         "exposure_avg", "trades", "cost_drag_ann"]}
    return pd.DataFrame(rows).T


def render(symbol, onoff, logs, modes):
    fmt = {**R.PERF_FMT, "raw flickers": "{:.0f}", "confirmed switches": "{:.0f}", "absorbed": "{:.0%}"}
    on = logs["gating ON (profile)"]
    sample = on[on["type"] == "confirmed_switch"].tail(12) if len(on) else on
    body = ("<h2>Gating on vs off (out-of-sample replay)</h2>" + R.table(onoff, fmt) +
            R.card("Raw flickers = bar-to-bar changes in the HMM's most-likely state. Confirmed switches = what "
                   "reached the router. Trades are lower than switches when adjacent states share a playbook.") +
            "<h3>Last confirmed switches (full log in the returned DataFrame)</h3>" +
            (R.table(sample.set_index("time"), {"prob": "{:.2f}"}) if len(sample) else "<p>none</p>") +
            "<h2>Sizing modes (Prompt 6)</h2>" + R.table(modes, R.PERF_FMT) +
            R.card("Whichever mode wins here is a candidate only: it must also hold up in the walk-forward "
                   "validation harness (Prompt 9) before adoption.", "warn"))
    return R.write(f"gating_sizing_{symbol}.html", f"Gating & sizing — {symbol}", body, "replay of cached walk-forward")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    inp = load_inputs(a.symbol)
    eng = walk_forward(inp)
    oo, logs = gating_on_off(inp, eng)
    md = sizing_modes(inp, eng)
    print(oo.round(3).to_string())
    print(md.round(3).to_string())
    print(render(a.symbol, oo, logs, md))


if __name__ == "__main__":
    main()
