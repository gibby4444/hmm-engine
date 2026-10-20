"""Prompt 10 — Detection lag analyzer. Lag cannot be eliminated, only traded against noise.

    python -m tuning.detection_lag --symbol SPY

Reference turns are identified WITH HINDSIGHT (that's the point — they are the
answer key, never a trading input):
  * drawdown onsets: the peak before every drawdown deeper than the threshold
  * volatility spikes: 21-bar realised vol crossing k x its trailing 1-year median
  * trend starts: the trough of each of those drawdowns
Lag is measured two ways: to the gated model confirming its highest-vol state
("state lag"), and to the final exposure actually going defensive ("exposure lag").
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.walk_forward import load_inputs, system_exposure, walk_forward
from core.confidence_sizing import SizingConfig
from core.config import settings
from core.gating import GateConfig, apply_gates
from tuning import report as R

MAX_LAG = 126


def reference_turns(close: pd.Series, dd_thr: float, spike_mult: float, min_gap: int, bpy: int) -> pd.DataFrame:
    turns = []
    peak_i, in_dd, trough_i = 0, False, 0
    c = close.values
    for i in range(1, len(c)):
        if not in_dd:
            if c[i] >= c[peak_i]:
                peak_i = i
            elif c[i] <= c[peak_i] * (1 - dd_thr):
                in_dd, trough_i = True, i
                turns.append({"time": close.index[peak_i], "type": "drawdown onset", "i": peak_i})
        else:
            if c[i] < c[trough_i]:
                trough_i = i
            if c[i] >= c[peak_i]:
                turns.append({"time": close.index[trough_i], "type": "trend start", "i": trough_i})
                in_dd, peak_i = False, i
    if in_dd:
        turns.append({"time": close.index[trough_i], "type": "trend start", "i": trough_i})
    rv = np.log(close).diff().rolling(21).std()
    med = rv.rolling(bpy, min_periods=bpy // 2).median().shift(1)
    spike = (rv > spike_mult * med) & ~(rv.shift(1) > spike_mult * med.shift(1))
    last = -10**9
    for i in np.where(spike.values)[0]:
        if i - last >= min_gap:
            turns.append({"time": close.index[i], "type": "vol spike", "i": int(i)})
            last = i
    return pd.DataFrame(turns).sort_values("i").reset_index(drop=True)


def measure(close, turns, regime, exposure, stress_state, start_i):
    rows = []
    for _, t in turns.iterrows():
        i = int(t["i"])
        if i < start_i:
            continue
        defensive = t["type"] != "trend start"
        s_hit = e_hit = None
        for j in range(i, min(i + MAX_LAG, len(close))):
            if s_hit is None and ((regime[j] == stress_state) == defensive):
                s_hit = j - i
            if e_hit is None and ((exposure[j] <= 1e-9) == defensive):
                e_hit = j - i
        # adverse move from the reference turn to exposure confirmation
        adverse = np.nan
        if e_hit is not None:
            mv = close.iloc[i + e_hit] / close.iloc[i] - 1
            adverse = mv if defensive else -mv   # negative = cost paid by being late
        rows.append({"time": t["time"].date(), "type": t["type"], "state lag": s_hit, "exposure lag": e_hit,
                     "adverse move": adverse})
    return pd.DataFrame(rows)


def analyse(inp, eng, gate: GateConfig):
    s = settings()["tuning"]["lag"]
    thr = s["dd_threshold_crypto"] if inp.prof["profile_name"] == "crypto" else s["dd_threshold_equity"]
    close = inp.px["close"]
    turns = reference_turns(close, thr, s["vol_spike_mult"], s["min_gap_bars"], inp.prof["bars_per_year"])
    exp, g, _, fr = system_exposure(eng, inp, gate, SizingConfig.from_profile(inp.prof))
    reg = pd.Series(-2, index=close.index)
    reg.loc[fr.index] = g.regime
    start_i = close.index.get_loc(fr.index[0])
    res = measure(close, turns, reg.values, exp.values, eng.n_states - 1, start_i)
    # false alarms: confirmed entries into the stress state not within MAX_LAG/2 bars after a defensive turn
    entries = np.where((np.diff(g.regime) != 0) & (g.regime[1:] == eng.n_states - 1))[0] + 1
    ref_i = turns.loc[turns["type"] != "trend start", "i"].values - start_i
    fa = sum(not ((ref_i <= e) & (e - ref_i <= MAX_LAG // 2)).any() for e in entries)
    yrs = len(fr) / inp.prof["bars_per_year"]
    return res, {"stress entries": len(entries), "false alarms/yr": fa / yrs}


def summary(res: pd.DataFrame) -> dict:
    d = res[res["type"] != "trend start"]
    sl = d["state lag"].dropna()
    return {"turns": len(d), "median state lag": sl.median() if len(sl) else np.nan,
            "worst state lag": sl.max() if len(sl) else np.nan, "missed (state)": int(d["state lag"].isna().sum()),
            "median exposure lag": d["exposure lag"].dropna().median() if d["exposure lag"].notna().any() else np.nan,
            "missed (exposure)": int(d["exposure lag"].isna().sum()),
            "avg adverse move": d["adverse move"].mean()}


def tradeoff(inp, eng) -> pd.DataFrame:
    gg = settings()["tuning"]["gate_grid"]
    rows = []
    for th in gg["thresholds"]:
        for n in gg["persistence"]:
            res, fa = analyse(inp, eng, GateConfig(th, n))
            rows.append({"threshold": th, "persistence": n, **summary(res), **fa})
    return pd.DataFrame(rows).set_index(["threshold", "persistence"])


def render(symbol, res, sm, fa, grid):
    fmt = {"state lag": "{:.0f}", "exposure lag": "{:.0f}", "adverse move": "{:+.1%}"}
    gfmt = {"turns": "{:.0f}", "median state lag": "{:.1f}", "worst state lag": "{:.0f}", "missed (state)": "{:.0f}",
            "median exposure lag": "{:.1f}", "missed (exposure)": "{:.0f}", "avg adverse move": "{:+.1%}",
            "stress entries": "{:.0f}", "false alarms/yr": "{:.2f}"}
    g2 = grid.copy()
    g2.index = [f"thr {a:.2f} · N {b}" for a, b in grid.index]
    body = (R.card("Lag cannot be eliminated, only traded against noise. This report shows where your current "
                   "settings sit on that curve. Missed = no confirmation within "
                   f"{MAX_LAG} bars.", "verdict") +
            "<h2>Current settings — summary</h2>" + R.table(pd.DataFrame([{**sm, **fa}], index=["profile"]), gfmt) +
            "<h2>Every reference turn (out-of-sample period)</h2>" + R.table(res.set_index("time"), fmt) +
            "<h2>Lag vs false-alarm trade-off across gate settings</h2>" + R.table(g2, gfmt))
    return R.write(f"detection_lag_{symbol}.html", f"Detection lag — {symbol}", body,
                   "adverse move: price change between the hindsight turn and the model's confirmation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    inp = load_inputs(a.symbol)
    eng = walk_forward(inp)
    res, fa = analyse(inp, eng, GateConfig.from_profile(inp.prof))
    grid = tradeoff(inp, eng)
    print(pd.Series(summary(res)))
    print(render(a.symbol, res, summary(res), fa, grid))


if __name__ == "__main__":
    main()
