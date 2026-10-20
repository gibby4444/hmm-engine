"""Prompt 1 — State inspection report. Run after every tuning change.

    python -m tuning.state_report --symbol SPY
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.walk_forward import load_inputs, walk_forward
from core.config import settings
from core.hmm_engine import RegimeModel, name_state, predict_regime_filtered
from tuning import report as R


def visits(states: np.ndarray) -> dict[int, list[int]]:
    out, cur, n = {}, states[0], 1
    for s in states[1:]:
        if s == cur:
            n += 1
        else:
            out.setdefault(int(cur), []).append(n)
            cur, n = s, 1
    out.setdefault(int(cur), []).append(n)
    return out


def inspect(model, inp, bpy: int | None = None) -> dict:
    """Per-state statistics using FILTERED assignments over the full history."""
    bpy = bpy or inp.prof["bars_per_year"]
    valid = np.isfinite(inp.X).all(1)
    idx = inp.px.index[valid]
    P = predict_regime_filtered(model, inp.X[valid])
    st = P.argmax(1)
    r = np.log(inp.px["close"]).diff().loc[idx].values
    fwd = inp.fwd_ret[valid]
    rv = pd.Series(np.log(inp.px["close"]).diff()).rolling(21).std().loc[idx].values * np.sqrt(bpy)
    vol = inp.px["volume"].replace(0, np.nan)
    vr = (vol / vol.rolling(50).mean()).loc[idx].values
    v = visits(st)
    K = model.n_states
    rows = []
    for s in range(K):
        m = st == s
        rows.append({
            "state": s, "share": m.mean(), "ret_mean_ann": np.nanmean(r[m]) * bpy if m.any() else np.nan,
            "ret_std_ann": np.nanstd(r[m]) * np.sqrt(bpy) if m.any() else np.nan,
            "fwd_ret_ann": np.nanmean(fwd[m]) * bpy if m.any() else np.nan,
            "realized_vol": np.nanmean(rv[m]) if m.any() else np.nan,
            "volume_ratio": np.nanmean(vr[m]) if m.any() else np.nan,
            "avg_visit_bars": np.mean(v.get(s, [0])), "visits": len(v.get(s, [])),
            "model_dur_bars": float(model.expected_durations()[s]) if hasattr(model, "expected_durations") else np.nan,
            "avg_conf": P[m, s].mean() if m.any() else np.nan,
        })
    df = pd.DataFrame(rows).set_index("state")
    df["name"] = [name_state(df.loc[s, "ret_mean_ann"], s, K) for s in df.index]
    # redundancy: state means closer than z (scaled feature std units) on every feature
    z = settings()["tuning"]["redundancy_z"]
    red = []
    if hasattr(model, "means"):
        for a in range(K):
            for b in range(a + 1, K):
                d = np.abs(model.means[a] - model.means[b]).max()
                if d < z:
                    red.append((a, b, float(d)))
    # names that collide also signal redundancy
    for a in range(K):
        for b in range(a + 1, K):
            if df.loc[a, "name"] == df.loc[b, "name"] and not any(x[:2] == (a, b) for x in red):
                red.append((a, b, np.nan))
    return {"table": df, "states": pd.Series(st, index=idx), "probs": P, "redundant": red,
            "transmat": getattr(model, "transmat", None)}


def timeline_img(inp, states: pd.Series, title: str) -> str:
    fig, ax = R.styled_ax((11, 4.2))
    px = inp.px["close"].loc[states.index]
    ax.plot(px.index, px.values, color=R.TEXT, lw=0.8)
    ax.set_yscale("log")
    s = states.values
    starts = np.r_[0, np.where(np.diff(s) != 0)[0] + 1]
    ends = np.r_[starts[1:], len(s)]
    for a, b in zip(starts, ends):
        ax.axvspan(states.index[a], states.index[min(b, len(s) - 1)],
                   color=R.STATE_COLORS[s[a] % 6], alpha=0.28, lw=0)
    ax.set_title(title)
    return R.fig_to_img(fig)


def render(symbol: str, model, inp, ins: dict, tag: str = "baseline") -> str:
    df = ins["table"]
    K = model.n_states
    tm = pd.DataFrame(ins["transmat"], index=[f"from {i}" for i in range(K)],
                      columns=[f"to {j}" for j in range(K)]) if ins["transmat"] is not None else None
    legend = " ".join(f'<span style="color:{R.STATE_COLORS[s % 6]}">■</span> {s}: {df.loc[s, "name"]}'
                      for s in df.index)
    summary = "<ul>" + "".join(
        f"<li><b>State {s} — {df.loc[s, 'name']}</b>: {df.loc[s, 'share']:.0%} of bars, "
        f"realised vol {df.loc[s, 'realized_vol']:.0%}, return {df.loc[s, 'ret_mean_ann']:+.0%}/yr "
        f"while in state (next-bar {df.loc[s, 'fwd_ret_ann']:+.0%}/yr), "
        f"average visit {df.loc[s, 'avg_visit_bars']:.1f} bars over {df.loc[s, 'visits']} visits.</li>"
        for s in df.index) + "</ul>"
    if ins["redundant"]:
        summary += R.card("<b>Possible redundancy:</b> " + "; ".join(
            f"states {a} & {b}" + (f" (max mean gap {d:.2f}σ)" if np.isfinite(d) else " (same name)")
            for a, b, d in ins["redundant"]) + ". Consider one fewer state.", "warn")
    else:
        summary += R.card("No redundant state pairs detected.", "verdict")
    flick = (np.diff(ins["states"].values) != 0).sum()
    body = (R.card(f"Model: {K} states · features {list(inp.feats.columns)} · trained "
                   f"{getattr(model, 'meta', {}).get('train_span', '')} · raw (ungated) state changes over "
                   f"history: <b>{flick}</b>") +
            "<h2>Per-state statistics</h2>" +
            R.table(df.drop(columns="name").assign(name=df["name"]),
                    {"share": "{:.0%}", "ret_mean_ann": "{:+.1%}", "ret_std_ann": "{:.1%}",
                     "fwd_ret_ann": "{:+.1%}", "realized_vol": "{:.1%}", "volume_ratio": "{:.2f}",
                     "avg_visit_bars": "{:.1f}", "model_dur_bars": "{:.1f}", "avg_conf": "{:.2f}"}) +
            "<h2>Transition probabilities</h2>" + (R.table(tm, {c: "{:.3f}" for c in tm.columns}) if tm is not None else "") +
            "<h2>Timeline</h2><div class='sub'>" + legend + "</div>" +
            timeline_img(inp, ins["states"], f"{symbol} — filtered state assignments") +
            "<h2>Plain-language summary</h2>" + summary)
    return R.write(f"state_report_{symbol}_{tag}.html", f"State inspection — {symbol}", body,
                   f"{tag} · filtered (forward-only) assignments")


def latest_model(symbol: str):
    inp = load_inputs(symbol)
    eng = walk_forward(inp)
    m = eng.model
    m.meta["train_span"] = f"{eng.versions[-1].train_start} → {eng.versions[-1].train_end}"
    return m, inp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    m, inp = latest_model(a.symbol)
    fp = render(a.symbol, m, inp, inspect(m, inp), a.tag)
    print(fp)


if __name__ == "__main__":
    main()
