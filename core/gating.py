"""Signal gating layer between raw HMM output and the strategy router (Prompt 5).

Stacked gates: a switch to a new state is confirmed only when that state has been
BOTH the most likely state AND above the confidence threshold for N consecutive
bars. Sure enough, for long enough, then switch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NEUTRAL = -1


@dataclass
class GateConfig:
    confidence_threshold: float = 0.70
    persistence_bars: int = 3
    neutral_on_low_confidence: bool = False

    @classmethod
    def from_profile(cls, prof: dict, **over):
        g = dict(prof.get("gating", {}))
        g.update(over)
        return cls(**{k: g[k] for k in ("confidence_threshold", "persistence_bars",
                                         "neutral_on_low_confidence") if k in g})


@dataclass
class GateResult:
    regime: np.ndarray          # confirmed state per bar (NEUTRAL = -1)
    confidence: np.ndarray      # prob of the confirmed state at each bar
    raw: np.ndarray             # raw argmax per bar
    log: pd.DataFrame           # every raw flicker and every confirmed switch

    @property
    def raw_flickers(self) -> int:
        return int((np.diff(self.raw) != 0).sum())

    @property
    def confirmed_switches(self) -> int:
        r = self.regime
        return int((np.diff(r) != 0).sum())

    @property
    def absorbed(self) -> float:
        return 1 - self.confirmed_switches / max(1, self.raw_flickers)


def apply_gates(probs: np.ndarray, cfg: GateConfig, index=None) -> GateResult:
    T, K = probs.shape
    raw = probs.argmax(1)
    regime = np.empty(T, dtype=int)
    conf = np.empty(T)
    confirmed = raw[0] if probs[0, raw[0]] >= cfg.confidence_threshold else (
        NEUTRAL if cfg.neutral_on_low_confidence else raw[0])
    cand, streak = None, 0
    events = []
    idx = index if index is not None else np.arange(T)
    for t in range(T):
        top = raw[t]
        p_top = probs[t, top]
        if t > 0 and raw[t] != raw[t - 1]:
            events.append({"time": idx[t], "type": "raw_flicker", "from": int(raw[t - 1]),
                           "to": int(top), "prob": float(p_top)})
        if top != confirmed and p_top >= cfg.confidence_threshold:
            streak = streak + 1 if cand == top else 1
            cand = top
            if streak >= cfg.persistence_bars:
                events.append({"time": idx[t], "type": "confirmed_switch", "from": int(confirmed),
                               "to": int(top), "prob": float(p_top)})
                confirmed, cand, streak = top, None, 0
        else:
            cand, streak = None, 0
            # optional neutral stance when nothing is convincing
            if cfg.neutral_on_low_confidence and confirmed != NEUTRAL and \
                    probs[t].max() < cfg.confidence_threshold and probs[t, confirmed] < cfg.confidence_threshold:
                events.append({"time": idx[t], "type": "confirmed_switch", "from": int(confirmed),
                               "to": NEUTRAL, "prob": float(probs[t].max())})
                confirmed = NEUTRAL
        regime[t] = confirmed
        conf[t] = probs[t, confirmed] if confirmed != NEUTRAL else probs[t].max()
    return GateResult(regime, conf, raw, pd.DataFrame(events))


def ungated(probs: np.ndarray) -> GateResult:
    return apply_gates(probs, GateConfig(confidence_threshold=0.0, persistence_bars=1))
