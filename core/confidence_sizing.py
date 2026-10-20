"""Confidence-scaled position sizing (Prompt 6).

Order of operations is fixed and enforced in `final_exposure`:
  gates decide the regime → router maps regime to base exposure →
  confidence multiplier scales it → risk manager caps ALWAYS win.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.risk_manager import cap_exposure


@dataclass
class SizingConfig:
    mode: str = "hard"          # hard | confidence | both
    floor_size: float = 0.5     # size at the gate threshold
    full_conf: float = 0.95     # size = 100% at or above this probability
    mapping: str = "linear"     # linear | stepped
    threshold: float = 0.70     # gate threshold (anchor for floor_size)
    n_states: int = 3

    @classmethod
    def from_profile(cls, prof: dict, **over):
        s = dict(prof.get("sizing", {}))
        s.setdefault("threshold", prof.get("gating", {}).get("confidence_threshold", 0.7))
        s.setdefault("n_states", prof.get("n_states", 3))
        s.update(over)
        return cls(**{k: v for k, v in s.items() if k in cls.__dataclass_fields__})


def confidence_multiplier(p: np.ndarray, cfg: SizingConfig) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    lo, thr, hi = 1.0 / cfg.n_states, cfg.threshold, cfg.full_conf
    if cfg.mapping == "stepped":
        m = np.where(p >= hi, 1.0, np.where(p >= (thr + hi) / 2, (1 + cfg.floor_size) / 2,
                     np.where(p >= thr, cfg.floor_size, cfg.floor_size / 2)))
        return m
    above = cfg.floor_size + (1 - cfg.floor_size) * np.clip((p - thr) / max(1e-9, hi - thr), 0, 1)
    below = cfg.floor_size * np.clip((p - lo) / max(1e-9, thr - lo), 0, 1)
    return np.where(p >= thr, above, below)


def final_exposure(base_exposure: np.ndarray, confidence: np.ndarray, cfg: SizingConfig,
                   risk: dict) -> np.ndarray:
    exp = np.asarray(base_exposure, dtype=float)
    if cfg.mode in ("confidence", "both"):
        exp = exp * confidence_multiplier(confidence, cfg)
    return cap_exposure(exp, risk)   # risk caps outrank everything above
