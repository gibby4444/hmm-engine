"""Retraining scheduler with drift detection (Prompt 8).

`RetrainEngine.on_bar(t)` is THE retraining path. Live operation calls it once per
new bar; walk-forward validation calls it in a loop over history. There is no
second implementation, and tests/test_retraining.py fails if a live-style
incremental run and the backtest loop ever disagree.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeModel, state_playbook


@dataclass
class RetrainConfig:
    every_bars: int = 21
    window: str = "expanding"
    window_bars: int = 2520
    min_train_bars: int = 756
    drift_ll_z: float = -3.0
    drift_lowconf_frac: float = 0.6
    drift_lookback: int = 20
    confidence_threshold: float = 0.70
    cooldown_bars: int = 5
    drift_enabled: bool = True

    @classmethod
    def from_profile(cls, prof: dict, **over):
        r = dict(prof.get("retraining", {}))
        r["confidence_threshold"] = prof.get("gating", {}).get("confidence_threshold", 0.7)
        r.update(over)
        return cls(**{k: v for k, v in r.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelVersion:
    version: int
    fit_bar: int
    fit_date: str
    train_start: str
    train_end: str
    reason: str
    playbook: dict
    durations: list
    means: list
    transmat: list


@dataclass
class RetrainEngine:
    X: np.ndarray                 # point-in-time features (T x d), rows may be NaN early
    fwd_ret: np.ndarray           # next-bar log return aligned to row t (r_{t+1})
    vol_proxy: np.ndarray         # realised vol for canonical state ordering
    index: pd.DatetimeIndex
    n_states: int
    cfg: RetrainConfig
    bars_per_year: int = 252
    seed: int = 0
    n_init: int = 3
    persist_dir: Path | None = None
    model_factory: callable = None   # lets the bake-off swap in GMM / k-means

    model: object = None
    probs: np.ndarray | None = None
    last_fit: int = -10**9
    versions: list = field(default_factory=list)
    hist_probs: dict = field(default_factory=dict)
    hist_pll: dict = field(default_factory=dict)
    hist_version: dict = field(default_factory=dict)
    retrain_log: list = field(default_factory=list)

    def __post_init__(self):
        self.valid = np.isfinite(self.X).all(axis=1)
        self.first_valid = int(np.argmax(self.valid))

    # ── policy ────────────────────────────────────────────────────────────
    def _train_slice(self, t: int) -> slice:
        lo = self.first_valid
        if self.cfg.window == "rolling":
            lo = max(lo, t + 1 - self.cfg.window_bars)
        return slice(lo, t + 1)

    def _due(self, t: int) -> str | None:
        if t - self.first_valid + 1 < self.cfg.min_train_bars:
            return None
        if self.model is None:
            return "initial"
        if t - self.last_fit >= self.cfg.every_bars:
            return "calendar"
        if self.cfg.drift_enabled and t - self.last_fit >= max(self.cfg.cooldown_bars, self.cfg.drift_lookback):
            L = self.cfg.drift_lookback
            recent = [self.hist_pll[i] for i in range(t - L, t)
                      if i in self.hist_pll and np.isfinite(self.hist_pll[i])]
            if len(recent) >= L - 1 and hasattr(self.model, "train_pred_ll_mean"):
                z = (np.mean(recent) - self.model.train_pred_ll_mean) / (
                    self.model.train_pred_ll_std / np.sqrt(L))
                if z < self.cfg.drift_ll_z:
                    return f"drift_loglik(z={z:.1f})"
            pr = [self.hist_probs[i].max() for i in range(t - L, t) if i in self.hist_probs]
            if len(pr) == L:
                frac = np.mean(np.array(pr) < self.cfg.confidence_threshold)
                if frac > self.cfg.drift_lowconf_frac:
                    return f"drift_lowconf({frac:.0%})"
        return None

    # ── fitting ───────────────────────────────────────────────────────────
    def _fit(self, t: int, reason: str):
        sl = self._train_slice(t)
        Xtr = self.X[sl]
        if self.model_factory is not None:
            m = self.model_factory(self.n_states, self.seed)
            m.fit(Xtr, vol_proxy=self.vol_proxy[sl])
        else:
            m = RegimeModel(self.n_states, seed=self.seed, n_init=self.n_init).fit(
                Xtr, vol_proxy=self.vol_proxy[sl])
        p_train = m.filter(Xtr)
        # playbook uses only forward returns already realised at t (exclude last row)
        hard = p_train.argmax(1)[:-1]
        fr = self.fwd_ret[sl][:-1]
        m.playbook = state_playbook(hard, fr, self.n_states, self.bars_per_year)
        self.model, self.probs, self.last_fit = m, p_train[-1], t
        v = ModelVersion(len(self.versions), t, str(self.index[t].date()),
                         str(self.index[sl.start].date()), str(self.index[t].date()), reason,
                         {int(k): v for k, v in m.playbook.items()},
                         list(np.round(m.expected_durations(), 2)) if hasattr(m, "expected_durations") else [],
                         np.round(getattr(m, "means", np.zeros((1, 1))), 3).tolist(),
                         np.round(getattr(m, "transmat", np.zeros((1, 1))), 3).tolist())
        self.versions.append(v)
        self.retrain_log.append({"bar": t, "date": v.fit_date, "reason": reason})
        if self.persist_dir:
            d = Path(self.persist_dir)
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"model_v{v.version:04d}.pkl", "wb") as f:
                pickle.dump(m, f)
            with open(d / f"model_v{v.version:04d}.json", "w") as f:
                json.dump(v.__dict__, f, indent=1, default=float)
        return p_train[-1], np.nan

    # ── the one path ──────────────────────────────────────────────────────
    def on_bar(self, t: int):
        """Process the close of bar t. Returns (probs, pred_ll) or None pre-warmup."""
        if not self.valid[t]:
            return None
        reason = self._due(t)
        if reason:
            p, ll = self._fit(t, reason)
        elif self.model is None:
            return None
        else:
            p, ll = self.model.step(self.probs, self.X[t])
            self.probs = p
        self.hist_probs[t], self.hist_pll[t], self.hist_version[t] = p, ll, len(self.versions) - 1
        return p, ll

    def run(self, start: int = 0, end: int | None = None):
        end = len(self.X) if end is None else end
        for t in range(start, end):
            self.on_bar(t)
        return self

    # ── outputs for the backtester ────────────────────────────────────────
    def frame(self) -> pd.DataFrame:
        ts = sorted(self.hist_probs)
        P = np.vstack([self.hist_probs[t] for t in ts])
        df = pd.DataFrame(P, index=self.index[ts], columns=[f"prob_{k}" for k in range(P.shape[1])])
        df["version"] = [self.hist_version[t] for t in ts]
        df["pred_ll"] = [self.hist_pll[t] for t in ts]
        df["bar"] = ts
        return df

    def exposure_for(self, regimes: np.ndarray, versions: np.ndarray, neutral_exposure=0.0) -> np.ndarray:
        out = np.empty(len(regimes))
        for i, (s, v) in enumerate(zip(regimes, versions)):
            out[i] = neutral_exposure if s < 0 else self.versions[v].playbook[int(s)]["exposure"]
        return out


class LiveRunner:
    """Live wrapper: each new bar, recompute point-in-time features from the price
    history available so far, append them, and call the SAME RetrainEngine.on_bar.
    Exists so tests can prove live and backtest retraining never diverge."""

    def __init__(self, feature_spec, n_states, cfg: RetrainConfig, bars_per_year=252, seed=0, n_init=None,
                 ext=None):
        self.spec, self.ext = feature_spec, ext
        self.engine = None
        self.kw = dict(n_states=n_states, cfg=cfg, bars_per_year=bars_per_year, seed=seed)
        if n_init is not None:
            self.kw["n_init"] = n_init

    def on_new_bar(self, px_so_far: pd.DataFrame):
        from data.features import build_features
        e = None if self.ext is None else self.ext.loc[: px_so_far.index[-1]]
        X = build_features(px_so_far, self.spec, ext=e).values
        r = np.log(px_so_far["close"]).diff()
        fwd = np.r_[r.values[1:], np.nan]  # future unknown at the last bar
        vp = r.rolling(21).std().fillna(r.expanding().std()).fillna(0).values
        if self.engine is None:
            self.engine = RetrainEngine(X, fwd, vp, px_so_far.index, **self.kw)
        else:
            self.engine.X, self.engine.fwd_ret, self.engine.vol_proxy = X, fwd, vp
            self.engine.index = px_so_far.index
            self.engine.valid = np.isfinite(X).all(axis=1)
            if not self.engine.valid[: len(X) - 1].any():
                self.engine.first_valid = int(np.argmax(self.engine.valid))
        return self.engine.on_bar(len(X) - 1)
