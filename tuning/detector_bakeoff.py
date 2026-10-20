"""Prompt 11 — Alternative-detector bake-off. Same features, gating, walk-forward,
costs and retraining schedule for every detector; the validation harness judges.

    python -m tuning.detector_bakeoff --symbol SPY

Detectors expose the same interface as RegimeModel (fit / filter / step), so the
one RetrainEngine runs all of them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from backtest.walk_forward import load_inputs, walk_forward
from core.hmm_engine import Scaler
from tuning import report as R
from tuning.regime_validation import validate


@dataclass
class _Static:
    """Detector with no transition structure: probabilities depend on x_t only."""
    n_states: int
    seed: int = 0
    scaler: Scaler = field(default_factory=Scaler)
    meta: dict = field(default_factory=dict)

    def _order(self, Z, vol_proxy):
        hard = self._raw(Z).argmax(1)
        key = np.array([np.nanmean(vol_proxy[hard == s]) if (hard == s).any() else np.inf
                        for s in range(self.n_states)])
        self.order = np.argsort(key)

    def filter(self, X, return_pred_ll=False, init=None):
        P = self._raw(self.scaler.transform(X))[:, self.order]
        return (P, np.full(len(X), np.nan)) if return_pred_ll else P

    def step(self, prev, x):
        return self.filter(np.atleast_2d(x))[-1], np.nan


@dataclass
class GMMDetector(_Static):
    def fit(self, X, vol_proxy):
        Z = self.scaler.fit(X).transform(X)
        self.m = GaussianMixture(self.n_states, covariance_type="full", n_init=3,
                                 random_state=self.seed).fit(Z)
        self._order(Z, vol_proxy)
        return self

    def _raw(self, Z):
        return self.m.predict_proba(Z)


@dataclass
class KMeansDetector(_Static):
    def fit(self, X, vol_proxy):
        Z = self.scaler.fit(X).transform(X)
        self.m = KMeans(self.n_states, n_init=10, random_state=self.seed).fit(Z)
        self._order(Z, vol_proxy)
        return self

    def _raw(self, Z):
        lab = self.m.predict(Z)
        P = np.zeros((len(Z), self.n_states))
        P[np.arange(len(Z)), lab] = 1.0   # hard assignment: confidence gate is moot, persistence still applies
        return P


def gmm_factory(k, seed):
    return GMMDetector(k, seed)


def kmeans_factory(k, seed):
    return KMeansDetector(k, seed)


def bakeoff(symbol: str):
    inp = load_inputs(symbol)
    results = {}
    for name, fac, tag in [("HMM (tuned)", None, "hmm"), ("Gaussian mixture", gmm_factory, "gmm"),
                           ("k-means", kmeans_factory, "kmeans")]:
        eng = walk_forward(inp, model_factory=fac, tag=tag)
        results[name] = validate(inp=inp, eng=eng, label=name)
    rows = {}
    for name, res in results.items():
        rows[name] = res["OOS"].iloc[0]
    any_res = next(iter(results.values()))
    rows[any_res["OOS"].index[1]] = any_res["OOS"].iloc[1]   # MA floor
    rows[any_res["OOS"].index[2]] = any_res["OOS"].iloc[2]   # best single
    table = pd.DataFrame(rows).T
    detectors = list(results)
    best = max(detectors, key=lambda d: table.loc[d, "sharpe"])
    floor = max(table.loc[[any_res["OOS"].index[1], any_res["OOS"].index[2]], "sharpe"])
    if table.loc[best, "sharpe"] <= floor:
        rec = (f"No detector beats the simple floor out-of-sample (best detector {best}: "
               f"{table.loc[best, 'sharpe']:.2f} Sharpe vs floor {floor:.2f}). Run the simpler alternative.")
    elif best == "HMM (tuned)":
        rec = "HMM wins against both the simpler detectors and the floor — keep it, with this evidence."
    else:
        rec = (f"{best} beats the HMM out-of-sample with less machinery. Switch to it (or "
               "explain why transition structure should matter before keeping the HMM).")
    return table, rec, results


def render(symbol, table, rec):
    body = (R.card(f"<b>{rec}</b>", "verdict") +
            R.card("Standing rule: complexity must beat the simpler alternative out-of-sample or it doesn't ship. "
                   "All detectors use identical features, gating, walk-forward refits and costs.") +
            R.table(table, R.PERF_FMT))
    return R.write(f"detector_bakeoff_{symbol}.html", f"Detector bake-off — {symbol}", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    a = ap.parse_args()
    t, rec, _ = bakeoff(a.symbol)
    print(t.round(3).to_string())
    print(rec)
    print(render(a.symbol, t, rec))


if __name__ == "__main__":
    main()
