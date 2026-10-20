"""HMM engine.

Two details that most regime-bot templates get wrong, both fixed here:

1. hmmlearn's predict()/predict_proba() use the forward-BACKWARD algorithm: the
   state probability at bar t is conditioned on bars t+1..T. That is look-ahead.
   Trading decisions must use FILTERED probabilities p(s_t | x_1..x_t), computed
   here with an explicit forward recursion.
2. Feature scaling is fitted on the training window only.

States are relabelled canonically after every fit (0 = lowest-vol state), so a
refit that merely permutes labels does not look like a regime change downstream.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.config import settings

_HMM = settings()["hmm"]

warnings.filterwarnings("ignore", module="hmmlearn")


@dataclass
class Scaler:
    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


@dataclass
class RegimeModel:
    n_states: int
    seed: int = 0
    n_init: int = _HMM["n_init"]
    covariance_type: str = _HMM["covariance_type"]
    scaler: Scaler = field(default_factory=Scaler)
    startprob: np.ndarray | None = None
    transmat: np.ndarray | None = None
    means: np.ndarray | None = None
    covars: np.ndarray | None = None
    train_loglik_per_bar: float = np.nan
    train_pred_ll_mean: float = np.nan
    train_pred_ll_std: float = np.nan
    feature_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ── fitting ────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, vol_proxy: np.ndarray | None = None) -> "RegimeModel":
        from hmmlearn.hmm import GaussianHMM

        self.scaler.fit(X)
        Z = self.scaler.transform(X)
        best, best_ll = None, -np.inf
        for k in range(self.n_init):
            m = GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type,
                            n_iter=_HMM["n_iter"], tol=_HMM["tol"],
                            random_state=self.seed * 1000 + k, min_covar=_HMM["min_covar"])
            try:
                m.fit(Z)
                ll = m.score(Z)
            except Exception:  # noqa: BLE001  degenerate init
                continue
            if np.isfinite(ll) and ll > best_ll:
                best, best_ll = m, ll
        if best is None:
            raise RuntimeError("HMM failed to fit on all initialisations")
        self.startprob, self.transmat = best.startprob_.copy(), best.transmat_.copy()
        self.means, self.covars = best.means_.copy(), best.covars_.copy()
        self._canonicalise(Z, vol_proxy)
        self.train_loglik_per_bar = best_ll / len(Z)
        _, pred_ll = self.filter(X, return_pred_ll=True)
        self.train_pred_ll_mean = float(np.mean(pred_ll[20:]))
        self.train_pred_ll_std = float(np.std(pred_ll[20:]) + 1e-9)
        return self

    def _canonicalise(self, Z, vol_proxy):
        """Order states by average realised vol in-state (ascending)."""
        post = self._filter_scaled(Z)
        hard = post.argmax(1)
        if vol_proxy is None:
            vol_proxy = np.abs(Z[:, 0])
        key = np.array([np.nanmean(vol_proxy[hard == s]) if (hard == s).any() else np.inf
                        for s in range(self.n_states)])
        order = np.argsort(key)
        self.startprob = self.startprob[order]
        self.transmat = self.transmat[np.ix_(order, order)]
        self.means, self.covars = self.means[order], self.covars[order]

    # ── inference ──────────────────────────────────────────────────────────
    def _prep(self):
        """Cache Cholesky factors for fast emission log-densities."""
        key = id(self.covars), id(self.means)
        if getattr(self, "_cache_key", None) == key:
            return
        d = self.means.shape[1]
        self._L, self._logdet = [], []
        for s in range(self.n_states):
            C = self.covars[s] + 1e-9 * np.eye(d)
            L = np.linalg.cholesky(C)
            self._L.append(np.linalg.inv(L))
            self._logdet.append(2 * np.log(np.diag(L)).sum())
        self._const = -0.5 * d * np.log(2 * np.pi)
        self._cache_key = key

    def _log_emission(self, Z):
        self._prep()
        Z = np.atleast_2d(Z)
        out = np.empty((len(Z), self.n_states))
        for s in range(self.n_states):
            y = (Z - self.means[s]) @ self._L[s].T
            out[:, s] = self._const - 0.5 * self._logdet[s] - 0.5 * (y * y).sum(1)
        return out

    def _filter_scaled(self, Z, return_pred_ll=False, init=None):
        logB = self._log_emission(Z)
        mx = logB.max(1, keepdims=True)
        B = np.exp(logB - mx)
        A = self.transmat
        T, K = B.shape
        alpha = np.empty((T, K))
        pred_ll = np.empty(T)
        prior = self.startprob if init is None else np.asarray(init, dtype=float)
        for t in range(T):
            if t > 0:
                prior = alpha[t - 1] @ A
            j = prior * B[t]
            c = j.sum()
            if c <= 0:
                j, c = np.full(K, 1.0 / K) * B[t] + 1e-300, (np.full(K, 1.0 / K) * B[t]).sum() + 1e-300
            alpha[t] = j / c
            pred_ll[t] = np.log(c) + mx[t, 0]
        return (alpha, pred_ll) if return_pred_ll else alpha

    def filter(self, X, return_pred_ll=False, init=None):
        """Filtered probabilities p(s_t | x_1..t). No look-ahead."""
        return self._filter_scaled(self.scaler.transform(X), return_pred_ll, init)

    def step(self, prev_probs: np.ndarray, x: np.ndarray):
        """One incremental filter update for live operation. Returns (probs, pred_ll)."""
        z = self.scaler.transform(np.atleast_2d(x))
        p, ll = self._filter_scaled(z, return_pred_ll=True, init=np.asarray(prev_probs) @ self.transmat)
        return p[-1], float(ll[-1])

    def score_oos(self, X_test, X_warm=None) -> float:
        """Out-of-sample predictive log-likelihood per bar, filter warmed on X_warm."""
        if X_warm is not None and len(X_warm):
            p = self.filter(X_warm)[-1]
            init = p @ self.transmat
        else:
            init = None
        _, ll = self.filter(X_test, return_pred_ll=True, init=init)
        return float(np.mean(ll))

    # ── model selection ────────────────────────────────────────────────────
    def n_params(self, d: int) -> int:
        k = self.n_states
        cov = k * d * (d + 1) // 2 if self.covariance_type == "full" else k * d
        return k * (k - 1) + (k - 1) + k * d + cov

    def bic_aic(self, X) -> tuple[float, float]:
        _, ll = self.filter(X, return_pred_ll=True)
        L = float(ll.sum())
        p, n = self.n_params(X.shape[1]), len(X)
        return -2 * L + p * np.log(n), -2 * L + 2 * p

    def expected_durations(self) -> np.ndarray:
        return 1.0 / (1.0 - np.clip(np.diag(self.transmat), 0, 0.999999))


def predict_regime_filtered(model: RegimeModel, X: np.ndarray) -> np.ndarray:
    """THE inference entry point (CLAUDE.md rule 1). Filtered p(s_t | x_1..t) via the
    forward algorithm. Never call hmmlearn predict()/predict_proba() for decisions:
    both condition on the future (Viterbi / forward-backward)."""
    return model.filter(X)


# ── playbook router ────────────────────────────────────────────────────────

def state_playbook(states: np.ndarray, fwd_ret: np.ndarray, n_states: int,
                   bars_per_year: int = 252) -> dict:
    """Map each state to an exposure from its TRAINING-window forward returns.

    The rule is deliberately simple: long when the state's average next-bar return
    is positive, flat otherwise. A richer router (different strategies per state)
    plugs in here; the harnesses don't care what the router does.
    """
    out = {}
    for s in range(n_states):
        m = states == s
        r = fwd_ret[m]
        r = r[np.isfinite(r)]
        mu = float(r.mean()) if len(r) else 0.0
        out[s] = {"exposure": 1.0 if mu > 0 else 0.0, "ann_ret": mu * bars_per_year,
                  "ann_vol": float(r.std() * np.sqrt(bars_per_year)) if len(r) > 1 else np.nan,
                  "n": int(m.sum())}
    return out


def name_state(ann_ret: float, vol_rank: int, n_states: int) -> str:
    vol = ["low-vol", "mid-vol", "high-vol"][min(2, round(2 * vol_rank / max(1, n_states - 1)))]
    if ann_ret > 0.08:
        drift = "uptrend"
    elif ann_ret < -0.08:
        drift = "selloff" if vol != "low-vol" else "drift-down"
    else:
        drift = "chop"
    return f"{vol} {drift}"
