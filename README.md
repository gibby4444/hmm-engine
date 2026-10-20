# hmm-engine — Advanced HMM tuning toolkit

Implements all 12 prompts of the *Advanced HMM Tuning: Build Library* (Track B) on top of a
minimal regime bot, following the rules in `CLAUDE.md`.

The loop: **inspect (1) → change one dial → validate (9) → log it (12) → keep only what survives.**

## Quick start

```bash
pip install -r requirements.txt
pytest tests/test_look_ahead.py -v     # the one that matters most
pytest tests/ -v                       # full suite (~4 min; the walk-forward parity tests refit HMMs)
python main.py tune --symbol SPY       # runs every harness, writes reports/*.html
python main.py tune --symbol BTC-USD
```

First run downloads daily data (yfinance) into `.cache/` and fits the walk-forward models
(~5 min SPY, ~9 min BTC on 2 cores). Afterwards every harness reuses the cached walk-forward,
so changing a gate, sizing or breaker dial is instant; changing features, state count or
retraining invalidates the cache key automatically.

## Map: prompt → module

| # | Prompt | Module |
|---|---|---|
| 1 | State inspection report | `tuning/state_report.py` |
| 2 | State-count harness (BIC/AIC, seed stability, checklist — never auto-selects) | `tuning/state_count.py` |
| 3 | Feature lab (A/B, hygiene, look-ahead + correlation checks) | `tuning/feature_lab.py`, `data/features.py` |
| 4 | Per-asset profiles; back-adjusted futures + roll test | `config/settings.yaml`, `core/config.py`, `data/market_data.py:back_adjust`, `tests/test_futures_roll.py` |
| 5 | Gating stack + on/off replay | `core/gating.py`, `tuning/gating_eval.py` |
| 6 | Confidence-scaled sizing (risk caps always win) | `core/confidence_sizing.py`, `core/risk_manager.py` |
| 7 | External vol: feature A/B + circuit breaker sweep, fail-safe | `core/vol_integration.py`, `tuning/ext_vol.py` |
| 8 | Retraining scheduler, drift tripwire, model versioning, live/backtest parity | `core/retraining.py`, `tests/test_retraining.py` |
| 9 | Regime validation harness (HMM vs MA filter vs best single strategy) | `tuning/regime_validation.py` |
| 10 | Detection lag + lag/false-alarm trade-off | `tuning/detection_lag.py` |
| 11 | Detector bake-off (GMM, k-means, MA floor) | `tuning/detector_bakeoff.py` |
| 12 | Tuning log + deflated Sharpe | `tuning/tuning_log.py` |

## Design decisions worth knowing

- **Filtered inference only.** `core/hmm_engine.py:predict_regime_filtered` runs an explicit
  forward recursion. hmmlearn's `predict` (Viterbi) and `predict_proba` (forward-backward) both
  condition on future bars; a static test fails the suite if either appears in decision code.
- **Canonical state labels.** After every refit states are re-ordered by in-state realised vol
  (0 = calmest), so a refit that merely permutes labels doesn't look like a regime change.
- **Router.** Each state's exposure comes from its training-window *next-bar* return
  (long if positive, flat if not). Deliberately simple; swap in your `BaseStrategy` router at
  `core/hmm_engine.py:state_playbook` / `core/retraining.py:exposure_for`.
- **One retraining path.** `RetrainEngine.on_bar` is used by both walk-forward and live
  (`LiveRunner`); `tests/test_retraining.py` proves they produce identical refits, probabilities
  and playbooks bar for bar.
- **Comparable feature A/B.** HMM log-likelihood isn't comparable across feature sets (different
  dimensions/units), so the feature lab's headline metric is the out-of-sample predictive log
  score of next-bar *returns*, which is.
- **VIX timing.** VIX settles after the equity close, so bar *t* uses VIX(*t−1*).
- **Leak detector.** `lookahead_check` recomputes features on history truncated at *t* and
  requires an exact match — including NaN-vs-value, which a naive diff misses (there's a
  regression test that plants a leak).

## Integrating with an existing bot

If your repo already has `core/risk_manager.py`, keep yours and point `cap_exposure` at it.
The toolkit places no orders and touches no broker code; `paper_trading: true` is asserted at startup.
