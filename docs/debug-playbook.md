# Debug playbook

- **test_look_ahead fails on a feature** → a rolling window is centered, something is bfilled, or an
  external series isn't lagged to its publication time. Run `data.features.lookahead_check` on that set.
- **Static inference check fails** → someone called hmmlearn `predict`/`predict_proba`/`decode`.
  Use `core.hmm_engine.predict_regime_filtered`.
- **Walk-forward suddenly slow** → the cache key changed (features, n_states, retraining config).
  Expected once per configuration; results land in `.cache/wf/`.
- **Many `drift_loglik` refits** → the z-score assumes independent bars; fat-tailed assets trip it
  often. Loosen `drift_ll_z` per profile rather than disabling drift.
- **All states map to long** → the router found positive next-bar drift in every state over the
  training window; the regime layer then degenerates to buy-and-hold. That's a finding, not a bug.
