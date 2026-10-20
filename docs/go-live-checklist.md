# Go-live checklist (regime layer)

- [ ] `pytest tests/ -v` green, including `test_look_ahead.py`
- [ ] Regime validation verdict = PASS (beats MA filter AND best single strategy out-of-sample)
- [ ] Deflated Sharpe ≥ 95% given every trial in the tuning log
- [ ] Detection lag known; sizing budgets for the median adverse move
- [ ] Paper traded ≥ 3 months with logs reconciled to the backtest
- [ ] Live switch requires the explicit confirmation prompt; `paper_trading: true` until then
