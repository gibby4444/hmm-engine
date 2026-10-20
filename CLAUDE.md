# Trading Bot Project — Context for Claude

This file is loaded at the start of every Claude Code session. It tells Claude what this project is, how to work on it safely, and what is non-negotiable.

---

## What this project is

A Python algorithmic trading bot. Core components:

- `core/` — signal generation, regime detection, risk management
- `broker/` — broker API wrappers (Alpaca, Hyperliquid, MT5)
- `data/` — market data and feature engineering
- `backtest/` — walk-forward backtester, performance metrics, stress tests
- `monitoring/` — logging, dashboard, alerts
- `tests/` — pytest suite, including critical look-ahead bias tests

The bot trades real money (eventually). Bugs cost money. Act like it.

---

## Absolute rules — NEVER violate these

1. **NEVER use `model.predict()` from hmmlearn for live or backtest inference.** It runs Viterbi across the whole sequence, which is look-ahead bias. ALWAYS use the forward algorithm (filtered inference) that uses only data up to time t. See `core/hmm_engine.py:predict_regime_filtered`.

2. **NEVER submit an order without a stop loss.** The risk manager rejects any signal without one. Do not add bypass paths. Do not "temporarily" disable this check.

3. **NEVER hardcode API keys.** Credentials load from `.env` only. `.env` is in `.gitignore`. If you see a key in code, that is a bug — remove it immediately.

4. **NEVER default to live trading.** `paper_trading: true` is the default in `settings.yaml`. Switching to live requires explicit confirmation prompt in the code — do not remove that prompt.

5. **NEVER skip the look-ahead bias test.** `tests/test_look_ahead.py` must pass before any commit that touches HMM or feature code. If it fails, something is feeding future data into present decisions.

6. **NEVER widen stops after a position is open.** Stops can only tighten. This is enforced in `broker/order_executor.py:modify_stop`.

---

## Commands

```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # then fill in Alpaca paper keys

# Run
python main.py --dry-run              # full pipeline, no real orders
python main.py --backtest --symbols SPY --start 2020-01-01
python main.py --train-only           # retrain HMM, exit
python main.py                        # live paper trading

# Test (run these often)
pytest tests/test_look_ahead.py -v    # the one that matters most
pytest tests/test_risk.py -v
pytest tests/ -v                      # full suite
```

If you change HMM or feature code, run `pytest tests/test_look_ahead.py -v` before you consider the work done.

---

## Workflow

- Before writing code, read the relevant module and its tests. Match existing patterns.
- Small changes > large refactors. If a task is big, break it into phases and run tests between each.
- When adding a feature, add tests in the same PR.
- When fixing a bug, add a regression test that would have caught it.
- Do not touch `core/risk_manager.py` thresholds without asking me first.

---

## Project-specific conventions

- All configurable numbers live in `config/settings.yaml`. No magic numbers in code.
- All log output is structured JSON (`monitoring/logger.py`). Every entry includes: `timestamp`, `regime`, `equity`, `daily_pnl`.
- Broker adapters implement `broker/base.py:BaseBroker`. Never call Alpaca SDK directly outside `broker/alpaca_client.py`.
- Strategies inherit from `core/regime_strategies.py:BaseStrategy` and implement `generate_signal()`.

---

## When stuck

- Check `docs/debug-playbook.md` first for common issues.
- Check `tests/` for examples of how a module is expected to be used.
- If unclear about a design decision, ask before implementing. Do not guess.

---

## Python version

This project requires Python 3.10+ (for modern type hint syntax like `list[int]`, `dict[str, float]`, and `X | None` unions). If the member is on an older Python, either upgrade or adapt the type hints to the `typing` module (`List`, `Dict`, `Optional`).

---

## Imports — load these on demand, not at startup

@docs/debug-playbook.md
@docs/go-live-checklist.md
