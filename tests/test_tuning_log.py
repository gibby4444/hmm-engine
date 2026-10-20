import numpy as np

from tuning import tuning_log as TL


def test_dsr_falls_as_trials_rise():
    kw = dict(sr_per_bar=0.04, n_bars=2500, skew=-0.3, kurt=6.0, trial_sr_var=0.0004)
    d = [TL.deflated_sharpe(n_trials=n, **kw)["dsr"] for n in (1, 5, 20, 100)]
    assert all(a > b for a, b in zip(d, d[1:]))


def test_record_counts_trials(tmp_path, monkeypatch):
    monkeypatch.setattr(TL, "LOG", tmp_path / "log.jsonl")
    for i in range(3):
        TL.record("SPY", f"c{i}", "h", "revert", "r", after={"sharpe": 0.5})
    assert [e["n"] for e in TL.entries("SPY")] == [1, 2, 3]


def test_dsr_never_assumes_zero_trial_variance(tmp_path, monkeypatch):
    """Regression: with one recorded return series the trial-variance was 0, so the
    luck benchmark collapsed to 0 and DSR reported ~99% for a system that failed."""
    import pandas as pd
    monkeypatch.setattr(TL, "LOG", tmp_path / "log.jsonl")
    r = pd.Series(np.random.default_rng(0).normal(0.0003, 0.01, 2000))
    TL.record("X", "base", "h", "baseline", "r", after={"sharpe": 0.5}, returns=r)
    for i in range(20):
        TL.record("X", f"t{i}", "h", "revert", "r")
    d = TL.current_dsr("X")
    assert d["sr0_per_bar"] > 0.01
