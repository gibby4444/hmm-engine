import numpy as np

from core.confidence_sizing import SizingConfig, confidence_multiplier, final_exposure

RISK = {"max_gross_exposure": 1.0, "max_position": 1.0}


def test_linear_mapping_anchors():
    c = SizingConfig(mode="confidence", floor_size=0.5, full_conf=0.95, threshold=0.7)
    m = confidence_multiplier(np.array([0.7, 0.95, 0.99, 0.825]), c)
    np.testing.assert_allclose(m, [0.5, 1.0, 1.0, 0.75])


def test_mapping_monotone():
    for mapping in ("linear", "stepped"):
        c = SizingConfig(mode="confidence", mapping=mapping)
        p = np.linspace(0.34, 1, 200)
        assert (np.diff(confidence_multiplier(p, c)) >= -1e-12).all()


def test_hard_mode_ignores_confidence():
    e = final_exposure(np.ones(3), np.array([0.4, 0.7, 0.99]), SizingConfig(mode="hard"), RISK)
    np.testing.assert_allclose(e, 1.0)


def test_risk_cap_outranks_confidence_multiplier():
    risk = {"max_gross_exposure": 0.6, "max_position": 1.0}
    e = final_exposure(np.array([2.0, 1.0]), np.array([0.99, 0.99]), SizingConfig(mode="both"), risk)
    assert e.max() <= 0.6 + 1e-12


def test_sizing_anchors_to_active_gate_threshold():
    """Regression: system_exposure anchored the confidence curve to the profile
    threshold even when a different gate threshold was in use."""
    import pandas as pd
    from types import SimpleNamespace
    from backtest.walk_forward import system_exposure
    from core.gating import GateConfig

    P = np.array([[0.65, 0.2, 0.15]] * 10)
    fr = pd.DataFrame(P, columns=["prob_0", "prob_1", "prob_2"], index=pd.bdate_range("2024-01-01", periods=10))
    fr["version"] = 0
    eng = SimpleNamespace(frame=lambda: fr,
                          exposure_for=lambda reg, ver, n: np.ones(len(reg)))
    inp = SimpleNamespace(px=pd.DataFrame({"close": 1.0}, index=fr.index),
                          prof={"risk": {"max_gross_exposure": 1.0, "max_position": 1.0}})
    exp, *_ = system_exposure(eng, inp, GateConfig(0.6, 1), SizingConfig(mode="both", threshold=0.8))
    # p=0.65 is above the active gate (0.6): size must be >= floor 0.5, not scaled below it
    assert exp.iloc[-1] >= 0.5
