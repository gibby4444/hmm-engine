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
