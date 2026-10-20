import numpy as np

from core.gating import NEUTRAL, GateConfig, apply_gates, ungated


def probs(seq, conf=0.9, K=3):
    P = np.full((len(seq), K), (1 - conf) / (K - 1))
    P[np.arange(len(seq)), seq] = conf
    return P


def test_persistence_requires_n_consecutive_bars():
    seq = [0] * 5 + [1, 1] + [0] * 3 + [1] * 3 + [1] * 2
    g = apply_gates(probs(seq), GateConfig(0.7, 3))
    assert list(g.regime[:10]) == [0] * 10          # 2-bar blip absorbed
    assert g.regime[12] == 1 and g.regime[11] == 0  # third consecutive bar confirms


def test_confidence_threshold_blocks_weak_switch():
    P = probs([0] * 5 + [1] * 6, conf=0.6)
    g = apply_gates(P, GateConfig(0.7, 2))
    assert (g.regime == 0).all()


def test_neutral_stance_when_nothing_convincing():
    P = probs([0] * 4 + [1] * 4, conf=0.5)
    P[:4] = probs([0] * 4, conf=0.9)
    g = apply_gates(P, GateConfig(0.7, 3, neutral_on_low_confidence=True))
    assert g.regime[-1] == NEUTRAL


def test_ungated_follows_argmax_and_log_counts():
    seq = [0, 1, 0, 1, 2, 2]
    g = ungated(probs(seq))
    assert list(g.regime) == seq
    assert g.raw_flickers == 4 and g.confirmed_switches == 4
    gg = apply_gates(probs(seq), GateConfig(0.7, 3))
    assert gg.confirmed_switches < gg.raw_flickers
    assert set(gg.log["type"]) <= {"raw_flicker", "confirmed_switch"}
