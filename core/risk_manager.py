"""Exposure caps used by the tuning layer.

If your bot already has core/risk_manager.py, keep YOURS and delete this file:
`cap_exposure` is the only function the tuning layer calls, so wire it to your
existing limits. Thresholds come from config/settings.yaml (risk block); per
CLAUDE.md they are not to be changed without the owner's sign-off.
"""
from __future__ import annotations

import numpy as np


def cap_exposure(exposure, risk: dict) -> np.ndarray:
    cap = min(risk["max_gross_exposure"], risk["max_position"])
    return np.clip(np.asarray(exposure, dtype=float), -cap, cap)
