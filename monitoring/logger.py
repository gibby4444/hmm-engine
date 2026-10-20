"""Structured JSON logging (CLAUDE.md convention). Every entry carries
timestamp, regime, equity, daily_pnl — null when not applicable to the event."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings

_ROOT = Path(__file__).resolve().parents[1]


def log_event(event: str, regime=None, equity=None, daily_pnl=None, to_stdout: bool = False, **fields):
    cfg = settings()["logging"]
    d = _ROOT / cfg["dir"]
    d.mkdir(exist_ok=True)
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
           "regime": regime, "equity": equity, "daily_pnl": daily_pnl, **fields}
    line = json.dumps(rec, default=str)
    with open(d / cfg["file"], "a") as f:
        f.write(line + "\n")
    if to_stdout:
        print(line, file=sys.stdout)
    return rec
