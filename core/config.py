"""Profile loading (Prompt 4). The bot resolves its whole configuration from the
symbol it trades; nothing asset-specific lives in code."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(path: Path | str = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def profile_for(symbol: str, path: Path | str = CONFIG_PATH, overrides: dict | None = None) -> dict:
    raw = load_raw(path)
    sym = symbol.upper()
    for name, prof in raw["profiles"].items():
        if sym in [s.upper() for s in prof["symbols"]]:
            merged = _deep_merge(raw.get("defaults", {}), prof)
            merged["profile_name"] = name
            merged["symbol"] = symbol
            merged["features"] = raw["feature_sets"][merged["feature_set"]]
            merged["_feature_sets"] = raw["feature_sets"]
            if overrides:
                merged = _deep_merge(merged, overrides)
                if "feature_set" in overrides:
                    merged["features"] = raw["feature_sets"][overrides["feature_set"]]
            return merged
    raise KeyError(f"No profile lists symbol {symbol!r}; add it to config/profiles.yaml")


def feature_set(name: str, path: Path | str = CONFIG_PATH) -> list[dict]:
    return load_raw(path)["feature_sets"][name]


def settings(path: Path | str = CONFIG_PATH) -> dict:
    """Top-level (non-profile) settings: hmm, tuning, logging, paper_trading."""
    raw = load_raw(path)
    return {k: v for k, v in raw.items() if k not in ("profiles", "feature_sets")}
