"""Configuration loading: config/default.yaml, overridden by config/local.yaml."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text())
    local = ROOT / "config" / "local.yaml"
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text()) or {})
    if path:
        cfg = _merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
