#!/usr/bin/env python3
"""
Load repo config from YAML and merge with CLI overrides.

Config file path resolution (first match wins):
  1. Explicit ``path`` argument
  2. ``SLM_CONFIG`` environment variable
  3. ``config.yaml`` at repository root
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def repo_root() -> Path:
    return _REPO_ROOT


def resolve_config_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SLM_CONFIG", "").strip()
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML config; return empty dict if file is missing or YAML unavailable."""
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        return {}
    if yaml is None:
        raise ImportError("pyyaml is required to load config files (pip install pyyaml)")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {cfg_path}")
    return data


def get_section(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = config
    for key in keys:
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def resolve_path(config: Mapping[str, Any], *keys: str, default: str = "") -> Path:
    raw = get_section(config, *keys, default=default)
    path = Path(str(raw))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def merge_cli(config: Mapping[str, Any], args: Mapping[str, Any], mapping: Mapping[str, tuple]) -> Dict[str, Any]:
    """
    Build kwargs from config, letting explicit CLI values override.

    ``mapping`` maps output key -> (config section keys..., argparse dest name).
    """
    merged: Dict[str, Any] = {}
    for out_key, spec in mapping.items():
        *cfg_keys, arg_name = spec
        value = get_section(config, *cfg_keys) if cfg_keys else None
        if arg_name in args and args[arg_name] is not None:
            cli_val = args[arg_name]
            if cli_val != "" and cli_val != Path("."):
                value = cli_val
        if value is not None:
            merged[out_key] = value
    return merged
