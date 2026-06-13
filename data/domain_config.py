#!/usr/bin/env python3
"""Load domain-specific extraction patterns from YAML."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_CONFIG_PATH = _REPO_ROOT / "domain_config.yaml"

_active_path: Optional[Path] = None
_active_config: Optional[Dict[str, Any]] = None
_content_re: Optional[Pattern[str]] = None
_function_re: Optional[Pattern[str]] = None
_non_func_suffix_re: Optional[Pattern[str]] = None


def resolve_domain_config_path(path: Optional[Path] = None) -> Path:
    if path is not None:
        return Path(path)
    env = os.environ.get("SLM_DOMAIN_CONFIG", "").strip()
    if env:
        return Path(env)
    if _active_path is not None:
        return _active_path
    return DEFAULT_DOMAIN_CONFIG_PATH


def load_domain_config(path: Optional[Path] = None, *, reload: bool = False) -> Dict[str, Any]:
    global _active_path, _active_config, _content_re, _function_re, _non_func_suffix_re

    cfg_path = resolve_domain_config_path(path)
    if not reload and _active_config is not None and _active_path == cfg_path:
        return _active_config

    if not cfg_path.exists():
        data: Dict[str, Any] = {}
    else:
        if yaml is None:
            raise ImportError("pyyaml is required to load domain config (pip install pyyaml)")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    _active_path = cfg_path
    _active_config = data if isinstance(data, dict) else {}
    _content_re = None
    _function_re = None
    _non_func_suffix_re = None
    return _active_config


def _extraction(cfg: Dict[str, Any]) -> Dict[str, Any]:
    ext = cfg.get("extraction", {})
    return ext if isinstance(ext, dict) else {}


def _compile_content_regex(cfg: Dict[str, Any]) -> Pattern[str]:
    ext = _extraction(cfg)
    explicit = ext.get("content_keyword_regex")
    if explicit:
        return re.compile(str(explicit), re.IGNORECASE)
    keywords = ext.get("content_keywords") or []
    if not keywords:
        return re.compile(r"(?!)")
    joined = "|".join(re.escape(str(k)) for k in keywords)
    return re.compile(rf"\b(?:{joined})\b", re.IGNORECASE)


def get_content_regex(cfg: Optional[Dict[str, Any]] = None) -> Pattern[str]:
    global _content_re
    cfg = cfg or load_domain_config()
    if _content_re is None:
        _content_re = _compile_content_regex(cfg)
    return _content_re


def get_function_regex(cfg: Optional[Dict[str, Any]] = None) -> Optional[Pattern[str]]:
    global _function_re
    cfg = cfg or load_domain_config()
    if _function_re is None:
        pattern = _extraction(cfg).get("function_pattern")
        _function_re = re.compile(str(pattern), re.IGNORECASE) if pattern else None
    return _function_re


def get_non_function_suffix_regex(cfg: Optional[Dict[str, Any]] = None) -> Pattern[str]:
    global _non_func_suffix_re
    cfg = cfg or load_domain_config()
    if _non_func_suffix_re is None:
        suffixes = _extraction(cfg).get("non_function_suffixes") or []
        if suffixes:
            joined = "|".join(re.escape(str(s)) for s in suffixes)
            _non_func_suffix_re = re.compile(rf"(?:{joined})$", re.IGNORECASE)
        else:
            _non_func_suffix_re = re.compile(r"(?!)")
    return _non_func_suffix_re


def min_keyword_matches(cfg: Optional[Dict[str, Any]] = None) -> int:
    cfg = cfg or load_domain_config()
    return int(_extraction(cfg).get("min_keyword_matches", 2))


def has_structured_content(text: str, cfg: Optional[Dict[str, Any]] = None) -> bool:
    """True when text contains enough domain keywords to be treated as structured content."""
    if not text:
        return False
    return len(get_content_regex(cfg).findall(text)) >= min_keyword_matches(cfg)


def extract_named_pattern(text: str, cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Extract a named function/protocol from structured example text, if configured."""
    func_re = get_function_regex(cfg)
    if func_re is None or not text:
        return None
    suffix_re = get_non_function_suffix_regex(cfg)
    for match in func_re.finditer(text):
        candidate = match.group(1)
        if suffix_re.search(candidate):
            continue
        if len(candidate) <= 2:
            continue
        return candidate
    return None


def yaml_pattern_settings(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    cfg = cfg or load_domain_config()
    section = cfg.get("yaml_patterns", {})
    if not isinstance(section, dict):
        section = {}
    return {
        "alias_question": str(section.get("alias_question", "What is {alias}?")),
        "domain_label": str(section.get("domain_label", "")),
    }


def example_section_label_regexes(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, re.Pattern[str]]:
    cfg = cfg or load_domain_config()
    labels = _extraction(cfg).get("example_section_labels") or {}
    if not isinstance(labels, dict):
        labels = {}

    def _label_re(names: List[str], fallback: str) -> re.Pattern[str]:
        options = names or [fallback]
        inner = "|".join(re.escape(str(n)).replace(r"\ ", r"\s+") for n in options)
        return re.compile(rf"^\s*(?:{inner})\s*:?\s*$", re.IGNORECASE)

    return {
        "input": _label_re(list(labels.get("input") or []), "input"),
        "structured": _label_re(list(labels.get("structured") or []), "protocol"),
        "output": _label_re(list(labels.get("output") or []), "output"),
    }

