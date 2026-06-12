"""Tests for config.yaml and domain_config.yaml loaders."""

from pathlib import Path

import pytest

from data.domain_config import (
    extract_named_pattern,
    has_structured_content,
    load_domain_config,
    yaml_pattern_settings,
)
from train.config import get_section, load_config, resolve_path


REPO_ROOT = Path(__file__).resolve().parents[2]
SQL_DOMAIN = REPO_ROOT / "examples" / "domain_config_sql.yaml"
MEDICAL_DOMAIN = REPO_ROOT / "domain_config.yaml"
MAIN_CONFIG = REPO_ROOT / "config.yaml"


class TestMainConfig:
    def test_load_config_yaml(self):
        cfg = load_config(MAIN_CONFIG)
        assert get_section(cfg, "domain", "name") == "medical"
        assert "medical AI assistant" in get_section(cfg, "domain", "system_prompt", default="")
        assert get_section(cfg, "model", "base_model")

    def test_resolve_paths_relative_to_repo(self):
        cfg = load_config(MAIN_CONFIG)
        assert resolve_path(cfg, "paths", "training_data") == REPO_ROOT / "training_data"
        assert resolve_path(cfg, "paths", "output_model") == REPO_ROOT / "output_model"

    def test_missing_config_returns_empty_dict(self, tmp_path):
        assert load_config(tmp_path / "missing.yaml") == {}


class TestDomainConfig:
    def test_medical_keywords_detect_clinical_text(self):
        load_domain_config(MEDICAL_DOMAIN, reload=True)
        text = "The patient has hypertension and requires medication dosage review."
        assert has_structured_content(text) is True

    def test_sql_profile_detects_queries(self):
        load_domain_config(SQL_DOMAIN, reload=True)
        sql = "SELECT id FROM orders WHERE status = 'active' GROUP BY id"
        assert has_structured_content(sql) is True
        assert extract_named_pattern("FROM nPath(ON ...)") == "nPath"

    def test_yaml_pattern_settings_medical(self):
        load_domain_config(MEDICAL_DOMAIN, reload=True)
        settings = yaml_pattern_settings()
        assert "{alias}" in settings["alias_question"]
        assert settings["domain_label"] == ""

    def test_yaml_pattern_settings_sql(self):
        load_domain_config(SQL_DOMAIN, reload=True)
        settings = yaml_pattern_settings()
        assert settings["domain_label"] == "in SQL"

    def test_plain_text_not_structured_in_medical_profile(self):
        load_domain_config(MEDICAL_DOMAIN, reload=True)
        assert has_structured_content("This is a plain sentence with no clinical terms.") is False
