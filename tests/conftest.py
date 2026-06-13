"""
Shared pytest fixtures for the SLM Domain Foundry test suite.
"""
import csv
import json
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOMAIN_CONFIG = REPO_ROOT / "domain_config.yaml"


@pytest.fixture(autouse=True)
def _load_default_domain_config():
    """Unit tests use the default medical domain extraction profile."""
    if not DEFAULT_DOMAIN_CONFIG.exists():
        return
    from data.domain_config import load_domain_config

    load_domain_config(DEFAULT_DOMAIN_CONFIG, reload=True)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path):
    """Return a factory that writes a CSV and returns its path."""
    def _make(rows: list[dict], filename: str = "test.csv") -> Path:
        if not rows:
            p = tmp_path / filename
            p.write_text("", encoding="utf-8")
            return p
        p = tmp_path / filename
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return p
    return _make


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Return a factory that writes a JSONL file and returns its path."""
    def _make(items: list, filename: str = "test.jsonl") -> Path:
        p = tmp_path / filename
        with open(p, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return p
    return _make


@pytest.fixture
def tmp_yaml(tmp_path):
    """Return a factory that writes a YAML file and returns its path."""
    def _make(content: str, filename: str = "pattern.yaml") -> Path:
        p = tmp_path / filename
        p.write_text(content, encoding="utf-8")
        return p
    return _make


# ---------------------------------------------------------------------------
# Canonical sample data (medical domain)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plain_text():
    return textwrap.dedent("""\
        Introduction

        This document covers hypertension management in primary care.
        Blood pressure targets depend on comorbidities and patient age.

        Overview of Lifestyle Modification

        Sodium reduction and regular exercise are first-line interventions.
        Shared decision-making improves adherence to treatment plans.

        The target blood pressure for most adults is below 130/80 mmHg.
        Reassessment should occur within four weeks of starting therapy.
    """)


@pytest.fixture
def sample_structured_text():
    return textwrap.dedent("""\
        Hypertension Protocol

        Case presentation with elevated blood pressure readings in a patient
        with diabetes requires medication review and lifestyle counseling.

        Treatment plan

        Initiate an ACE inhibitor or thiazide-type diuretic based on formulary.
        Recommend sodium reduction, exercise, and home blood pressure monitoring.

        Outcome

        Reassess blood pressure in four weeks and adjust therapy if targets are not met.
    """)


@pytest.fixture
def sample_qa_pairs():
    return [
        ("What is hypertension?", "Hypertension is chronic elevation of blood pressure."),
        ("What is the target blood pressure?", "Most adults aim for below 130/80 mmHg."),
        ("When is aspirin used?", "Low-dose aspirin is used for secondary cardiovascular prevention in selected patients."),
    ]


@pytest.fixture
def sample_sharegpt_examples(sample_qa_pairs):
    return [
        {
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }
        for q, a in sample_qa_pairs
    ]


@pytest.fixture
def sample_alpaca_examples(sample_qa_pairs):
    return [
        {"instruction": q, "input": "", "output": a}
        for q, a in sample_qa_pairs
    ]


@pytest.fixture
def sample_yaml_pattern():
    """Minimal valid YAML pattern string."""
    return textwrap.dedent("""\
        name: hypertension
        title: "Hypertension Management"
        description: "Hypertension is chronic elevation of blood pressure that increases cardiovascular risk."
        category: cardiology
        pattern_alias: HTN
        use_cases:
          - Primary prevention in adults with elevated blood pressure
          - Secondary prevention after stroke or myocardial infarction
        parameters:
          - name: target_systolic
            type: integer
            required: true
            description: Target systolic blood pressure in mmHg
            example: "130"
        templates:
          basic:
            description: "Initial lifestyle and medication plan"
            content: |
              Case: repeated office readings above target.
              Treatment plan: lifestyle counseling plus first-line antihypertensive therapy.
              Outcome: reassess in four weeks.
        best_practices: "Confirm elevated readings before starting long-term therapy."
    """)


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """True when Ollama responds at http://localhost:11434."""
    try:
        import requests

        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.ok
    except Exception:
        return False


@pytest.fixture
def sample_pattern_dict():
    return {
        "name": "hypertension",
        "title": "Hypertension Management",
        "description": "Hypertension is chronic elevation of blood pressure that increases cardiovascular risk.",
        "category": "cardiology",
        "pattern_alias": "HTN",
        "use_cases": ["Primary prevention in adults with elevated blood pressure"],
        "parameters": [
            {
                "name": "target_systolic",
                "type": "integer",
                "required": True,
                "description": "Target systolic blood pressure in mmHg",
                "example": "130",
            }
        ],
        "templates": {
            "basic": {
                "description": "Initial lifestyle and medication plan",
                "content": "Case: elevated readings. Treatment plan: lifestyle plus first-line therapy.",
            }
        },
        "best_practices": "Confirm elevated readings before starting long-term therapy.",
    }
