"""
Shared pytest fixtures for the ai_slm_training test suite.
"""
import csv
import json
import textwrap
from pathlib import Path

import pytest


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
# Canonical sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plain_text():
    return textwrap.dedent("""\
        Introduction

        This document covers analytic functions in Teradata SQL.
        Analytic functions compute values over groups of rows.

        Overview of Window Functions

        Window functions operate on a set of rows called a window.
        They are defined using the OVER clause.

        The PARTITION BY clause divides rows into partitions.
        The ORDER BY clause sorts rows within each partition.
    """)


@pytest.fixture
def sample_sql_text():
    return textwrap.dedent("""\
        CSUM Function

        The CSUM function returns a cumulative sum.

        SELECT customer_id, order_date,
               CSUM(amount, order_date) OVER (PARTITION BY customer_id ORDER BY order_date)
               AS running_total
        FROM orders;

        This query computes a running total per customer.
    """)


@pytest.fixture
def sample_qa_pairs():
    return [
        ("What is CSUM?", "CSUM returns a cumulative sum over an ordered window."),
        ("How do I use PARTITION BY?", "PARTITION BY divides rows into groups for analytic functions."),
        ("What is nPath?", "nPath is a Teradata function for path analysis on sequences of events."),
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
        name: csum
        title: "CSUM"
        description: "Computes a cumulative sum over an ordered window partition."
        category: analytics
        teradata_function: CSUM
        use_cases:
          - Running totals per group
          - Cumulative revenue analysis
        parameters:
          - name: value_expression
            type: numeric
            required: true
            description: The value to accumulate
            example: amount
          - name: ordering_column
            type: column
            required: true
            description: Column to order by within the window
            example: order_date
        templates:
          basic:
            description: "Basic cumulative sum"
            sql: |
              SELECT id, CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) AS total
              FROM t;
        best_practices: "Always specify ORDER BY to get deterministic results."
    """)


@pytest.fixture
def sample_pattern_dict():
    return {
        "name": "csum",
        "title": "CSUM",
        "description": "Computes a cumulative sum over an ordered window partition.",
        "category": "analytics",
        "teradata_function": "CSUM",
        "use_cases": ["Running totals per group", "Cumulative revenue analysis"],
        "parameters": [
            {
                "name": "value_expression",
                "type": "numeric",
                "required": True,
                "description": "The value to accumulate",
                "example": "amount",
            }
        ],
        "templates": {
            "basic": {
                "description": "Basic cumulative sum",
                "sql": "SELECT id, CSUM(amount, ts) OVER (PARTITION BY id ORDER BY ts) AS total FROM t;",
            }
        },
        "best_practices": "Always specify ORDER BY to get deterministic results.",
    }
