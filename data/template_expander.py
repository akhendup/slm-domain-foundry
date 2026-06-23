#!/usr/bin/env python3
"""
template_expander.py — Combinatorial Q&A generation from vocabulary YAML files.

Loads structured vocabulary YAML files (medical_vocabulary.yaml, financial_vocabulary.yaml)
and expands each entry against all matching question templates from question_templates.yaml,
producing a large set of Q&A pairs without manually writing each question.

Scale:
  - Medical vocabulary × templates per topic
  - Financial vocabulary (~30 topics) × ~50 templates per topic ≈ 1,500+ pairs

Usage:
    from data.template_expander import expand_vocab_dir

    qa_pairs = expand_vocab_dir(Path("data/"))           # returns List[Dict]
    multiturn = expand_vocab_dir(Path("data/"), multiturn=True)  # returns ShareGPT JSONL-ready dicts

Public API:
    expand_vocab_dir(vocab_dir, multiturn=False) -> List[Dict]
    VocabularyExpander  — class for testing / per-file expansion
"""

import logging
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import question template lists
# ---------------------------------------------------------------------------

from data.question_templates import (
    DESCRIPTION_QUESTIONS,
    ONE_SENTENCE_QUESTIONS,
    CATEGORY_QUESTIONS,
    USE_CASE_QUESTIONS,
    PARAMETER_QUESTIONS,
    SYNTAX_QUESTIONS,
    ARGUMENT_QUESTIONS,
    NOTES_QUESTIONS,
    EXAMPLE_QUESTIONS,
    NULL_BEHAVIOR_QUESTIONS,
    PERFORMANCE_QUESTIONS,
    ERROR_QUESTIONS,
    COMPARISON_QUESTIONS,
    GENERAL_OVERVIEW_QUESTIONS,
    GENERAL_KEYCONCEPT_QUESTIONS,
    GENERAL_DETAIL_QUESTIONS,
    GENERAL_COMPARISON_QUESTIONS,
    GENERAL_APPLICATION_QUESTIONS,
    GENERAL_CAUSES_QUESTIONS,
    GENERAL_REQUIREMENTS_QUESTIONS,
    FINANCIAL_TRANSACTION_QUESTIONS,
    FINANCIAL_ACCOUNT_QUESTIONS,
    FINANCIAL_ANALYSIS_QUESTIONS,
)

# ---------------------------------------------------------------------------
# Helper: detect {placeholder} names in a template string
# ---------------------------------------------------------------------------

def _get_placeholders(template: str) -> List[str]:
    """Return the list of unique placeholder names in a format string.

    Example:
        _get_placeholders("How does {fn} differ from {related}?")
        → ["fn", "related"]
    """
    formatter = string.Formatter()
    return [
        field_name
        for _, field_name, _, _ in formatter.parse(template)
        if field_name is not None
    ]


# ---------------------------------------------------------------------------
# YAML loading helper
# ---------------------------------------------------------------------------

def _load_vocabulary(path: Path) -> Optional[Dict]:
    """Load a vocabulary YAML file and return its dict, or None on failure."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            _log.warning("Vocabulary file %s did not load as a dict", path)
            return None
        return data
    except Exception as exc:
        _log.warning("Could not load vocabulary file %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# VocabularyExpander
# ---------------------------------------------------------------------------

class VocabularyExpander:
    """Expand a vocabulary YAML file into Q&A pairs using question templates.

    The vocabulary YAML should have this structure at the top level:
        metadata:    (optional) dict with domain, version
        <section>:   list of entry dicts

    Each entry dict is expected to have:
        name          — string: the topic name, used as {fn} placeholder
        description   — string: multi-sentence explanation
        one_sentence  — string: brief one-liner
        category      — string: e.g. "window_function", "fee_type"
        syntax        — string or dict: structured clinical syntax or definition
        null_behavior — string: NULL handling notes
        analysis_notes / performance_tips — string or list
        common_errors — list of {error, cause, solution}
        examples      — list of {sql/scenario, description}
        related       — list of related topic names (drives comparison pairs)

    Not all fields are required — absent fields simply skip the corresponding
    template group.
    """

    # Map from entry field names to (question_templates, answer_field)
    # These are the "single-param" expansions where {fn} = entry["name"]
    _FIELD_TEMPLATE_MAP: List[tuple] = [
        # (answer_field,          question_list,         answer_formatter)
        ("description",           DESCRIPTION_QUESTIONS,   None),
        ("one_sentence",          ONE_SENTENCE_QUESTIONS,  None),
        ("category",              CATEGORY_QUESTIONS,      None),
        ("description",           USE_CASE_QUESTIONS,      None),
        ("syntax",                SYNTAX_QUESTIONS,        None),
        ("syntax",                ARGUMENT_QUESTIONS,      None),
        ("description",           NOTES_QUESTIONS,         None),
        ("null_behavior",         NULL_BEHAVIOR_QUESTIONS, None),
        ("performance_tips",      PERFORMANCE_QUESTIONS,   None),
    ]

    def __init__(self, vocab_data: Dict):
        self._data = vocab_data
        self._domain = vocab_data.get("metadata", {}).get("domain", "general")

    def _collect_entries(self) -> List[Dict]:
        """Flatten all section lists into a single list of entry dicts."""
        entries = []
        for key, value in self._data.items():
            if key == "metadata":
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "name" in item:
                        entries.append(item)
            elif isinstance(value, dict):
                # Nested: {section_name: {subsection: [entries]}}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, list):
                        for item in sub_value:
                            if isinstance(item, dict) and "name" in item:
                                entries.append(item)
        return entries

    def _answer_for_field(self, entry: Dict, field: str) -> Optional[str]:
        """Extract a string answer from an entry field, handling list and dict types."""
        val = entry.get(field)
        if val is None:
            return None
        if isinstance(val, str):
            return val.strip() if val.strip() else None
        if isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    # e.g., {error: ..., cause: ..., solution: ...}
                    parts.append("; ".join(f"{k}: {v}" for k, v in item.items() if isinstance(v, str)))
            return "\n".join(parts) if parts else None
        if isinstance(val, dict):
            # Flatten to key: value lines
            return "\n".join(f"{k}: {v}" for k, v in val.items() if isinstance(v, str))
        return str(val)

    def _expand_single(self, entry: Dict) -> List[Dict]:
        """Generate Q&A pairs from single-placeholder templates ({fn} only)."""
        name = entry.get("name", "")
        if not name:
            return []

        pairs = []

        for answer_field, question_list, _ in self._FIELD_TEMPLATE_MAP:
            answer = self._answer_for_field(entry, answer_field)
            if not answer:
                continue
            for template in question_list:
                placeholders = _get_placeholders(template)
                if placeholders == ["fn"]:
                    question = template.format(fn=name)
                    pairs.append({"question": question, "answer": answer, "source": name})

        # Examples — use each example's description as the answer
        examples = entry.get("examples") or []
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            ex_desc = ex.get("description") or ex.get("content") or ""
            if not ex_desc:
                continue
            ex_desc = ex_desc.strip()
            for template in EXAMPLE_QUESTIONS:
                placeholders = _get_placeholders(template)
                if placeholders == ["fn"]:
                    question = template.format(fn=name)
                    pairs.append({"question": question, "answer": ex_desc, "source": name})
            # Only use first example to avoid massive repetition
            break

        # Common errors — aggregate into a single answer block
        errors = entry.get("common_errors") or []
        if errors:
            error_text = self._answer_for_field(entry, "common_errors")
            if error_text:
                for template in ERROR_QUESTIONS:
                    placeholders = _get_placeholders(template)
                    if placeholders == ["fn"]:
                        question = template.format(fn=name)
                        pairs.append({"question": question, "answer": error_text, "source": name})

        # Domain-specific templates
        if self._domain == "financial":
            category = entry.get("category", "")
            desc = self._answer_for_field(entry, "description") or ""
            if category == "transaction_type":
                for template in FINANCIAL_TRANSACTION_QUESTIONS:
                    placeholders = _get_placeholders(template)
                    if placeholders == ["fn"] and desc:
                        question = template.format(fn=name)
                        pairs.append({"question": question, "answer": desc, "source": name})
            elif category == "balance_concept":
                for template in FINANCIAL_ACCOUNT_QUESTIONS:
                    placeholders = _get_placeholders(template)
                    if placeholders == ["fn"] and desc:
                        question = template.format(fn=name)
                        pairs.append({"question": question, "answer": desc, "source": name})
            elif category == "analysis_pattern":
                for template in FINANCIAL_ANALYSIS_QUESTIONS:
                    placeholders = _get_placeholders(template)
                    if placeholders == ["fn"] and desc:
                        question = template.format(fn=name)
                        pairs.append({"question": question, "answer": desc, "source": name})

        # General overview templates (always add for non-technical domains)
        if self._domain != "sql":
            desc = self._answer_for_field(entry, "description") or ""
            for template in GENERAL_OVERVIEW_QUESTIONS:
                placeholders = _get_placeholders(template)
                if placeholders == ["fn"] and desc:
                    question = template.format(fn=name)
                    pairs.append({"question": question, "answer": desc, "source": name})
            notes = self._answer_for_field(entry, "analysis_notes") or ""
            if notes:
                for template in GENERAL_DETAIL_QUESTIONS:
                    placeholders = _get_placeholders(template)
                    if placeholders == ["fn"]:
                        question = template.format(fn=name)
                        pairs.append({"question": question, "answer": notes, "source": name})

        return pairs

    def _expand_comparison(self, entry: Dict) -> List[Dict]:
        """Generate comparison Q&A pairs for each related topic."""
        name = entry.get("name", "")
        related_list = entry.get("related") or []
        if not name or not related_list:
            return []

        desc = self._answer_for_field(entry, "description") or ""
        if not desc:
            return []

        pairs = []
        templates = COMPARISON_QUESTIONS

        for related_name in related_list:
            if not isinstance(related_name, str) or not related_name.strip():
                continue
            related_name = related_name.strip()
            for template in templates:
                placeholders = _get_placeholders(template)
                try:
                    if set(placeholders) == {"fn", "related"}:
                        question = template.format(fn=name, related=related_name)
                        answer = (
                            f"{name}: {desc}\n\n"
                            f"Compare with {related_name} for differences in behavior and use cases."
                        )
                        pairs.append({"question": question, "answer": answer, "source": name})
                    elif placeholders == ["fn"]:
                        question = template.format(fn=name)
                        answer = f"{name}: {desc}"
                        pairs.append({"question": question, "answer": answer, "source": name})
                except KeyError:
                    pass  # Template has unexpected placeholder; skip

        return pairs

    def expand(self) -> List[Dict]:
        """Expand all entries into a flat list of Q&A pair dicts.

        Each dict has keys: question, answer, source
        """
        entries = self._collect_entries()
        all_pairs: List[Dict] = []

        for entry in entries:
            all_pairs.extend(self._expand_single(entry))
            all_pairs.extend(self._expand_comparison(entry))

        _log.info(
            "VocabularyExpander[%s]: %d entries → %d Q&A pairs",
            self._domain,
            len(entries),
            len(all_pairs),
        )
        return all_pairs

    def expand_to_multiturn(self) -> List[Dict]:
        """Wrap each Q&A pair as a ShareGPT-format multi-turn conversation dict.

        Returns a list of dicts:
            {
                "conversations": [
                    {"from": "human", "value": "<question>"},
                    {"from": "gpt",   "value": "<answer>"},
                ]
            }
        """
        pairs = self.expand()
        return [
            {
                "conversations": [
                    {"from": "human", "value": p["question"]},
                    {"from": "gpt",   "value": p["answer"]},
                ]
            }
            for p in pairs
        ]


# ---------------------------------------------------------------------------
# Module-level public API
# ---------------------------------------------------------------------------

def expand_vocab_dir(
    vocab_dir: Path,
    multiturn: bool = False,
) -> List[Dict]:
    """Expand vocabulary YAML file(s) under *vocab_dir* or a single vocabulary file.

    Looks for files matching '*_vocabulary.yaml' in a directory, or accepts one
    such file path directly (e.g. ``data/medical_vocabulary.yaml``).
    Each file is loaded and expanded via VocabularyExpander.

    Args:
        vocab_dir:  Directory or single *_vocabulary.yaml file.
        multiturn:  If True, return ShareGPT-format dicts.
                    If False, return flat {question, answer, source} dicts.

    Returns:
        Combined list of Q&A pairs or multi-turn conversation dicts.
    """
    vocab_path = Path(vocab_dir)
    all_results: List[Dict] = []

    if vocab_path.is_file():
        vocab_files = [vocab_path]
    else:
        vocab_files = sorted(vocab_path.glob("*_vocabulary.yaml"))
    if not vocab_files:
        _log.warning("No *_vocabulary.yaml files found at %s", vocab_path)
        return all_results

    for vf in vocab_files:
        data = _load_vocabulary(vf)
        if data is None:
            continue
        expander = VocabularyExpander(data)
        if multiturn:
            results = expander.expand_to_multiturn()
        else:
            results = expander.expand()
        _log.info("expand_vocab_dir: %s → %d records", vf.name, len(results))
        all_results.extend(results)

    _log.info(
        "expand_vocab_dir: %d vocabulary files → %d total records",
        len(vocab_files),
        len(all_results),
    )
    return all_results
