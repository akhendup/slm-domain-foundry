#!/usr/bin/env python3
"""
Knowledge capture from non-technical users via guided Q&A forms.

A user who understands their domain but not YAML fills in plain-English fields.
This module converts those answers into YAML patterns, generates training Q&A pairs,
and maintains a persistent library that grows over time (continuous learning).

Library directory layout:
  library/
    <slug>.yaml     — one file per captured concept/function
    _index.json     — metadata index (name, created, last_updated, qa_count)

The library is automatically included whenever training data is prepared.
Users can add partial knowledge and fill in more fields later — nothing requires
all fields to be present at once.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from data.yaml_pattern_loader import generate_qa_from_pattern, generate_multiturn_from_pattern


# ---------------------------------------------------------------------------
# Field definitions — plain-English labels shown to the user in forms
# ---------------------------------------------------------------------------

FIELD_DEFS = [
    {
        "key": "title",
        "label": "Name / Title",
        "help": "What is this function or feature called? (e.g. 'nPath', 'CSUM', 'Sessionize')",
        "required": True,
        "type": "text",
    },
    {
        "key": "description",
        "label": "What does it do?",
        "help": (
            "Explain in plain English what this does and why someone would use it. "
            "You can mention what data it works on, what result it produces, "
            "and any key concepts. A few sentences to a paragraph is ideal."
        ),
        "required": True,
        "type": "textarea",
    },
    {
        "key": "use_cases_text",
        "label": "When would you use this?",
        "help": (
            "List the situations where this is useful. One use case per line. "
            "E.g.:\n  User journey analysis\n  Funnel analysis\n  Churn prediction"
        ),
        "required": False,
        "type": "textarea",
    },
    {
        "key": "parameters_text",
        "label": "What are the inputs / parameters?",
        "help": (
            "Describe each input or parameter. One per line, in the format:\n"
            "  name: description (example value)\n"
            "E.g.:\n"
            "  partition_columns: Columns to group events by, e.g. user_id (user_id)\n"
            "  order_columns: Columns to sort events by, e.g. timestamp (event_ts)\n"
            "  pattern: Regex-like sequence to match, e.g. A.B+.C (A.B.C)"
        ),
        "required": False,
        "type": "textarea",
    },
    {
        "key": "sql_example",
        "label": "SQL example",
        "help": (
            "Paste a representative SQL query. Don't worry about placeholders — "
            "real concrete values are fine and actually better for training."
        ),
        "required": False,
        "type": "code",
    },
    {
        "key": "sql_description",
        "label": "What does this SQL example do?",
        "help": "One sentence describing what the SQL query above demonstrates.",
        "required": False,
        "type": "text",
    },
    {
        "key": "example_output",
        "label": "What does the result look like?",
        "help": (
            "Paste sample output rows, or describe the shape of the result. "
            "E.g.:\n  path                          | count\n  login|browse|purchase         | 1234\n  login|browse                  | 876"
        ),
        "required": False,
        "type": "textarea",
    },
    {
        "key": "common_errors_text",
        "label": "Common errors or gotchas",
        "help": (
            "What mistakes do people make? What are the tricky parts? "
            "Format as:\n  Problem: solution\n"
            "E.g.:\n  Spaces in pattern string: Remove all spaces from PATTERN, use A.B.C not 'A B C'\n"
            "  Reserved keyword as symbol: Avoid Exit/Count — use Exited/Counted instead"
        ),
        "required": False,
        "type": "textarea",
    },
    {
        "key": "best_practices",
        "label": "Tips and best practices",
        "help": (
            "Any advice you'd give to someone using this for the first time. "
            "Free text — write as much or as little as you know."
        ),
        "required": False,
        "type": "textarea",
    },
    {
        "key": "category",
        "label": "Category (optional)",
        "help": "E.g. analytics, data_quality, ml, timeseries, text",
        "required": False,
        "type": "text",
    },
]


# ---------------------------------------------------------------------------
# Parsing helpers for free-text fields
# ---------------------------------------------------------------------------

def _parse_use_cases(text: str) -> List[str]:
    """Parse one-per-line use cases text into a list."""
    lines = [ln.strip().lstrip("-•*").strip() for ln in text.strip().splitlines()]
    return [ln for ln in lines if len(ln) > 3]


def _parse_parameters(text: str) -> List[Dict[str, str]]:
    """
    Parse 'name: description (example)' format into parameter dicts.
    Forgiving parser — handles various formats users might use.
    """
    params = []
    for line in text.strip().splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        # Try 'name: description (example)'
        m = re.match(r"^(\w[\w\s_\-]*?)\s*:\s*(.+)$", line)
        if m:
            pname = m.group(1).strip().replace(" ", "_")
            rest = m.group(2).strip()
            # Extract (example) from end
            ex_m = re.search(r"\(([^)]+)\)\s*$", rest)
            example = ex_m.group(1) if ex_m else ""
            desc = rest[: ex_m.start()].strip() if ex_m else rest
            params.append({
                "name": pname,
                "type": "string",
                "required": True,
                "description": desc,
                "example": example,
            })
    return params


def _parse_errors(text: str) -> List[Dict[str, str]]:
    """
    Parse 'Problem: solution' or 'Error — solution' lines into error dicts.
    """
    errors = []
    for line in text.strip().splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        # Try 'Problem: solution'
        for sep in (":", "—", "-", "→"):
            if sep in line:
                parts = line.split(sep, 1)
                error = parts[0].strip()
                solution = parts[1].strip()
                if len(error) > 3 and len(solution) > 3:
                    errors.append({
                        "error": error,
                        "cause": "",
                        "solution": solution,
                    })
                    break
    return errors


# ---------------------------------------------------------------------------
# Form data → YAML pattern dict
# ---------------------------------------------------------------------------

def form_to_pattern(form_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Convert flat form field values (all strings) into a structured pattern dict
    matching the YAML pattern format used by yaml_pattern_loader.
    """
    title = form_data.get("title", "").strip()
    name = re.sub(r"[^\w]", "_", title).lower().strip("_")

    pattern: Dict[str, Any] = {
        "name": name,
        "title": title,
        "description": form_data.get("description", "").strip(),
        "category": form_data.get("category", "general").strip() or "general",
        "difficulty": "beginner",
        "_source": "user_library",
        "_captured_at": datetime.now(timezone.utc).isoformat(),
    }

    # Use cases
    uc_text = form_data.get("use_cases_text", "").strip()
    if uc_text:
        pattern["use_cases"] = _parse_use_cases(uc_text)

    # Parameters
    params_text = form_data.get("parameters_text", "").strip()
    if params_text:
        parsed_params = _parse_parameters(params_text)
        if parsed_params:
            pattern["parameters"] = parsed_params

    # SQL template
    sql = form_data.get("sql_example", "").strip()
    sql_desc = form_data.get("sql_description", "").strip() or "Example SQL query"
    if sql:
        pattern["templates"] = {
            "example": {
                "description": sql_desc,
                "sql": sql,
            }
        }

    # Example with output
    output = form_data.get("example_output", "").strip()
    if sql and output:
        pattern["examples"] = [
            {
                "name": sql_desc,
                "description": sql_desc,
                "expected_result": output,
            }
        ]

    # Common errors
    errors_text = form_data.get("common_errors_text", "").strip()
    if errors_text:
        parsed_errors = _parse_errors(errors_text)
        if parsed_errors:
            pattern["common_errors"] = parsed_errors

    # Best practices
    bp = form_data.get("best_practices", "").strip()
    if bp:
        pattern["best_practices"] = bp

    return pattern


# ---------------------------------------------------------------------------
# Library management
# ---------------------------------------------------------------------------

def _library_index_path(library_dir: Path) -> Path:
    return library_dir / "_index.json"


def _load_index(library_dir: Path) -> Dict[str, Any]:
    index_path = _library_index_path(library_dir)
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"entries": {}}


def _save_index(library_dir: Path, index: Dict[str, Any]) -> None:
    _library_index_path(library_dir).write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_to_library(
    pattern: Dict[str, Any], library_dir: Path
) -> Tuple[Path, int]:
    """
    Save a pattern dict to the library as a YAML file.
    Returns (saved_path, qa_count).
    Updates the library index.
    Raises ImportError if pyyaml is not installed.
    """
    if not _YAML_AVAILABLE:
        raise ImportError("pyyaml is required: pip install pyyaml")

    library_dir.mkdir(parents=True, exist_ok=True)
    name = pattern.get("name") or "entry"
    slug = re.sub(r"[^\w\-]", "_", name).lower().strip("_")
    yaml_path = library_dir / f"{slug}.yaml"

    # Generate Q&A to count them
    qa_pairs = generate_qa_from_pattern(pattern)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(pattern, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    # Update index
    index = _load_index(library_dir)
    index["entries"][slug] = {
        "title": pattern.get("title", name),
        "category": pattern.get("category", ""),
        "created": pattern.get("_captured_at", datetime.now(timezone.utc).isoformat()),
        "qa_count": len(qa_pairs),
        "file": yaml_path.name,
    }
    _save_index(library_dir, index)

    return yaml_path, len(qa_pairs)


def load_library_entries(library_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all YAML pattern files from the library directory.
    Returns list of pattern dicts.
    """
    if not library_dir.exists():
        return []
    entries = []
    for yaml_path in sorted(library_dir.glob("*.yaml")):
        if yaml_path.name.startswith("_"):
            continue
        if not _YAML_AVAILABLE:
            continue
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("name"):
                data["_source_file"] = str(yaml_path)
                entries.append(data)
        except Exception:
            pass
    return entries


def library_stats(library_dir: Path) -> Dict[str, Any]:
    """Return summary stats for the library."""
    index = _load_index(library_dir)
    entries = index.get("entries", {})
    total_qa = sum(e.get("qa_count", 0) for e in entries.values())
    by_category: Dict[str, int] = {}
    for e in entries.values():
        cat = e.get("category") or "general"
        by_category[cat] = by_category.get(cat, 0) + 1
    return {
        "total_patterns": len(entries),
        "total_qa_pairs": total_qa,
        "by_category": by_category,
        "entries": [
            {"slug": slug, **meta}
            for slug, meta in sorted(entries.items(), key=lambda x: x[1].get("created", ""), reverse=True)
        ],
    }


def delete_from_library(slug: str, library_dir: Path) -> bool:
    """Delete a library entry by slug. Returns True if deleted."""
    yaml_path = library_dir / f"{slug}.yaml"
    deleted = False
    if yaml_path.exists():
        yaml_path.unlink()
        deleted = True
    index = _load_index(library_dir)
    if slug in index.get("entries", {}):
        del index["entries"][slug]
        _save_index(library_dir, index)
        deleted = True
    return deleted


def load_pattern_for_edit(slug: str, library_dir: Path) -> Optional[Dict[str, str]]:
    """
    Load a library entry and convert it back to flat form fields for editing.
    Returns None if the entry doesn't exist.
    """
    if not _YAML_AVAILABLE:
        return None
    yaml_path = library_dir / f"{slug}.yaml"
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            pattern = yaml.safe_load(f)
    except Exception:
        return None

    form: Dict[str, str] = {
        "title": pattern.get("title", ""),
        "description": pattern.get("description", ""),
        "category": pattern.get("category", ""),
        "best_practices": pattern.get("best_practices", ""),
    }

    # Use cases → text
    uc = pattern.get("use_cases", [])
    form["use_cases_text"] = "\n".join(str(u) for u in uc) if uc else ""

    # Parameters → text
    params = pattern.get("parameters", [])
    if params:
        lines = []
        for p in params:
            if isinstance(p, dict):
                ex = f" ({p.get('example', '')})" if p.get("example") else ""
                lines.append(f"{p.get('name', '')}: {p.get('description', '')}{ex}")
        form["parameters_text"] = "\n".join(lines)
    else:
        form["parameters_text"] = ""

    # SQL template
    tmpl = pattern.get("templates", {})
    if isinstance(tmpl, dict):
        first = next(iter(tmpl.values()), {})
        form["sql_example"] = first.get("sql", "") or first.get("content", "")
        form["sql_description"] = first.get("description", "")
    else:
        form["sql_example"] = ""
        form["sql_description"] = ""

    # Example output
    examples = pattern.get("examples", [])
    if examples and isinstance(examples[0], dict):
        form["example_output"] = examples[0].get("expected_result", "")
    else:
        form["example_output"] = ""

    # Errors → text
    errors = pattern.get("common_errors", [])
    if errors:
        lines = []
        for e in errors:
            if isinstance(e, dict):
                lines.append(f"{e.get('error', '')}: {e.get('solution', '')}")
        form["common_errors_text"] = "\n".join(lines)
    else:
        form["common_errors_text"] = ""

    return form


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def preview_qa(form_data: Dict[str, str]) -> Tuple[List[Tuple[str, str]], Optional[List[Dict]]]:
    """
    Given current form field values, return (qa_pairs, multiturn_conversation)
    without saving anything. Used for the "Preview" button in the UI.
    """
    pattern = form_to_pattern(form_data)
    qa = generate_qa_from_pattern(pattern)
    mt = generate_multiturn_from_pattern(pattern)
    return qa, mt
