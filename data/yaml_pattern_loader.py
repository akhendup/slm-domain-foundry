#!/usr/bin/env python3
"""
Load Teradata/SQL pattern YAML files and convert them into training Q&A pairs.

Pattern YAML files follow the structure used in sample_data/patternexamples/:
  name, title, description, use_cases, parameters, templates, examples,
  common_errors, best_practices, pattern_syntax, guardrails, related_patterns

Each section generates a distinct set of typed Q&A pairs and multi-turn conversations.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a YAML pattern file. Returns None on failure."""
    if not _YAML_AVAILABLE:
        raise ImportError("pyyaml is required: pip install pyyaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_yaml_patterns_dir(directory: Path) -> List[Dict[str, Any]]:
    """
    Recursively find and load all .yaml files under directory.
    Returns list of parsed pattern dicts (skips files that fail to parse).
    """
    patterns = []
    for yaml_path in sorted(directory.rglob("*.yaml")):
        data = _load_yaml(yaml_path)
        if data and data.get("name"):
            data["_source_file"] = str(yaml_path)
            patterns.append(data)
    return patterns


# ---------------------------------------------------------------------------
# Q&A generation helpers
# ---------------------------------------------------------------------------

def _join_list(items: Any, sep: str = "\n- ") -> str:
    """Join a list of strings into a bullet list."""
    if not items:
        return ""
    if isinstance(items, str):
        return items.strip()
    return sep + sep.join(str(x).strip() for x in items if x)


def _fmt_param(p: Dict) -> str:
    """Format a single parameter as readable text."""
    parts = [f"**{p.get('name', '?')}**"]
    if p.get("type"):
        parts.append(f"(type: {p['type']})")
    if p.get("required") is False:
        parts.append("(optional)")
    desc = p.get("description", "")
    if desc:
        parts.append(f"— {desc}")
    ex = p.get("example", "")
    if ex:
        parts.append(f"Example: {ex}")
    default = p.get("default")
    if default is not None:
        parts.append(f"Default: {default}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main Q&A generator
# ---------------------------------------------------------------------------

def generate_qa_from_pattern(pattern: Dict) -> List[Tuple[str, str]]:
    """
    Generate diverse typed Q&A pairs from a single parsed pattern dict.
    Covers: description, use cases, parameters, templates, examples,
            common errors, best practices, pattern syntax, guardrails.
    """
    pairs: List[Tuple[str, str]] = []

    title = (pattern.get("title") or pattern.get("name") or "").strip()
    name = (pattern.get("name") or "").strip()
    fn = title or name
    td_func = pattern.get("teradata_function", "")

    # -- Description ----------------------------------------------------------
    desc = (pattern.get("description") or "").strip()
    if desc and fn:
        pairs.append((f"What is {fn}?", desc))
        pairs.append((f"What does {fn} do?", desc))
        if len(desc) > 120:
            pairs.append((f"When should I use {fn}?", desc))
            pairs.append((f"Describe the purpose of {fn}.", desc))
        if td_func and td_func.upper() != name.upper():
            pairs.append((f"What is the {td_func} function in Teradata?", desc))

    # -- Use cases ------------------------------------------------------------
    use_cases = pattern.get("use_cases", [])
    if use_cases and fn:
        uc_text = _join_list(use_cases)
        pairs.append((f"What are the use cases for {fn}?", uc_text))
        pairs.append((f"What problems can {fn} solve?", uc_text))
        # Individual use-case questions if there are enough
        for uc in use_cases[:5]:
            uc = str(uc).strip()
            if len(uc) > 10:
                pairs.append((f"Can {fn} be used for {uc.lower()}?", desc or uc_text))

    # -- Parameters -----------------------------------------------------------
    parameters = pattern.get("parameters", [])
    if parameters and fn:
        all_params_text = "\n\n".join(_fmt_param(p) for p in parameters if isinstance(p, dict))
        if all_params_text:
            pairs.append((f"What are the parameters for {fn}?", all_params_text))
            pairs.append((f"What inputs does {fn} require?", all_params_text))
        # One question per parameter
        for p in parameters:
            if not isinstance(p, dict):
                continue
            pname = p.get("name", "")
            pdesc = p.get("description", "")
            pex = p.get("example", "")
            if pname and pdesc:
                answer = pdesc
                if pex:
                    answer += f"\n\nExample value: {pex}"
                hint = p.get("LLM-HINT") or next(
                    (v for k, v in p.items() if "hint" in k.lower()), ""
                )
                if hint:
                    answer += f"\n\nNote: {hint}"
                pairs.append((
                    f"What does the '{pname}' parameter do in {fn}?",
                    answer,
                ))

    # -- Templates (SQL or content) -------------------------------------------
    templates = pattern.get("templates", {})
    if isinstance(templates, dict) and fn:
        for tname, tval in templates.items():
            if not isinstance(tval, dict):
                continue
            tdesc = (tval.get("description") or tname.replace("_", " ")).strip()
            sql = (tval.get("sql") or tval.get("content") or "").strip()
            if not sql:
                continue
            pairs.append((f"How do I {tdesc.lower()}?", sql))
            pairs.append((f"Write a SQL query to {tdesc.lower()}.", sql))
            req = tval.get("required_parameters", [])
            if req:
                req_note = f"Required parameters: {', '.join(req)}"
                pairs.append((
                    f"What does the {tname.replace('_', ' ')} template for {fn} look like?",
                    f"{sql}\n\n{req_note}",
                ))

    # -- Examples -------------------------------------------------------------
    examples = pattern.get("examples", [])
    if isinstance(examples, list) and fn:
        for ex in examples:
            if not isinstance(ex, dict):
                continue
            exname = (ex.get("name") or "").strip()
            exdesc = (ex.get("description") or exname).strip()
            expected = (ex.get("expected_result") or "").strip()
            ex_params = ex.get("parameters", {})

            if exname and expected:
                pairs.append((
                    f"Show me an example of {fn} for {exname}.",
                    expected,
                ))
            if exdesc and expected:
                pairs.append((
                    f"How do I use {fn} to {exdesc.lower()}?",
                    expected,
                ))
            # If example has a pattern param, ask specifically about it
            if isinstance(ex_params, dict) and "pattern" in ex_params:
                pat_val = ex_params["pattern"]
                pairs.append((
                    f"What pattern would I use for '{exname}' with {fn}?",
                    f"Pattern: {pat_val}\n\n{expected}" if expected else f"Pattern: {pat_val}",
                ))

    # -- Common errors --------------------------------------------------------
    common_errors = pattern.get("common_errors", [])
    if isinstance(common_errors, list) and common_errors and fn:
        # Combined errors answer
        error_lines = []
        for err in common_errors:
            if not isinstance(err, dict):
                continue
            e = err.get("error", "")
            cause = err.get("cause", "")
            sol = err.get("solution", "")
            if e:
                error_lines.append(
                    f"**Error**: {e}\n**Cause**: {cause}\n**Solution**: {sol}"
                )
        if error_lines:
            combined = "\n\n".join(error_lines)
            pairs.append((f"What are common errors when using {fn}?", combined))
            pairs.append((f"What gotchas should I know about {fn}?", combined))
        # Individual error questions
        for err in common_errors:
            if not isinstance(err, dict):
                continue
            e = err.get("error", "")
            cause = err.get("cause", "")
            sol = err.get("solution", "")
            if e and sol:
                pairs.append((
                    f"What causes '{e}' when using {fn}?",
                    f"Cause: {cause}\n\nSolution: {sol}" if cause else sol,
                ))
                pairs.append((f"How do I fix '{e}'?", sol))

    # -- Best practices -------------------------------------------------------
    best_practices = (pattern.get("best_practices") or "").strip()
    if best_practices and fn:
        pairs.append((f"What are the best practices for {fn}?", best_practices))
        pairs.append((f"What should I know before using {fn}?", best_practices))

    # -- Pattern syntax (nPath-style) ----------------------------------------
    pattern_syntax = pattern.get("pattern_syntax", {})
    if isinstance(pattern_syntax, dict) and fn:
        operators = pattern_syntax.get("operators", [])
        if operators:
            op_lines = []
            for op in operators:
                if not isinstance(op, dict):
                    continue
                sym = op.get("symbol", "")
                meaning = op.get("meaning", "")
                example = op.get("example", "")
                if sym and meaning:
                    op_lines.append(f"'{sym}' — {meaning}. {example}")
                    pairs.append((
                        f"What does the '{sym}' operator mean in {fn} patterns?",
                        f"{meaning}\n\nExample: {example}" if example else meaning,
                    ))
            if op_lines:
                pairs.append((
                    f"What are the {fn} pattern syntax operators?",
                    "\n".join(op_lines),
                ))

    # -- Guardrails / limitations ---------------------------------------------
    guardrails = pattern.get("guardrails", [])
    if isinstance(guardrails, list) and guardrails and fn:
        gl_text = _join_list(guardrails)
        pairs.append((f"What are the limitations of {fn}?", gl_text))

    # -- Fallback if nothing generated ----------------------------------------
    if not pairs and fn and desc:
        pairs.append((f"What does the documentation say about {fn}?", desc))

    return pairs


# ---------------------------------------------------------------------------
# Multi-turn conversation generator
# ---------------------------------------------------------------------------

def generate_multiturn_from_pattern(pattern: Dict) -> Optional[List[Dict]]:
    """
    Build a multi-turn ShareGPT conversation from a pattern dict.
    Simulates a natural exploration: what is it → use cases → syntax → example → gotchas.
    Returns None if not enough content for at least 2 turns.
    """
    title = (pattern.get("title") or pattern.get("name") or "").strip()
    fn = title
    if not fn:
        return None

    desc = (pattern.get("description") or "").strip()
    use_cases = pattern.get("use_cases", [])
    templates = pattern.get("templates", {})
    examples = pattern.get("examples", [])
    best_practices = (pattern.get("best_practices") or "").strip()
    common_errors = pattern.get("common_errors", [])

    if not desc:
        return None

    turns: List[Dict] = [
        {"role": "user", "content": f"What is {fn}?"},
        {"role": "assistant", "content": desc},
    ]

    if use_cases:
        uc_text = _join_list(use_cases)
        turns += [
            {"role": "user", "content": f"What are the use cases for {fn}?"},
            {"role": "assistant", "content": uc_text},
        ]

    # Best template as syntax example
    first_sql = ""
    if isinstance(templates, dict):
        for tname, tval in templates.items():
            if isinstance(tval, dict):
                sql = (tval.get("sql") or tval.get("content") or "").strip()
                if sql:
                    tdesc = (tval.get("description") or tname.replace("_", " ")).strip()
                    first_sql = sql
                    turns += [
                        {"role": "user", "content": f"Can you show me the syntax for {fn}?"},
                        {"role": "assistant", "content": f"{tdesc}:\n\n{sql}"},
                    ]
                    break

    # First concrete example
    if isinstance(examples, list) and examples:
        ex = examples[0]
        if isinstance(ex, dict):
            exname = (ex.get("name") or "").strip()
            expected = (ex.get("expected_result") or "").strip()
            if exname and expected:
                turns += [
                    {"role": "user", "content": f"Can you show me a concrete example?"},
                    {"role": "assistant", "content": expected},
                ]

    # Gotchas
    if common_errors:
        error_lines = []
        for err in common_errors[:3]:  # limit to 3
            if isinstance(err, dict) and err.get("error"):
                error_lines.append(
                    f"- {err['error']}: {err.get('solution', '')}"
                )
        if error_lines:
            turns += [
                {"role": "user", "content": f"Are there any common errors or gotchas I should watch out for?"},
                {"role": "assistant", "content": "\n".join(error_lines)},
            ]
    elif best_practices:
        turns += [
            {"role": "user", "content": f"What are the best practices for {fn}?"},
            {"role": "assistant", "content": best_practices[:800]},
        ]

    return turns if len(turns) >= 4 else None


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def load_patterns_as_qa(
    yaml_dir: Path,
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """
    Load all YAML patterns from yaml_dir (recursively) and return:
      (qa_pairs, multiturn_conversations)
    where qa_pairs is a list of (question, answer) tuples and
    multiturn_conversations is a list of ShareGPT conversation dicts.
    """
    patterns = load_yaml_patterns_dir(yaml_dir)
    all_qa: List[Tuple[str, str]] = []
    all_mt: List[Dict] = []

    for pat in patterns:
        qa = generate_qa_from_pattern(pat)
        all_qa.extend(qa)
        mt = generate_multiturn_from_pattern(pat)
        if mt:
            all_mt.append({"conversations": mt})

    return all_qa, all_mt
