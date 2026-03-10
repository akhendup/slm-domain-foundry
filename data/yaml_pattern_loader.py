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
    from data.question_templates import (
        DESCRIPTION_QUESTIONS,
        ONE_SENTENCE_QUESTIONS,
        CATEGORY_QUESTIONS,
        USE_CASE_QUESTIONS,
        PARAMETER_QUESTIONS,
    )
except ImportError:
    # Fallback if run standalone (tests, scripts)
    DESCRIPTION_QUESTIONS = ["What is {fn}?", "What does {fn} do?"]
    ONE_SENTENCE_QUESTIONS = ["Define {fn} in one sentence."]
    CATEGORY_QUESTIONS = ["What category of SQL function is {fn}?"]
    USE_CASE_QUESTIONS = ["What are the use cases for {fn}?", "When should I use {fn}?"]
    PARAMETER_QUESTIONS = ["What are the parameters for {fn}?"]

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
    Covers: description, use cases, parameters (including cross-parameter pairs),
    templates, use-case×template combos, examples, common errors (per-error deep
    dive), best practices (per-item), pattern operators (cross-pair), guardrails,
    related-pattern contrastive questions.
    """
    pairs: List[Tuple[str, str]] = []

    title = (pattern.get("title") or pattern.get("name") or "").strip()
    name = (pattern.get("name") or "").strip()
    fn = title or name
    td_func = pattern.get("teradata_function", "")

    # -- Description ----------------------------------------------------------
    desc = (pattern.get("description") or "").strip()
    category = (pattern.get("category") or "").strip()
    if desc and fn:
        # Full-description questions (from template list — easy to add more)
        for q_tmpl in DESCRIPTION_QUESTIONS:
            pairs.append((q_tmpl.format(fn=fn), desc))

        # Single-sentence questions — first sentence only
        sentences = re.split(r"(?<=[.!?])\s+", desc)
        first_sentence = sentences[0].strip() if sentences else ""
        if first_sentence and len(first_sentence) < len(desc) - 20:
            for q_tmpl in ONE_SENTENCE_QUESTIONS:
                pairs.append((q_tmpl.format(fn=fn), first_sentence))

        # Category questions — only if category field is populated
        if category:
            cat_answer = (
                f"{fn} is a {category} function. {first_sentence}"
                if first_sentence else f"{fn} is a {category} function. {desc}"
            )
            for q_tmpl in CATEGORY_QUESTIONS:
                pairs.append((q_tmpl.format(fn=fn), cat_answer))

        # td_func alias question
        if td_func and td_func.upper() != name.upper():
            pairs.append((f"What is the {td_func} function in Teradata?", desc))

    # -- Use cases ------------------------------------------------------------
    use_cases = pattern.get("use_cases", [])
    use_cases_list = [str(uc).strip() for uc in (use_cases or []) if str(uc).strip()]
    if use_cases_list and fn:
        uc_text = _join_list(use_cases_list)
        # Use-case questions driven by template list (easy to add more in question_templates.py)
        for q_tmpl in USE_CASE_QUESTIONS:
            pairs.append((q_tmpl.format(fn=fn), uc_text))
        if desc:
            pairs.append((f"Who would typically use {fn} and for what purpose?",
                          f"Use cases:\n{uc_text}\n\nOverview: {desc}"))
        for uc in use_cases_list[:10]:
            if len(uc) > 10:
                pairs.append((
                    f"Can {fn} be used for {uc.lower()}?",
                    f"Yes, {fn} supports this: {uc}\n\nAll use cases:\n{uc_text}",
                ))
                pairs.append((
                    f"How would you use {fn} to handle {uc.lower()}?",
                    f"{fn} can be applied to: {uc}\n\n{desc}" if desc else uc_text,
                ))
                # NOTE: parameter-relevance-per-use-case is added after params are computed below

    # -- Parameters -----------------------------------------------------------
    parameters = pattern.get("parameters", [])
    valid_params = [p for p in parameters if isinstance(p, dict) and p.get("name") and p.get("description")]
    if valid_params and fn:
        all_params_text = "\n\n".join(_fmt_param(p) for p in valid_params)
        # Parameter questions driven by template list
        for q_tmpl in PARAMETER_QUESTIONS:
            pairs.append((q_tmpl.format(fn=fn), all_params_text))
        # Now that we have all_params_text, add per-use-case parameter questions
        for uc in use_cases_list[:5]:
            if len(uc) > 10:
                pairs.append((
                    f"What parameters are most important when using {fn} for {uc.lower()}?",
                    f"When using {fn} for {uc}, configure these parameters:\n\n{all_params_text}",
                ))

        required_params = [p for p in valid_params if p.get("required") is not False]
        optional_params = [p for p in valid_params if p.get("required") is False]
        if required_params:
            req_text = "\n\n".join(_fmt_param(p) for p in required_params)
            pairs.append((f"Which parameters are required for {fn}?", req_text))
        if optional_params:
            opt_text = "\n\n".join(_fmt_param(p) for p in optional_params)
            pairs.append((f"Which parameters are optional in {fn}?", opt_text))

        # Deep per-parameter questions
        for p in valid_params:
            pname = p.get("name", "")
            pdesc = p.get("description", "")
            pex = p.get("example", "")
            ptype = p.get("type", "")
            pdefault = p.get("default")
            prequired = p.get("required")
            hint = p.get("LLM-HINT") or next(
                (v for k, v in p.items() if "hint" in k.lower()), ""
            )
            base_answer = pdesc
            if pex:
                base_answer += f"\n\nExample value: {pex}"
            if hint:
                base_answer += f"\n\nNote: {hint}"

            req_word = "required" if prequired is not False else "optional"
            pairs.append((f"What does the '{pname}' parameter do in {fn}?", base_answer))
            pairs.append((f"Is '{pname}' required or optional in {fn}?",
                          f"'{pname}' is {req_word}. {pdesc}"))
            if ptype:
                pairs.append((f"What data type does '{pname}' accept in {fn}?",
                               f"The '{pname}' parameter accepts {ptype}. {pdesc}"))
            if pdefault is not None:
                pairs.append((f"What is the default value of '{pname}' in {fn}?",
                               f"The default value of '{pname}' is {pdefault}. {pdesc}"))
            pairs.append((f"What happens if you omit '{pname}' from {fn}?",
                          f"'{pname}' is {req_word}. {pdesc}"))
            if pex:
                pairs.append((f"Give me an example value for '{pname}' in {fn}.",
                               f"Example: {pex}\n\n{pdesc}"))
            pairs.append((f"How does '{pname}' affect the output of {fn}?", base_answer))
            pairs.append((f"What is the role of '{pname}' in controlling {fn} behavior?", base_answer))
            if ptype:
                pairs.append((f"What are valid values for '{pname}' in {fn}?",
                               f"'{pname}' accepts {ptype}. {pdesc}" + (f" Example: {pex}" if pex else "")))
            pairs.append((f"What error occurs if '{pname}' is set incorrectly in {fn}?",
                          f"'{pname}' must satisfy: {pdesc}. Incorrect values will cause errors."))

        # Cross-parameter combination questions
        for i, pA in enumerate(valid_params):
            for pB in valid_params[i + 1:]:
                nA, nB = pA.get("name", ""), pB.get("name", "")
                dA, dB = pA.get("description", ""), pB.get("description", "")
                exA, exB = pA.get("example", ""), pB.get("example", "")
                cross_answer = (
                    f"'{nA}': {dA}" + (f" (e.g. {exA})" if exA else "") +
                    f"\n\n'{nB}': {dB}" + (f" (e.g. {exB})" if exB else "")
                )
                pairs.append((f"Can '{nA}' and '{nB}' be used together in {fn}?", cross_answer))
                pairs.append((f"How does '{nA}' interact with '{nB}' in {fn}?", cross_answer))
                pairs.append((f"When configuring '{nA}', what should '{nB}' be set to in {fn}?", cross_answer))

    # -- Templates (SQL or content) -------------------------------------------
    templates = pattern.get("templates", {})
    first_use_case = use_cases_list[0] if use_cases_list else ""
    template_items: List[Tuple[str, Dict]] = []
    if isinstance(templates, dict):
        template_items = [(tn, tv) for tn, tv in templates.items() if isinstance(tv, dict)]

    if template_items and fn:
        for tname, tval in template_items:
            tdesc = (tval.get("description") or tname.replace("_", " ")).strip()
            sql = (tval.get("sql") or tval.get("content") or "").strip()
            if not sql:
                continue
            req = tval.get("required_parameters", [])
            req_note = f"\n\nRequired parameters: {', '.join(req)}" if req else ""
            full_answer = f"{sql}{req_note}"

            pairs.append((f"How do I {tdesc.lower()}?", full_answer))
            pairs.append((f"Write a SQL query to {tdesc.lower()}.", full_answer))
            if req:
                pairs.append((
                    f"What does the {tname.replace('_', ' ')} template for {fn} look like?",
                    full_answer,
                ))
            pairs.append((f"Walk me through the {tname.replace('_', ' ')} template for {fn}.", full_answer))
            pairs.append((
                f"What are the required parameters for the {tname.replace('_', ' ')} template in {fn}?",
                f"Required parameters: {', '.join(req)}\n\nTemplate:\n{sql}" if req
                else f"All parameters are optional for this template.\n\nTemplate:\n{sql}",
            ))
            pairs.append((
                f"What does each clause in the {tname.replace('_', ' ')} {fn} template do?",
                full_answer,
            ))
            if first_use_case:
                pairs.append((
                    f"How would I adapt the {tname.replace('_', ' ')} template for {first_use_case.lower()}?",
                    f"Starting from the {tname.replace('_', ' ')} template:\n\n{sql}\n\nAdapt it for: {first_use_case}",
                ))

        # Use-case × template cross questions (first 5 × first 5)
        for uc in use_cases_list[:5]:
            for tname, tval in template_items[:5]:
                sql = (tval.get("sql") or tval.get("content") or "").strip()
                if not sql:
                    continue
                tdisp = tname.replace("_", " ")
                pairs.append((
                    f"Which {fn} template would you use for {uc.lower()}?",
                    f"For '{uc}', the {tdisp} template is a good starting point:\n\n{sql}",
                ))
                pairs.append((
                    f"Adapt the {tdisp} template to solve: {uc.lower()}",
                    f"Base template ({tdisp}):\n\n{sql}\n\nAdapt the pattern and result clauses for: {uc}",
                ))

    # -- Examples -------------------------------------------------------------
    examples = pattern.get("examples", [])
    if isinstance(examples, list) and fn:
        # Canonical "Show me an example of nPath SQL" -> full SQL (fixes wrong filesystem-path answers)
        first_sql = ""
        for ex in examples:
            if isinstance(ex, dict):
                sq = (ex.get("sql") or "").strip()
                if sq:
                    first_sql = sq
                    break
        if first_sql:
            short = (pattern.get("teradata_function") or fn or "").strip()
            if short.upper() == "NPATH":
                short = "nPath"
            pairs.append((f"Show me an example of {short} SQL.", first_sql))
            pairs.append((f"Show me an example of {short}.", first_sql))
            if short != fn:
                pairs.append((f"Show me an example of {fn} SQL.", first_sql))

        for ex in examples:
            if not isinstance(ex, dict):
                continue
            exname = (ex.get("name") or "").strip()
            exdesc = (ex.get("description") or exname).strip()
            expected = (ex.get("expected_result") or "").strip()
            ex_params = ex.get("parameters", {})
            ex_sql = (ex.get("sql") or "").strip()
            content = expected or ex_sql or exdesc
            if not content:
                continue

            if exname and expected:
                pairs.append((f"Show me an example of {fn} for {exname}.", expected))
            if exdesc and expected:
                pairs.append((f"How do I use {fn} to {exdesc.lower()}?", expected))
            if isinstance(ex_params, dict) and "pattern" in ex_params:
                pat_val = ex_params["pattern"]
                pairs.append((
                    f"What pattern would I use for '{exname}' with {fn}?",
                    f"Pattern: {pat_val}\n\n{expected}" if expected else f"Pattern: {pat_val}",
                ))

            # Expanded example questions
            if exname:
                pairs.append((f"Explain the {fn} query for '{exname}'.", content))
                pairs.append((
                    f"What business question does the '{exname}' {fn} example answer?",
                    f"The '{exname}' example addresses: {exdesc}\n\n{content}",
                ))
                pairs.append((f"What output would you expect from the '{exname}' {fn} example?", content))
                if ex_params:
                    pairs.append((
                        f"What are the key parameters shown in the '{exname}' {fn} example?",
                        "\n".join(f"  {k}: {v}" for k, v in ex_params.items()) if isinstance(ex_params, dict) else str(ex_params),
                    ))
                if isinstance(ex_params, dict) and "pattern" in ex_params:
                    pat_val = ex_params["pattern"]
                    pairs.append((
                        f"What does the PATTERN '{pat_val}' match in the '{exname}' {fn} example?",
                        content,
                    ))
                pairs.append((
                    f"How would you modify the '{exname}' {fn} example to change the time window?",
                    f"Starting from:\n\n{content}\n\nAdjust the ORDER BY column and pattern conditions to change the time window.",
                ))

    # -- Common errors --------------------------------------------------------
    common_errors = pattern.get("common_errors", [])
    if isinstance(common_errors, list) and common_errors and fn:
        error_lines = []
        for err in common_errors:
            if not isinstance(err, dict):
                continue
            e = err.get("error", "")
            cause = err.get("cause", "")
            sol = err.get("solution", "")
            if e:
                error_lines.append(f"**Error**: {e}\n**Cause**: {cause}\n**Solution**: {sol}")
        if error_lines:
            combined = "\n\n".join(error_lines)
            pairs.append((f"What are common errors when using {fn}?", combined))
            pairs.append((f"What gotchas should I know about {fn}?", combined))

        for err in common_errors:
            if not isinstance(err, dict):
                continue
            e = err.get("error", "")
            cause = err.get("cause", "")
            sol = err.get("solution", "")
            if not e:
                continue
            cause_sol = f"Cause: {cause}\n\nSolution: {sol}" if cause else sol
            if sol:
                pairs.append((f"What causes '{e}' when using {fn}?", cause_sol))
                pairs.append((f"How do I fix '{e}'?", sol))
                pairs.append((
                    f"How do I prevent '{e}' in {fn}?",
                    f"To prevent this: {sol}" + (f"\n\nRoot cause: {cause}" if cause else ""),
                ))
                pairs.append((f"What should I check first when I see '{e}' in {fn}?", cause_sol))
            if cause:
                pairs.append((
                    f"What are the symptoms of '{e}' in {fn}?",
                    f"Error: {e}\nCause: {cause}" + (f"\nSolution: {sol}" if sol else ""),
                ))

    # -- Best practices -------------------------------------------------------
    best_practices = pattern.get("best_practices")
    if best_practices and fn:
        if isinstance(best_practices, str):
            bp_text = best_practices.strip()
            if bp_text:
                pairs.append((f"What are the best practices for {fn}?", bp_text))
                pairs.append((f"What should I know before using {fn}?", bp_text))
                pairs.append((f"What are the most important guidelines for {fn}?", bp_text))
        elif isinstance(best_practices, list):
            bp_text = _join_list(best_practices)
            pairs.append((f"What are the best practices for {fn}?", bp_text))
            pairs.append((f"What should I know before using {fn}?", bp_text))
            for bp in best_practices[:12]:
                bp = str(bp).strip()
                if len(bp) > 15:
                    topic = re.split(r"[,\.]", bp)[0][:60].strip()
                    pairs.append((f"What is the best practice regarding '{topic}' when using {fn}?", bp))
                    pairs.append((f"Why is it recommended to follow this {fn} guideline: {topic}?", bp))
                    pairs.append((f"What happens if you skip this {fn} practice: {topic}?", bp))
        elif isinstance(best_practices, dict):
            bp_text = "\n\n".join(f"**{k}**: {v}" for k, v in best_practices.items())
            pairs.append((f"What are the best practices for {fn}?", bp_text))
            pairs.append((f"What should I know before using {fn}?", bp_text))
            for bpk, bpv in list(best_practices.items())[:12]:
                topic = str(bpk).replace("_", " ").strip()
                bpv_str = str(bpv).strip()
                if topic and bpv_str:
                    pairs.append((f"What is the best practice for '{topic}' in {fn}?", bpv_str))
                    pairs.append((f"Why is '{topic}' important when using {fn}?", bpv_str))
                    pairs.append((f"What happens if you ignore '{topic}' in {fn}?", bpv_str))

    # -- Pattern syntax (nPath-style) ----------------------------------------
    pattern_syntax = pattern.get("pattern_syntax", {})
    if isinstance(pattern_syntax, dict) and fn:
        operators = pattern_syntax.get("operators", [])
        valid_ops = [op for op in operators if isinstance(op, dict) and op.get("symbol") and op.get("meaning")]
        if valid_ops:
            op_lines = []
            for op in valid_ops:
                sym = op.get("symbol", "")
                meaning = op.get("meaning", "")
                example = op.get("example", "")
                op_lines.append(f"'{sym}' — {meaning}." + (f" {example}" if example else ""))
                pairs.append((
                    f"What does the '{sym}' operator mean in {fn} patterns?",
                    f"{meaning}" + (f"\n\nExample: {example}" if example else ""),
                ))
                if example:
                    pairs.append((
                        f"Give me an example pattern using the '{sym}' operator in {fn}.",
                        f"Example: {example}\n\nMeaning: {meaning}",
                    ))
                pairs.append((
                    f"When would you use the '{sym}' operator in an {fn} pattern?",
                    meaning + (f"\n\nExample: {example}" if example else ""),
                ))
            if op_lines:
                pairs.append((f"What are the {fn} pattern syntax operators?", "\n".join(op_lines)))

            # Cross-operator pairs
            for i, opA in enumerate(valid_ops):
                for opB in valid_ops[i + 1:]:
                    symA, symB = opA.get("symbol", ""), opB.get("symbol", "")
                    mA, mB = opA.get("meaning", ""), opB.get("meaning", "")
                    exA, exB = opA.get("example", ""), opB.get("example", "")
                    cross = (
                        f"'{symA}' — {mA}" + (f" Example: {exA}" if exA else "") +
                        f"\n\n'{symB}' — {mB}" + (f" Example: {exB}" if exB else "")
                    )
                    pairs.append((f"What is the difference between '{symA}' and '{symB}' operators in {fn}?", cross))
                    pairs.append((f"Can the '{symA}' and '{symB}' operators be combined in an {fn} pattern?", cross))

    # -- Guardrails / limitations ---------------------------------------------
    guardrails = pattern.get("guardrails", [])
    if isinstance(guardrails, list) and guardrails and fn:
        gl_text = _join_list(guardrails)
        pairs.append((f"What are the limitations of {fn}?", gl_text))
        pairs.append((f"When should you NOT use {fn}?", gl_text))
        pairs.append((f"What are the restrictions on {fn}?", gl_text))

    # -- Related patterns — contrastive questions -----------------------------
    related_patterns = pattern.get("related_patterns", [])
    if isinstance(related_patterns, list) and related_patterns and fn and desc:
        for rel in related_patterns[:5]:
            rel = str(rel).strip()
            if not rel:
                continue
            pairs.append((f"How is {fn} different from {rel}?",
                          f"{fn}: {desc}\n\n{rel} is a related pattern — consult its documentation for details."))
            pairs.append((f"When should I use {fn} instead of {rel}?",
                          f"{fn} is designed for: {desc}"))
            pairs.append((f"What can {fn} do that {rel} cannot?",
                          f"{fn} specializes in: {desc}"))
            pairs.append((f"Can {rel} replace {fn} in all use cases?",
                          f"No. {fn} is specifically designed for: {desc}"))

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
    best_practices_raw = pattern.get("best_practices")
    if isinstance(best_practices_raw, str):
        best_practices = best_practices_raw.strip()
    elif isinstance(best_practices_raw, list):
        best_practices = _join_list(best_practices_raw)
    elif isinstance(best_practices_raw, dict):
        best_practices = "\n\n".join(f"**{k}**: {v}" for k, v in best_practices_raw.items())
    else:
        best_practices = ""
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
    if isinstance(templates, dict):
        for tname, tval in templates.items():
            if isinstance(tval, dict):
                sql = (tval.get("sql") or tval.get("content") or "").strip()
                if sql:
                    tdesc = (tval.get("description") or tname.replace("_", " ")).strip()
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
                    {"role": "user", "content": "Can you show me a concrete example?"},
                    {"role": "assistant", "content": expected},
                ]

    # Gotchas
    if common_errors:
        error_lines = []
        for err in common_errors[:3]:
            if isinstance(err, dict) and err.get("error"):
                error_lines.append(f"- {err['error']}: {err.get('solution', '')}")
        if error_lines:
            turns += [
                {"role": "user", "content": "Are there any common errors or gotchas I should watch out for?"},
                {"role": "assistant", "content": "\n".join(error_lines)},
            ]
    elif best_practices:
        turns += [
            {"role": "user", "content": f"What are the best practices for {fn}?"},
            {"role": "assistant", "content": best_practices[:800]},
        ]

    return turns if len(turns) >= 4 else None


def _build_debug_conversation(pattern: Dict, fn: str, desc: str) -> Optional[List[Dict]]:
    """Multi-turn: debug angle — diagnosing a query that returns no results."""
    common_errors = pattern.get("common_errors", [])
    templates = pattern.get("templates", {})
    first_sql = ""
    if isinstance(templates, dict):
        for tval in templates.values():
            if isinstance(tval, dict):
                sql = (tval.get("sql") or tval.get("content") or "").strip()
                if sql:
                    first_sql = sql
                    break
    if not first_sql and not common_errors:
        return None
    turns: List[Dict] = [
        {"role": "user", "content": f"I ran an {fn} query and it returned no results. How do I debug it?"},
        {"role": "assistant", "content": f"{fn} returning no results usually means the PATTERN didn't match any event sequences, the PARTITION data is too sparse, or the SYMBOLS clause is filtering too aggressively.\n\nDescription of {fn}: {desc}"},
    ]
    if first_sql:
        turns += [
            {"role": "user", "content": "Can you show me a diagnostic query I can start with?"},
            {"role": "assistant", "content": f"Start with this template and verify each clause:\n\n{first_sql}"},
        ]
    if common_errors:
        error_lines = [
            f"- {err['error']}: {err.get('solution', '')}"
            for err in common_errors[:4]
            if isinstance(err, dict) and err.get("error")
        ]
        if error_lines:
            turns += [
                {"role": "user", "content": "What are the most common mistakes that cause empty results?"},
                {"role": "assistant", "content": "\n".join(error_lines)},
            ]
    return turns if len(turns) >= 4 else None


def _build_performance_conversation(pattern: Dict, fn: str, desc: str) -> Optional[List[Dict]]:
    """Multi-turn: performance angle — optimizing a query."""
    best_practices_raw = pattern.get("best_practices")
    if isinstance(best_practices_raw, str):
        bp = best_practices_raw.strip()
    elif isinstance(best_practices_raw, list):
        bp = _join_list(best_practices_raw)
    elif isinstance(best_practices_raw, dict):
        bp = "\n\n".join(f"**{k}**: {v}" for k, v in best_practices_raw.items())
    else:
        bp = ""
    guardrails = pattern.get("guardrails", [])
    gl = _join_list(guardrails) if guardrails else ""
    if not bp and not gl:
        return None
    turns: List[Dict] = [
        {"role": "user", "content": f"How do I make {fn} queries run faster?"},
        {"role": "assistant", "content": f"Performance of {fn} depends on partition size, data ordering, and pattern complexity. {desc}"},
    ]
    if bp:
        turns += [
            {"role": "user", "content": "What are the performance best practices?"},
            {"role": "assistant", "content": bp[:800]},
        ]
    if gl:
        turns += [
            {"role": "user", "content": f"Are there any hard limits or restrictions I should know about?"},
            {"role": "assistant", "content": gl},
        ]
    return turns if len(turns) >= 4 else None


def _build_migration_conversation(pattern: Dict, fn: str, desc: str) -> Optional[List[Dict]]:
    """Multi-turn: migration angle — rewriting traditional SQL to use this function."""
    related = pattern.get("related_patterns", [])
    templates = pattern.get("templates", {})
    first_sql = ""
    if isinstance(templates, dict):
        for tval in templates.values():
            if isinstance(tval, dict):
                sql = (tval.get("sql") or tval.get("content") or "").strip()
                if sql:
                    first_sql = sql
                    break
    if not first_sql:
        return None
    turns: List[Dict] = [
        {"role": "user", "content": f"I'm currently using CASE statements and self-joins for path analysis. Should I switch to {fn}?"},
        {"role": "assistant", "content": f"{fn} is designed to replace complex CASE/self-join patterns for path analysis. {desc}"},
        {"role": "user", "content": f"What does a basic {fn} query look like?"},
        {"role": "assistant", "content": first_sql},
    ]
    if related:
        rel_str = ", ".join(str(r) for r in related[:3])
        turns += [
            {"role": "user", "content": f"Are there related functions I should also know about?"},
            {"role": "assistant", "content": f"Related patterns to consider alongside {fn}: {rel_str}"},
        ]
    return turns if len(turns) >= 4 else None


def generate_multiturn_conversations(pattern: Dict) -> List[List[Dict]]:
    """
    Generate multiple multi-turn conversations from different angles for a pattern.
    Returns a list of conversation turn lists (each suitable for ShareGPT format).
    """
    title = (pattern.get("title") or pattern.get("name") or "").strip()
    fn = title
    desc = (pattern.get("description") or "").strip()
    if not fn or not desc:
        return []

    conversations = []

    # Angle 1: standard exploration
    standard = generate_multiturn_from_pattern(pattern)
    if standard:
        conversations.append(standard)

    # Angle 2: debug
    debug = _build_debug_conversation(pattern, fn, desc)
    if debug:
        conversations.append(debug)

    # Angle 3: performance
    perf = _build_performance_conversation(pattern, fn, desc)
    if perf:
        conversations.append(perf)

    # Angle 4: migration
    migration = _build_migration_conversation(pattern, fn, desc)
    if migration:
        conversations.append(migration)

    return conversations


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
        for turns in generate_multiturn_conversations(pat):
            all_mt.append({"conversations": turns})

    return all_qa, all_mt
