#!/usr/bin/env python3
"""
Question templates for training Q&A generation.

Templates are loaded from question_templates.yaml in the same directory.
Edit that YAML file to add/remove/rephrase question variants without touching Python code.

If the YAML file cannot be loaded, built-in defaults are used so generation never fails.

Template placeholder: {fn} is replaced at generation time with the topic/function/section name.
Comparison templates use two placeholders: {fn} and {related}.

Exported lists (for import by yaml_pattern_loader.py, manual_extractor.py, template_expander.py):

  Technical (SQL / software documentation):
    DESCRIPTION_QUESTIONS, ONE_SENTENCE_QUESTIONS, CATEGORY_QUESTIONS,
    USE_CASE_QUESTIONS, PARAMETER_QUESTIONS, SYNTAX_QUESTIONS,
    ARGUMENT_QUESTIONS, NOTES_QUESTIONS, EXAMPLE_QUESTIONS,
    NULL_BEHAVIOR_QUESTIONS, PERFORMANCE_QUESTIONS,
    ERROR_QUESTIONS, COMPARISON_QUESTIONS

  General documents (textbooks, prose, non-technical):
    GENERAL_OVERVIEW_QUESTIONS, GENERAL_KEYCONCEPT_QUESTIONS,
    GENERAL_DETAIL_QUESTIONS, GENERAL_COMPARISON_QUESTIONS,
    GENERAL_APPLICATION_QUESTIONS, GENERAL_CAUSES_QUESTIONS,
    GENERAL_REQUIREMENTS_QUESTIONS

  Financial documents (bank statements, reports):
    FINANCIAL_TRANSACTION_QUESTIONS, FINANCIAL_ACCOUNT_QUESTIONS,
    FINANCIAL_ANALYSIS_QUESTIONS
"""

import logging
from pathlib import Path
from typing import Dict, List

_log = logging.getLogger(__name__)

_TEMPLATES_YAML = Path(__file__).parent / "question_templates.yaml"


def _load_yaml_templates() -> Dict:
    try:
        import yaml
        with open(_TEMPLATES_YAML, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        _log.warning("Could not load question_templates.yaml, using built-in defaults: %s", exc)
    return {}


def _get(templates: Dict, section: str, key: str, fallback: List[str]) -> List[str]:
    result = templates.get(section, {}).get(key)
    return result if isinstance(result, list) and result else fallback


_t = _load_yaml_templates()

# ---------------------------------------------------------------------------
# Technical — SQL functions, software documentation
# ---------------------------------------------------------------------------

DESCRIPTION_QUESTIONS: List[str] = _get(_t, "technical", "description", [
    "What is {fn}?",
    "What does {fn} do?",
    "Explain {fn} to me.",
    "What problem does {fn} solve?",
    "How would you describe {fn} to a new analyst?",
])

ONE_SENTENCE_QUESTIONS: List[str] = _get(_t, "technical", "one_sentence", [
    "Define {fn} in one sentence.",
    "Give me a brief one-line description of {fn}.",
    "Summarize {fn} in a single sentence.",
    "What is {fn} in plain English?",
])

CATEGORY_QUESTIONS: List[str] = _get(_t, "technical", "category", [
    "What category of SQL function is {fn}?",
    "What type of SQL function is {fn}?",
    "Is {fn} an analytic function, aggregate function, or something else?",
    "How is {fn} classified in SQL?",
])

USE_CASE_QUESTIONS: List[str] = _get(_t, "technical", "use_case", [
    "What are the use cases for {fn}?",
    "What problems can {fn} solve?",
    "In what scenarios is {fn} most useful?",
    "When should I use {fn}?",
    "What type of analysis does {fn} support?",
    "What kinds of questions can {fn} answer?",
    "Give me some examples of when to use {fn}.",
    "What business problems is {fn} designed for?",
    "What are real-world applications of {fn}?",
])

PARAMETER_QUESTIONS: List[str] = _get(_t, "technical", "parameter", [
    "What are the parameters for {fn}?",
    "What inputs does {fn} require?",
    "What configuration does {fn} need?",
    "List the parameters accepted by {fn}.",
    "What do I need to specify to call {fn}?",
])

SYNTAX_QUESTIONS: List[str] = _get(_t, "technical", "syntax", [
    "What is the syntax for {fn}?",
    "What are the required vs optional clauses in {fn}?",
    "Which keyword is mandatory in {fn} syntax?",
    "What is the minimum valid {fn} expression?",
    "How do you write a {fn} expression in SQL?",
    "Show me the {fn} SQL syntax.",
    "What does a basic {fn} query look like?",
    "What clauses does {fn} use?",
])

ARGUMENT_QUESTIONS: List[str] = _get(_t, "technical", "argument", [
    "What are the arguments to {fn}?",
    "What parameters does {fn} accept?",
    "Which arguments to {fn} are required?",
    "What is the default behavior when an optional argument is omitted from {fn}?",
    "What data type must be passed to {fn}?",
    "Can {fn} accept NULL values in its arguments?",
    "What happens if you pass invalid input to {fn}?",
    "How many arguments does {fn} require at minimum?",
])

NOTES_QUESTIONS: List[str] = _get(_t, "technical", "notes", [
    "What are the usage notes for {fn}?",
    "What are the prerequisites for {fn}?",
    "Are there any gotchas or restrictions when using {fn}?",
    "What are the performance implications of {fn}?",
    "Does {fn} support NULL values?",
    "What are the ordering requirements for {fn}?",
    "Can {fn} be nested inside another function?",
    "What data types does {fn} work with?",
    "What should I watch out for when using {fn}?",
    "What are the limitations of {fn}?",
])

EXAMPLE_QUESTIONS: List[str] = _get(_t, "technical", "example", [
    "Show me a complete example of {fn} with input and output.",
    "What is the purpose of this {fn} example?",
    "How would you modify this {fn} example to change the output?",
    "What are the key SQL clauses in this {fn} example?",
    "Explain the SQL logic in this {fn} example.",
    "What would happen if you changed the ORDER BY in this {fn} example?",
    "Write a SQL query using {fn}.",
    "Give me a real-world example of {fn}.",
])

NULL_BEHAVIOR_QUESTIONS: List[str] = _get(_t, "technical", "null_behavior", [
    "How does {fn} handle NULL values?",
    "What happens when NULL is passed to {fn}?",
    "Does {fn} return NULL if any input is NULL?",
    "What is the NULL handling behavior of {fn}?",
    "Can {fn} produce NULL in its output? When?",
    "How do NULLs affect the result of {fn}?",
    "Does {fn} propagate NULLs or ignore them?",
    "How do I handle NULL values when using {fn}?",
    "What does {fn} return when all inputs are NULL?",
    "Does {fn} include or exclude rows where the input is NULL?",
])

PERFORMANCE_QUESTIONS: List[str] = _get(_t, "technical", "performance", [
    "How do I optimize {fn} for large datasets?",
    "What are the performance tips for {fn}?",
    "What makes {fn} slow?",
    "How does data volume affect {fn} performance?",
    "What indexes or statistics help {fn} run faster?",
    "How does {fn} scale with increasing row count?",
    "When should I avoid {fn} for performance reasons?",
    "What is the memory footprint of {fn} for large windows?",
    "Is {fn} more efficient than its common alternatives?",
    "How can I use EXPLAIN PLAN to diagnose a slow {fn} query?",
])

ERROR_QUESTIONS: List[str] = _get(_t, "technical", "error_troubleshoot", [
    "What are common errors when using {fn}?",
    "Why does my {fn} query fail?",
    "How do I troubleshoot {fn} issues?",
    "What causes {fn} to return unexpected results?",
    "What are the most common mistakes with {fn}?",
    "What should I check first when {fn} gives wrong results?",
    "What edge cases break {fn}?",
    "Why does {fn} return NULL when I expect a value?",
    "How do I verify that my {fn} logic is correct?",
    "What test cases should I write to validate {fn} behavior?",
])

# Note: COMPARISON_QUESTIONS use TWO placeholders: {fn} and {related}.
# Use as: tmpl.format(fn=topic_name, related=related_name)
COMPARISON_QUESTIONS: List[str] = _get(_t, "technical", "comparison", [
    "How does {fn} differ from {related}?",
    "What is the difference between {fn} and {related}?",
    "When should I use {fn} instead of {related}?",
    "What does {fn} do that {related} cannot?",
    "Which is faster: {fn} or {related}?",
    "What are the trade-offs between {fn} and {related}?",
    "In what scenarios is {fn} preferable to {related}?",
    "How do {fn} and {related} handle NULL values differently?",
])

# ---------------------------------------------------------------------------
# General — Textbooks, prose manuals, articles, non-technical documents
# ---------------------------------------------------------------------------

GENERAL_OVERVIEW_QUESTIONS: List[str] = _get(_t, "general", "overview", [
    "What is {fn}?",
    "Summarize {fn}.",
    "Give me an overview of {fn}.",
    "What is the purpose of {fn}?",
    "What does this section say about {fn}?",
])

GENERAL_KEYCONCEPT_QUESTIONS: List[str] = _get(_t, "general", "key_concepts", [
    "What are the key concepts in {fn}?",
    "What are the most important ideas in {fn}?",
    "What terms are defined in {fn}?",
    "What is the central idea of {fn}?",
])

GENERAL_DETAIL_QUESTIONS: List[str] = _get(_t, "general", "detail", [
    "Explain {fn} in detail.",
    "How does {fn} work?",
    "Walk me through {fn} step by step.",
    "What are the components of {fn}?",
    "What factors affect {fn}?",
])

GENERAL_COMPARISON_QUESTIONS: List[str] = _get(_t, "general", "comparison", [
    "How does {fn} differ from other approaches?",
    "What makes {fn} unique?",
    "What are the advantages of {fn}?",
    "What are the disadvantages of {fn}?",
])

GENERAL_APPLICATION_QUESTIONS: List[str] = _get(_t, "general", "application", [
    "How is {fn} used in practice?",
    "Give an example of {fn}.",
    "What is a real-world application of {fn}?",
    "In what situations does {fn} apply?",
])

GENERAL_CAUSES_QUESTIONS: List[str] = _get(_t, "general", "causes_effects", [
    "What causes {fn}?",
    "What are the consequences of {fn}?",
    "What results from {fn}?",
])

GENERAL_REQUIREMENTS_QUESTIONS: List[str] = _get(_t, "general", "requirements", [
    "What are the requirements for {fn}?",
    "What is needed for {fn}?",
    "What are the conditions for {fn}?",
])

# ---------------------------------------------------------------------------
# Financial — Bank statements, financial reports, invoices
# ---------------------------------------------------------------------------

FINANCIAL_TRANSACTION_QUESTIONS: List[str] = _get(_t, "financial", "transaction", [
    "What transactions appear in {fn}?",
    "What are the debit entries in {fn}?",
    "What are the credit entries in {fn}?",
    "What is the largest transaction in {fn}?",
])

FINANCIAL_ACCOUNT_QUESTIONS: List[str] = _get(_t, "financial", "account_summary", [
    "What is the opening balance in {fn}?",
    "What is the closing balance in {fn}?",
    "What period does {fn} cover?",
    "What account does {fn} belong to?",
])

FINANCIAL_ANALYSIS_QUESTIONS: List[str] = _get(_t, "financial", "analysis", [
    "What spending categories appear in {fn}?",
    "What fees were charged in {fn}?",
    "Are there any unusual or large transactions in {fn}?",
])
