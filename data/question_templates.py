#!/usr/bin/env python3
"""
Question templates for training Q&A generation.

Templates are organized by the *source content type* they should be paired with.
Each template is a string with a single `{fn}` placeholder for the function/topic name.

To add new question variants: append to the appropriate list below.
The generation code in yaml_pattern_loader.py and manual_extractor.py imports
these lists and iterates over them — no logic changes needed.

Template categories:
    DESCRIPTION_QUESTIONS     → answered with the full description text
    ONE_SENTENCE_QUESTIONS    → answered with first sentence of description only
    CATEGORY_QUESTIONS        → answered with "X is a {category} function. <first sentence>"
    USE_CASE_QUESTIONS        → answered with the use_cases list
    PARAMETER_QUESTIONS       → answered with all parameters text
    SYNTAX_QUESTIONS          → answered with syntax text
    ARGUMENT_QUESTIONS        → answered with arguments/parameters text
    NOTES_QUESTIONS           → answered with usage notes / restrictions text
    EXAMPLE_QUESTIONS         → answered with the example SQL / expected output
"""

from typing import List

# ---------------------------------------------------------------------------
# Full description answers
# ---------------------------------------------------------------------------

DESCRIPTION_QUESTIONS: List[str] = [
    "What is {fn}?",
    "What does {fn} do?",
    "Explain {fn} to me.",
    "What problem does {fn} solve?",
    "How would you describe {fn} to a new analyst?",
]

# ---------------------------------------------------------------------------
# First-sentence / one-liner answers
# ---------------------------------------------------------------------------

ONE_SENTENCE_QUESTIONS: List[str] = [
    "Define {fn} in one sentence.",
    "Give me a brief one-line description of {fn}.",
    "Summarize {fn} in a single sentence.",
    "What is {fn} in plain English?",
]

# ---------------------------------------------------------------------------
# Category / function-type answers  (needs `category` field to be set)
# ---------------------------------------------------------------------------

CATEGORY_QUESTIONS: List[str] = [
    "What category of SQL function is {fn}?",
    "What type of SQL function is {fn}?",
    "Is {fn} an analytic function, aggregate function, or something else?",
    "How is {fn} classified in SQL?",
]

# ---------------------------------------------------------------------------
# Use-case answers  (answers with use_cases list)
# ---------------------------------------------------------------------------

USE_CASE_QUESTIONS: List[str] = [
    "What are the use cases for {fn}?",
    "What problems can {fn} solve?",
    "In what scenarios is {fn} most useful?",
    "When should I use {fn}?",
    "What type of analysis does {fn} support?",
    "What kinds of questions can {fn} answer?",
    "Give me some examples of when to use {fn}.",
    "What business problems is {fn} designed for?",
    "What are real-world applications of {fn}?",
]

# ---------------------------------------------------------------------------
# Parameter answers  (answers with full parameters text)
# ---------------------------------------------------------------------------

PARAMETER_QUESTIONS: List[str] = [
    "What are the parameters for {fn}?",
    "What inputs does {fn} require?",
    "What configuration does {fn} need?",
    "List the parameters accepted by {fn}.",
    "What do I need to specify to call {fn}?",
]

# ---------------------------------------------------------------------------
# Syntax answers
# ---------------------------------------------------------------------------

SYNTAX_QUESTIONS: List[str] = [
    "What is the syntax for {fn}?",
    "What are the required vs optional clauses in {fn}?",
    "Which keyword is mandatory in {fn} syntax?",
    "What is the minimum valid {fn} expression?",
    "How do you write a {fn} expression in SQL?",
    "Show me the {fn} SQL syntax.",
    "What does a basic {fn} query look like?",
    "What clauses does {fn} use?",
]

# ---------------------------------------------------------------------------
# Argument/parameter answers  (from parsed manual sections)
# ---------------------------------------------------------------------------

ARGUMENT_QUESTIONS: List[str] = [
    "What are the arguments to {fn}?",
    "What parameters does {fn} accept?",
    "Which arguments to {fn} are required?",
    "What is the default behavior when an optional argument is omitted from {fn}?",
    "What data type must be passed to {fn}?",
    "Can {fn} accept NULL values in its arguments?",
    "What happens if you pass invalid input to {fn}?",
    "How many arguments does {fn} require at minimum?",
]

# ---------------------------------------------------------------------------
# Notes / restrictions answers
# ---------------------------------------------------------------------------

NOTES_QUESTIONS: List[str] = [
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
]

# ---------------------------------------------------------------------------
# Example answers  (answered with SQL / expected output)
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS: List[str] = [
    "Show me a complete example of {fn} with input and output.",
    "What is the purpose of this {fn} example?",
    "How would you modify this {fn} example to change the output?",
    "What are the key SQL clauses in this {fn} example?",
    "Explain the SQL logic in this {fn} example.",
    "What would happen if you changed the ORDER BY in this {fn} example?",
]
