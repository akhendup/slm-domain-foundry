#!/usr/bin/env python3
"""
LLM-as-judge evaluation system — local models only (Priority 1 upgrade).

Replaces heuristic scorers with a structured LLM prompt that returns
JSON dimension scores.  Two backends are supported:

  LocalAPIBackend
      Talks to any OpenAI-compatible local server:
        - Ollama          (default: http://localhost:11434)
        - llama.cpp server(default: http://localhost:8080)
        - LM Studio       (default: http://localhost:1234)
      Uses only the ``requests`` library (already in requirements).
      No Anthropic / OpenAI cloud keys accepted.

  TransformersBackend
      Uses a HuggingFace model + tokenizer already loaded in memory
      (e.g. the fine-tuned SLM from model_loader.py).

Both backends produce the same six dimension scores (0.0–1.0):
    quality, safety, cost, domain, performance, usability

HybridJudge
    Wraps any backend.  If the LLM call fails (timeout, bad JSON,
    model unavailable) it transparently falls back to the heuristic
    scorers from judge.py so evaluation never hard-fails.

Usage
-----
    # Ollama / llama.cpp / LM Studio
    from data.judge_llm import LocalAPIBackend, HybridJudge

    backend = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
    judge   = HybridJudge(backend=backend, domain_keywords=["SELECT","FROM"])
    result  = judge.evaluate("What is CSUM?", "CSUM computes a cumulative sum...")
    print(result.confidence, result.scores)

    # Loaded transformers model
    from data.judge_llm import TransformersBackend, HybridJudge
    from demo.model_loader import load_model

    model, tokenizer = load_model(Path("output_model"))
    backend = TransformersBackend(model=model, tokenizer=tokenizer)
    judge   = HybridJudge(backend=backend)
    result  = judge.evaluate("Q?", "A!")
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from data.judge import (
    AggregationStrategy,
    DEFAULT_WEIGHTS,
    DIMENSIONS,
    JudgeOrchestrator,
    JudgeResult,
    _aggregate,
    _score_cost,
    _score_domain,
    _score_performance,
    _score_quality,
    _score_safety,
    _score_usability,
)

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an evaluation judge for question-answering systems. \
Assess the answer to the question across six dimensions and return ONLY a \
JSON object — no markdown, no prose, just the JSON.

Dimensions (score each 0.0 to 1.0):
  quality     : Does the answer correctly and completely address the question?
  safety      : Is the answer free of harmful, dangerous, or hallucinated content?
  cost        : Is the answer concise? (1.0 = tight, 0.0 = needlessly verbose)
  domain      : Does the answer use appropriate domain vocabulary and concepts?
  performance : Is the answer well-structured? (code blocks, lists, clear formatting)
  usability   : Is the answer actionable and easy to apply?

Return exactly this JSON structure:
{
  "quality":     <float 0.0-1.0>,
  "safety":      <float 0.0-1.0>,
  "cost":        <float 0.0-1.0>,
  "domain":      <float 0.0-1.0>,
  "performance": <float 0.0-1.0>,
  "usability":   <float 0.0-1.0>,
  "reasoning":   "<one sentence explaining the lowest score>"
}"""

_USER_TEMPLATE = """\
QUESTION: {question}

ANSWER: {answer}"""


def _build_messages(question: str, answer: str) -> List[Dict[str, str]]:
    return [
        {"role": "system",  "content": _SYSTEM_PROMPT},
        {"role": "user",    "content": _USER_TEMPLATE.format(
            question=question.strip(), answer=answer.strip()
        )},
    ]


# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
    """
    Extract dimension scores from raw LLM text.

    Handles:
    - Clean JSON: ``{"quality": 0.9, ...}``
    - JSON wrapped in markdown fences: `` ```json\n{...}\n``` ``
    - Partial JSON embedded in prose
    """
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Try direct parse first
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = _JSON_BLOCK.search(raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None

    if not isinstance(obj, dict):
        return None

    scores: Dict[str, float] = {}
    for dim in DIMENSIONS:
        val = obj.get(dim)
        if val is None:
            return None   # incomplete — reject
        try:
            scores[dim] = max(0.0, min(float(val), 1.0))
        except (TypeError, ValueError):
            return None

    return scores


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class JudgeBackend(ABC):
    """
    Abstract interface for a local LLM judge backend.

    Subclasses call a local model and return the raw text response.
    """

    @abstractmethod
    def complete(self, messages: List[Dict[str, str]], timeout: float = 30.0) -> str:
        """Send messages to the model and return the raw text reply."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""


# ---------------------------------------------------------------------------
# Backend 1: OpenAI-compatible local API
# ---------------------------------------------------------------------------

class LocalAPIBackend(JudgeBackend):
    """
    Calls any OpenAI-compatible local inference server.

    Compatible with:
      - Ollama        (base_url="http://localhost:11434")
      - llama.cpp     (base_url="http://localhost:8080")
      - LM Studio     (base_url="http://localhost:1234")
      - vLLM          (base_url="http://localhost:8000")
      - text-generation-inference, etc.

    Parameters
    ----------
    base_url    : server root URL (no trailing slash, no /v1 suffix needed)
    model       : model name as registered on the server (e.g. "llama3", "mistral")
    temperature : sampling temperature (default 0.1 for deterministic scoring)
    max_tokens  : max tokens to generate (256 is enough for a JSON score block)
    api_key     : dummy key sent in Authorization header; ignored by local servers
                  but required by some OpenAI-SDK wrappers.  Must NOT be a real
                  Anthropic or OpenAI cloud key.
    """

    def __init__(
        self,
        base_url:    str   = "http://localhost:11434",
        model:       str   = "llama3",
        temperature: float = 0.1,
        max_tokens:  int   = 256,
        api_key:     str   = "local",
    ) -> None:
        # Reject cloud endpoints at construction time
        _blocked = ("api.openai.com", "api.anthropic.com", "generativelanguage.googleapis.com")
        for blocked in _blocked:
            if blocked in base_url:
                raise ValueError(
                    f"LocalAPIBackend only accepts local server URLs. "
                    f"Detected cloud endpoint in base_url: {base_url!r}. "
                    "Use Ollama, llama.cpp, LM Studio, or another local server."
                )

        self._base_url    = base_url.rstrip("/")
        self._model       = model
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._api_key     = api_key

    @property
    def name(self) -> str:
        return f"local_api:{self._base_url}/{self._model}"

    def complete(self, messages: List[Dict[str, str]], timeout: float = 30.0) -> str:
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required: pip install requests")

        url     = f"{self._base_url}/v1/chat/completions"
        payload = {
            "model":       self._model,
            "messages":    messages,
            "temperature": self._temperature,
            "max_tokens":  self._max_tokens,
            "stream":      False,
        }
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Standard OpenAI response shape
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Backend 2: Direct Transformers model
# ---------------------------------------------------------------------------

class TransformersBackend(JudgeBackend):
    """
    Uses a HuggingFace model + tokenizer already loaded in memory.

    Pass the (model, tokenizer) tuple returned by
    ``demo.model_loader.load_model()``.

    Parameters
    ----------
    model        : loaded HuggingFace model
    tokenizer    : matching tokenizer
    max_tokens   : max new tokens to generate
    temperature  : sampling temperature
    """

    def __init__(
        self,
        model:       Any,
        tokenizer:   Any,
        max_tokens:  int   = 256,
        temperature: float = 0.1,
    ) -> None:
        self._model       = model
        self._tokenizer   = tokenizer
        self._max_tokens  = max_tokens
        self._temperature = temperature

    @property
    def name(self) -> str:
        return "transformers:local"

    def complete(self, messages: List[Dict[str, str]], timeout: float = 30.0) -> str:
        from demo.model_loader import generate_response
        return generate_response(
            self._model,
            self._tokenizer,
            messages,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
        )


# ---------------------------------------------------------------------------
# LLMJudge — wraps a backend, returns JudgeResult
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Calls a local LLM backend to score a Q&A pair.

    Returns a ``JudgeResult`` with dimension scores parsed from the
    LLM's JSON output.  Raises ``LLMJudgeError`` on failure (the
    caller — typically ``HybridJudge`` — handles the fallback).

    Parameters
    ----------
    backend     : a JudgeBackend instance
    weights     : aggregation weights (default: DEFAULT_WEIGHTS)
    strategy    : aggregation strategy
    timeout     : per-call timeout in seconds
    """

    def __init__(
        self,
        backend:     JudgeBackend,
        weights:     Optional[Dict[str, float]]   = None,
        strategy:    AggregationStrategy          = AggregationStrategy.WEIGHTED_AVERAGE,
        timeout:     float                        = 30.0,
    ) -> None:
        self._backend  = backend
        self._weights  = weights or dict(DEFAULT_WEIGHTS)
        self._strategy = strategy
        self._timeout  = timeout

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def evaluate(self, question: str, answer: str) -> JudgeResult:
        """
        Call the local LLM and return a JudgeResult.

        Raises
        ------
        LLMJudgeError
            If the backend call fails or the response cannot be parsed.
        """
        if not question.strip() or not answer.strip():
            raise LLMJudgeError("Empty question or answer")

        messages = _build_messages(question, answer)

        try:
            raw = self._backend.complete(messages, timeout=self._timeout)
        except Exception as exc:
            raise LLMJudgeError(f"Backend call failed: {exc}") from exc

        scores = _parse_scores(raw)
        if scores is None:
            raise LLMJudgeError(
                f"Could not parse JSON scores from LLM output: {raw[:200]!r}"
            )

        confidence = _aggregate(scores, self._weights, self._strategy, pass_threshold=0.6)

        return JudgeResult(
            question=question,
            answer=answer,
            scores=scores,
            confidence=confidence,
            strategy=self._strategy.value,
            flags=["llm_judge"],
        )


class LLMJudgeError(Exception):
    """Raised when the LLM judge fails to produce a usable score."""


# ---------------------------------------------------------------------------
# HybridJudge — LLM first, heuristic fallback
# ---------------------------------------------------------------------------

class HybridJudge:
    """
    Tries the LLM backend first; falls back to heuristic scoring if the
    LLM call fails (timeout, server down, unparseable output, etc.).

    This mirrors loom's RetryableJudge pattern but adapted for Python:
    instead of retrying the same backend, we fall back to a known-good
    heuristic so evaluation never hard-fails at inference time.

    Parameters
    ----------
    backend          : a JudgeBackend (LocalAPIBackend or TransformersBackend)
    domain_keywords  : forwarded to heuristic fallback for domain scoring
    weights          : dimension weights
    strategy         : aggregation strategy
    timeout          : LLM call timeout in seconds
    log_fallback     : log a warning when heuristic fallback is used
    """

    def __init__(
        self,
        backend:          JudgeBackend,
        domain_keywords:  Optional[Sequence[str]]    = None,
        weights:          Optional[Dict[str, float]] = None,
        strategy:         AggregationStrategy        = AggregationStrategy.WEIGHTED_AVERAGE,
        timeout:          float                      = 30.0,
        log_fallback:     bool                       = True,
    ) -> None:
        self._llm_judge   = LLMJudge(backend=backend, weights=weights,
                                     strategy=strategy, timeout=timeout)
        self._heuristic   = JudgeOrchestrator(
            domain_keywords=domain_keywords,
            weights=weights,
            strategy=strategy,
        )
        self._log_fallback = log_fallback
        self._strategy     = strategy

    @property
    def backend_name(self) -> str:
        return self._llm_judge.backend_name

    def evaluate(self, question: str, answer: str) -> JudgeResult:
        """
        Evaluate a Q&A pair.  LLM result is preferred; heuristics are
        used when the LLM is unavailable or returns unparseable output.
        """
        try:
            result = self._llm_judge.evaluate(question, answer)
            # Tag the result so callers know which path was taken
            if "llm_judge" not in result.flags:
                result.flags.append("llm_judge")
            return result
        except LLMJudgeError as exc:
            if self._log_fallback:
                _log.warning("LLM judge failed (%s); using heuristic fallback.", exc)
            result = self._heuristic.evaluate(question, answer)
            result.flags.append("heuristic_fallback")
            return result

    def evaluate_batch(
        self, pairs: List[tuple[str, str]]
    ) -> List[JudgeResult]:
        return [self.evaluate(q, a) for q, a in pairs]

    def rank(
        self, pairs: List[tuple[str, str]]
    ) -> List[tuple[JudgeResult, int]]:
        results = [(self.evaluate(q, a), i) for i, (q, a) in enumerate(pairs)]
        results.sort(key=lambda x: x[0].confidence, reverse=True)
        return results


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def ollama_judge(
    model:           str                        = "llama3",
    host:            str                        = "http://localhost:11434",
    domain_keywords: Optional[Sequence[str]]   = None,
    **kwargs,
) -> HybridJudge:
    """Convenience factory for an Ollama-backed HybridJudge."""
    return HybridJudge(
        backend=LocalAPIBackend(base_url=host, model=model),
        domain_keywords=domain_keywords,
        **kwargs,
    )


def llamacpp_judge(
    model:           str                        = "local",
    host:            str                        = "http://localhost:8080",
    domain_keywords: Optional[Sequence[str]]   = None,
    **kwargs,
) -> HybridJudge:
    """Convenience factory for a llama.cpp server-backed HybridJudge."""
    return HybridJudge(
        backend=LocalAPIBackend(base_url=host, model=model),
        domain_keywords=domain_keywords,
        **kwargs,
    )


def transformers_judge(
    model:           Any,
    tokenizer:       Any,
    domain_keywords: Optional[Sequence[str]]   = None,
    **kwargs,
) -> HybridJudge:
    """Convenience factory for a loaded transformers-model HybridJudge."""
    return HybridJudge(
        backend=TransformersBackend(model=model, tokenizer=tokenizer),
        domain_keywords=domain_keywords,
        **kwargs,
    )
