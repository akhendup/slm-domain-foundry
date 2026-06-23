"""
Unit tests for data/judge_llm.py — LLM-as-judge with local backends.

All LLM calls are mocked so no running model is required.
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from data.judge import AggregationStrategy, JudgeOrchestrator, JudgeResult
from data.judge_llm import (
    HybridJudge,
    LLMJudge,
    LLMJudgeError,
    LocalAPIBackend,
    TransformersBackend,
    _build_messages,
    _parse_scores,
    llamacpp_judge,
    ollama_judge,
    transformers_judge,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_JSON = json.dumps({
    "quality":     0.9,
    "safety":      1.0,
    "cost":        0.8,
    "domain":      0.7,
    "performance": 0.6,
    "usability":   0.75,
    "reasoning":   "Answer is accurate and well-structured.",
})

_GOOD_SCORES = {
    "quality": 0.9, "safety": 1.0, "cost": 0.8,
    "domain": 0.7, "performance": 0.6, "usability": 0.75,
}


def _mock_backend(response: str = _GOOD_JSON) -> MagicMock:
    b = MagicMock()
    b.complete.return_value = response
    b.name = "mock:local"
    return b


# ---------------------------------------------------------------------------
# _parse_scores
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_valid_json(self):
        scores = _parse_scores(_GOOD_JSON)
        assert scores is not None
        assert scores["quality"] == 0.9
        assert scores["safety"]  == 1.0

    def test_strips_markdown_fence(self):
        wrapped = f"```json\n{_GOOD_JSON}\n```"
        scores  = _parse_scores(wrapped)
        assert scores is not None
        assert scores["quality"] == 0.9

    def test_extracts_embedded_json(self):
        text = f"Here is my evaluation:\n{_GOOD_JSON}\nEnd."
        scores = _parse_scores(text)
        assert scores is not None

    def test_clamps_above_one(self):
        obj = {d: 1.5 for d in ["quality", "safety", "cost", "domain", "performance", "usability"]}
        scores = _parse_scores(json.dumps(obj))
        assert scores is not None
        assert all(v <= 1.0 for v in scores.values())

    def test_clamps_below_zero(self):
        obj = {d: -0.5 for d in ["quality", "safety", "cost", "domain", "performance", "usability"]}
        scores = _parse_scores(json.dumps(obj))
        assert scores is not None
        assert all(v >= 0.0 for v in scores.values())

    def test_missing_dimension_returns_none(self):
        obj = {"quality": 0.9, "safety": 1.0}   # incomplete
        assert _parse_scores(json.dumps(obj)) is None

    def test_empty_string_returns_none(self):
        assert _parse_scores("") is None

    def test_pure_prose_returns_none(self):
        assert _parse_scores("The answer was good overall.") is None

    def test_reasoning_field_ignored(self):
        scores = _parse_scores(_GOOD_JSON)
        assert "reasoning" not in scores

    def test_integer_values_accepted(self):
        obj = {d: 1 for d in ["quality", "safety", "cost", "domain", "performance", "usability"]}
        scores = _parse_scores(json.dumps(obj))
        assert scores is not None
        assert all(isinstance(v, float) for v in scores.values())


# ---------------------------------------------------------------------------
# _build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_two_messages(self):
        msgs = _build_messages("Q?", "A!")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_content_included(self):
        msgs = _build_messages("What is hypertension?", "Hypertension computes a cumulative sum.")
        assert "What is hypertension?" in msgs[1]["content"]
        assert "Hypertension computes" in msgs[1]["content"]

    def test_system_prompt_contains_dimensions(self):
        msgs = _build_messages("Q?", "A!")
        system = msgs[0]["content"]
        for dim in ("quality", "safety", "cost", "domain", "performance", "usability"):
            assert dim in system


# ---------------------------------------------------------------------------
# LocalAPIBackend
# ---------------------------------------------------------------------------

class TestLocalAPIBackend:
    def test_rejects_openai_cloud(self):
        with pytest.raises(ValueError, match="cloud endpoint"):
            LocalAPIBackend(base_url="https://api.openai.com/v1")

    def test_rejects_anthropic_cloud(self):
        with pytest.raises(ValueError, match="cloud endpoint"):
            LocalAPIBackend(base_url="https://api.anthropic.com")

    def test_accepts_localhost(self):
        b = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
        assert "localhost" in b.name

    def test_accepts_local_ip(self):
        b = LocalAPIBackend(base_url="http://192.168.1.10:8080", model="mistral")
        assert b.name  # no exception

    def test_name_contains_model(self):
        b = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
        assert "llama3" in b.name

    def test_complete_sends_correct_payload(self):
        b = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": _GOOD_JSON}}]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = b.complete([{"role": "user", "content": "Q?"}])
        assert result == _GOOD_JSON
        call_kwargs = mock_post.call_args
        payload = call_kwargs[1]["json"]
        assert payload["model"]    == "llama3"
        assert payload["stream"]   is False
        assert "messages"          in payload

    def test_complete_raises_on_http_error(self):
        b = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="404"):
                b.complete([{"role": "user", "content": "Q?"}])

    def test_complete_raises_on_connection_error(self):
        b = LocalAPIBackend(base_url="http://localhost:11434", model="llama3")
        with patch("requests.post", side_effect=ConnectionError("refused")):
            with pytest.raises(ConnectionError):
                b.complete([{"role": "user", "content": "Q?"}])


# ---------------------------------------------------------------------------
# TransformersBackend
# ---------------------------------------------------------------------------

class TestTransformersBackend:
    def test_name(self):
        b = TransformersBackend(model=MagicMock(), tokenizer=MagicMock())
        assert b.name == "transformers:local"

    def test_complete_calls_generate_response(self):
        b = TransformersBackend(model=MagicMock(), tokenizer=MagicMock())
        with patch("app.model_loader.generate_response", return_value=_GOOD_JSON) as mock_gen:
            result = b.complete([{"role": "user", "content": "Q?"}])
        mock_gen.assert_called_once()
        assert result == _GOOD_JSON

    def test_complete_passes_temperature(self):
        b = TransformersBackend(model=MagicMock(), tokenizer=MagicMock(), temperature=0.05)
        with patch("app.model_loader.generate_response", return_value=_GOOD_JSON) as mock_gen:
            b.complete([])
        _, kwargs = mock_gen.call_args
        assert kwargs.get("temperature") == 0.05


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class TestLLMJudge:
    def test_evaluate_returns_result(self):
        judge  = LLMJudge(backend=_mock_backend())
        result = judge.evaluate("What is hypertension?", "Hypertension computes cumulative sums.")
        assert isinstance(result, JudgeResult)
        assert result.scores == _GOOD_SCORES
        assert 0.0 <= result.confidence <= 1.0

    def test_flags_contain_llm_judge(self):
        judge  = LLMJudge(backend=_mock_backend())
        result = judge.evaluate("Q?", "A!")
        assert "llm_judge" in result.flags

    def test_backend_failure_raises_llm_judge_error(self):
        b = _mock_backend()
        b.complete.side_effect = ConnectionError("refused")
        judge = LLMJudge(backend=b)
        with pytest.raises(LLMJudgeError, match="Backend call failed"):
            judge.evaluate("Q?", "A!")

    def test_unparseable_response_raises(self):
        judge = LLMJudge(backend=_mock_backend("not json at all"))
        with pytest.raises(LLMJudgeError, match="parse"):
            judge.evaluate("Q?", "A!")

    def test_empty_question_raises(self):
        judge = LLMJudge(backend=_mock_backend())
        with pytest.raises(LLMJudgeError):
            judge.evaluate("", "Some answer")

    def test_empty_answer_raises(self):
        judge = LLMJudge(backend=_mock_backend())
        with pytest.raises(LLMJudgeError):
            judge.evaluate("Some question?", "")

    def test_strategy_applied(self):
        judge  = LLMJudge(backend=_mock_backend(),
                          strategy=AggregationStrategy.MIN_SCORE)
        result = judge.evaluate("Q?", "A!")
        assert result.confidence == min(_GOOD_SCORES.values())

    def test_confidence_in_range(self):
        judge  = LLMJudge(backend=_mock_backend())
        result = judge.evaluate("How does Hypertension work?",
                                "Confirm hypertension readings and start lifestyle counseling with monitoring.")
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# HybridJudge
# ---------------------------------------------------------------------------

class TestHybridJudge:
    def test_uses_llm_when_available(self):
        judge  = HybridJudge(backend=_mock_backend())
        result = judge.evaluate("Q?", "A!")
        assert "llm_judge" in result.flags
        assert "heuristic_fallback" not in result.flags

    def test_falls_back_on_connection_error(self):
        b = _mock_backend()
        b.complete.side_effect = ConnectionError("refused")
        judge  = HybridJudge(backend=b, log_fallback=False)
        result = judge.evaluate("What is hypertension?",
                                "Hypertension is sustained elevated blood pressure managed with lifestyle changes and medication.")
        assert "heuristic_fallback" in result.flags
        assert 0.0 <= result.confidence <= 1.0

    def test_falls_back_on_bad_json(self):
        judge  = HybridJudge(backend=_mock_backend("not json"), log_fallback=False)
        result = judge.evaluate("Q?", "A!")
        assert "heuristic_fallback" in result.flags

    def test_evaluate_batch(self):
        judge   = HybridJudge(backend=_mock_backend())
        pairs   = [("Q1?", "A1!"), ("Q2?", "A2!")]
        results = judge.evaluate_batch(pairs)
        assert len(results) == 2
        assert all(isinstance(r, JudgeResult) for r in results)

    def test_rank_sorted_descending(self):
        # First pair gets good JSON; second gets bad JSON → fallback with lower score
        call_count = {"n": 0}
        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _GOOD_JSON
            return "bad json"

        b = _mock_backend()
        b.complete.side_effect = _side_effect
        judge  = HybridJudge(backend=b, log_fallback=False)
        ranked = judge.rank([("Q1?", "A1 excellent answer with hypertension monitoring and aspirin guidance."),
                             ("Q2?", "")])
        assert ranked[0][0].confidence >= ranked[-1][0].confidence

    def test_domain_keywords_forwarded_to_fallback(self):
        b = _mock_backend()
        b.complete.side_effect = ConnectionError("refused")
        judge  = HybridJudge(backend=b, domain_keywords=["aspirin", "monitoring"],
                              log_fallback=False)
        result = judge.evaluate("How to manage hypertension?",
                                "Use aspirin 81 mg daily with blood pressure monitoring.")
        assert result.scores["domain"] == 1.0  # both keywords present

    def test_backend_name_exposed(self):
        judge = HybridJudge(backend=_mock_backend())
        assert judge.backend_name == "mock:local"

    def test_fallback_result_has_all_dimensions(self):
        b = _mock_backend()
        b.complete.side_effect = ConnectionError()
        judge  = HybridJudge(backend=b, log_fallback=False)
        result = judge.evaluate("Q?", "A answer.")
        from data.judge import DIMENSIONS
        assert set(result.scores.keys()) == set(DIMENSIONS)


# ---------------------------------------------------------------------------
# JudgeOrchestrator with llm_backend
# ---------------------------------------------------------------------------

class TestJudgeOrchestratorLLMIntegration:
    def test_uses_llm_backend_when_provided(self):
        orch   = JudgeOrchestrator(llm_backend=_mock_backend())
        result = orch.evaluate("Q?", "A!")
        assert "llm_judge" in result.flags

    def test_no_backend_uses_heuristics(self):
        orch   = JudgeOrchestrator()
        result = orch.evaluate("Q?", "A!")
        assert "llm_judge" not in result.flags

    def test_llm_failure_falls_back_silently(self):
        b = _mock_backend()
        b.complete.side_effect = ConnectionError()
        orch   = JudgeOrchestrator(llm_backend=b)
        result = orch.evaluate("Q?", "A!")
        assert "heuristic_fallback" in result.flags


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactoryHelpers:
    def test_ollama_judge_accepts_local_host(self):
        judge = ollama_judge(model="llama3", host="http://localhost:11434")
        assert isinstance(judge, HybridJudge)
        assert "llama3" in judge.backend_name

    def test_llamacpp_judge(self):
        judge = llamacpp_judge(model="local", host="http://localhost:8080")
        assert isinstance(judge, HybridJudge)

    def test_transformers_judge(self):
        judge = transformers_judge(model=MagicMock(), tokenizer=MagicMock())
        assert isinstance(judge, HybridJudge)
        assert judge.backend_name == "transformers:local"

    def test_ollama_judge_rejects_cloud(self):
        with pytest.raises(ValueError, match="cloud endpoint"):
            ollama_judge(host="https://api.openai.com/v1")
