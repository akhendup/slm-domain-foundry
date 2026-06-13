"""Unit tests for data/yaml_pattern_loader.py"""
import pytest

from data.yaml_pattern_loader import (
    generate_multiturn_from_pattern,
    generate_qa_from_pattern,
    load_patterns_as_qa,
    load_yaml_patterns_dir,
)


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# load_yaml_patterns_dir
# ---------------------------------------------------------------------------

class TestLoadYamlPatternsDir:
    def test_loads_valid_yaml(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        patterns = load_yaml_patterns_dir(tmp_path)
        assert len(patterns) == 1
        assert patterns[0]["name"] == "hypertension"

    def test_skips_invalid_yaml(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "good.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        (tmp_path / "bad.yaml").write_text(":::invalid yaml:::", encoding="utf-8")
        patterns = load_yaml_patterns_dir(tmp_path)
        assert len(patterns) == 1

    def test_skips_yaml_without_name(self, tmp_path):
        (tmp_path / "noname.yaml").write_text(
            "title: Something\ndescription: A thing.\n", encoding="utf-8"
        )
        patterns = load_yaml_patterns_dir(tmp_path)
        assert len(patterns) == 0

    def test_recursive_search(self, tmp_path, sample_yaml_pattern):
        subdir = tmp_path / "cardiology"
        subdir.mkdir()
        (subdir / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        patterns = load_yaml_patterns_dir(tmp_path)
        assert len(patterns) == 1

    def test_empty_dir_returns_empty(self, tmp_path):
        assert load_yaml_patterns_dir(tmp_path) == []

    def test_source_file_annotated(self, tmp_path, sample_yaml_pattern):
        (tmp_path / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        patterns = load_yaml_patterns_dir(tmp_path)
        assert "_source_file" in patterns[0]


# ---------------------------------------------------------------------------
# generate_qa_from_pattern
# ---------------------------------------------------------------------------

class TestGenerateQaFromPattern:
    def test_returns_list_of_tuples(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_description_generates_qa(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        questions = [q for q, _ in result]
        assert any("Hypertension" in q or "hypertension" in q.lower() for q in questions)

    def test_use_cases_generate_qa(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        questions = [q.lower() for q, _ in result]
        assert any("use case" in q or "what" in q for q in questions)

    def test_parameters_generate_qa(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        questions = [q.lower() for q, _ in result]
        assert any("parameter" in q or "input" in q for q in questions)

    def test_templates_generate_qa(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        answers = [a for _, a in result]
        assert any("Treatment plan" in a or "Case" in a for a in answers)

    def test_best_practices_generate_qa(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        questions = [q.lower() for q, _ in result]
        assert any("best practice" in q or "tip" in q or "advice" in q for q in questions)

    def test_empty_pattern_returns_empty(self):
        result = generate_qa_from_pattern({})
        assert result == []

    def test_all_answers_nonempty(self, sample_pattern_dict):
        for _, a in generate_qa_from_pattern(sample_pattern_dict):
            assert a.strip() != ""

    def test_all_questions_nonempty(self, sample_pattern_dict):
        for q, _ in generate_qa_from_pattern(sample_pattern_dict):
            assert q.strip() != ""

    def test_minimum_qa_count(self, sample_pattern_dict):
        result = generate_qa_from_pattern(sample_pattern_dict)
        assert len(result) >= 5


# ---------------------------------------------------------------------------
# generate_multiturn_from_pattern
# ---------------------------------------------------------------------------

class TestGenerateMultiturnFromPattern:
    def test_returns_list_or_none(self, sample_pattern_dict):
        result = generate_multiturn_from_pattern(sample_pattern_dict)
        assert result is None or isinstance(result, list)

    def test_turns_have_role_and_content(self, sample_pattern_dict):
        result = generate_multiturn_from_pattern(sample_pattern_dict)
        if result is None:
            pytest.skip("Pattern did not produce enough turns")
        for turn in result:
            assert "role" in turn
            assert "content" in turn

    def test_turns_alternate_user_assistant(self, sample_pattern_dict):
        result = generate_multiturn_from_pattern(sample_pattern_dict)
        if result is None:
            pytest.skip("Pattern did not produce enough turns")
        for i, turn in enumerate(result):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert turn["role"] == expected_role

    def test_minimum_four_turns(self, sample_pattern_dict):
        result = generate_multiturn_from_pattern(sample_pattern_dict)
        if result is None:
            pytest.skip("Pattern did not produce enough content")
        assert len(result) >= 4


# ---------------------------------------------------------------------------
# load_patterns_as_qa
# ---------------------------------------------------------------------------

class TestLoadPatternsAsQa:
    def test_loads_and_generates_qa(self, tmp_path, sample_yaml_pattern):
        """load_patterns_as_qa returns (qa_pairs, multiturn_convs)."""
        (tmp_path / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        qa_pairs, multiturn = load_patterns_as_qa(tmp_path)
        assert len(qa_pairs) >= 1
        for q, a in qa_pairs:
            assert isinstance(q, str) and q.strip()
            assert isinstance(a, str) and a.strip()

    def test_empty_dir_returns_empty_tuple(self, tmp_path):
        qa_pairs, multiturn = load_patterns_as_qa(tmp_path)
        assert qa_pairs == []
        assert multiturn == []
