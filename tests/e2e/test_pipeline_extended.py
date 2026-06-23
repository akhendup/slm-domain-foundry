"""
Extended end-to-end pipeline tests.
Covers: prepare_training_data main() function branches not exercised by the primary tests.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "data.prepare_training_data", *args],
        capture_output=True, text=True,
        cwd=_PROJECT_ROOT,
    )


# ---------------------------------------------------------------------------
# main() — format both (default)
# ---------------------------------------------------------------------------

class TestCliFormatBoth:
    def test_both_format_produces_alpaca_and_sharegpt(self, tmp_path, tmp_csv):
        csv_path = tmp_csv([
            {"question": "What is hypertension?", "answer": "Hypertension computes a cumulative sum."},
            {"question": "What is RANK?", "answer": "RANK assigns a rank."},
            {"question": "What is MSUM?", "answer": "MSUM computes a moving sum."},
        ])
        result = _run_cli(
            "--csv", str(csv_path),
            "--output-dir", str(tmp_path / "out"),
            "--format", "both",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        out_dir = tmp_path / "out"
        assert (out_dir / "train_alpaca.jsonl").exists()
        assert (out_dir / "train_sharegpt.jsonl").exists()
        assert (out_dir / "val_alpaca.jsonl").exists()
        assert (out_dir / "val_sharegpt.jsonl").exists()

    def test_alpaca_format_only(self, tmp_path, tmp_csv):
        csv_path = tmp_csv([
            {"question": f"Q{i}?", "answer": f"A{i}."} for i in range(5)
        ])
        result = _run_cli(
            "--csv", str(csv_path),
            "--output-dir", str(tmp_path / "out"),
            "--format", "alpaca",
        )
        assert result.returncode == 0
        out_dir = tmp_path / "out"
        assert (out_dir / "train_alpaca.jsonl").exists()
        # sharegpt should NOT be written
        assert not (out_dir / "train_sharegpt.jsonl").exists()


# ---------------------------------------------------------------------------
# main() — no-multiturn flag
# ---------------------------------------------------------------------------

class TestCliNoMultiturn:
    def test_no_multiturn_flag(self, tmp_path, sample_yaml_pattern):
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        result = _run_cli(
            "--yaml-dir", str(pattern_dir),
            "--output-dir", str(tmp_path / "out"),
            "--format", "sharegpt",
            "--no-multiturn",
        )
        assert result.returncode == 0
        out_dir = tmp_path / "out"
        # multiturn file should not be created when --no-multiturn
        assert not (out_dir / "train_multiturn.jsonl").exists()


# ---------------------------------------------------------------------------
# main() — with memory-dir
# ---------------------------------------------------------------------------

class TestCliWithMemoryDir:
    def test_approved_interactions_included(self, tmp_path, tmp_csv):
        """Approved interactions from memory dir are included in output."""
        from data.conversation_memory import log_interaction, set_approval

        mem_dir = tmp_path / "memory"
        rid1 = log_interaction(mem_dir, "Memory Q1?", "Memory A1.")
        rid2 = log_interaction(mem_dir, "Memory Q2?", "Memory A2.")
        log_interaction(mem_dir, "Pending Q?", "Pending A.")  # not approved
        set_approval(mem_dir, rid1, True)
        set_approval(mem_dir, rid2, True)

        result = _run_cli(
            "--memory-dir", str(mem_dir),
            "--output-dir", str(tmp_path / "out"),
            "--format", "sharegpt",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        out_dir = tmp_path / "out"
        assert any(out_dir.glob("*.jsonl"))

    def test_no_approved_with_memory_dir_fails(self, tmp_path):
        """If only memory dir is provided but no approved records, exit with error."""
        mem_dir = tmp_path / "memory"
        from data.conversation_memory import log_interaction
        log_interaction(mem_dir, "Q?", "A.")  # not approved

        result = _run_cli(
            "--memory-dir", str(mem_dir),
            "--output-dir", str(tmp_path / "out"),
            "--format", "sharegpt",
        )
        # Should fail with no training examples
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# main() — no arguments
# ---------------------------------------------------------------------------

class TestCliNoArgs:
    def test_no_args_fails_with_message(self):
        result = _run_cli("--output-dir", "/tmp/no_source_out")
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "No training examples" in combined or "provide" in combined.lower()


# ---------------------------------------------------------------------------
# main() — val-ratio and seed
# ---------------------------------------------------------------------------

class TestCliValRatioAndSeed:
    def test_custom_val_ratio(self, tmp_path, tmp_csv):
        rows = [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(20)]
        csv_path = tmp_csv(rows)
        result = _run_cli(
            "--csv", str(csv_path),
            "--output-dir", str(tmp_path / "out"),
            "--format", "sharegpt",
            "--val-ratio", "0.2",
        )
        assert result.returncode == 0
        out_dir = tmp_path / "out"
        train_lines = [
            json.loads(ln)
            for ln in (out_dir / "train_sharegpt.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        val_lines = [
            json.loads(ln)
            for ln in (out_dir / "val_sharegpt.jsonl").read_text().splitlines()
            if ln.strip()
        ]
        total = len(train_lines) + len(val_lines)
        assert total == 20
        # val should be ~4 (20% of 20)
        assert len(val_lines) == 4

    def test_seed_deterministic(self, tmp_path, tmp_csv):
        """Same seed → same output ordering."""
        rows = [{"question": f"Q{i}?", "answer": f"A{i}."} for i in range(10)]

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        for out_dir in (out1, out2):
            csv_path = tmp_csv(rows, f"data_{out_dir.name}.csv")
            _run_cli(
                "--csv", str(csv_path),
                "--output-dir", str(out_dir),
                "--format", "sharegpt",
                "--seed", "42",
            )

        lines1 = (out1 / "train_sharegpt.jsonl").read_text().splitlines()
        lines2 = (out2 / "train_sharegpt.jsonl").read_text().splitlines()
        # Same seed, same data → same result
        assert lines1 == lines2


# ---------------------------------------------------------------------------
# main() — yaml + memory combined
# ---------------------------------------------------------------------------

class TestCliCombinedSources:
    def test_yaml_plus_memory_combined(self, tmp_path, sample_yaml_pattern):
        from data.conversation_memory import log_interaction, set_approval

        # YAML patterns
        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")

        # Memory with approved records
        mem_dir = tmp_path / "memory"
        rid = log_interaction(mem_dir, "Combined Q?", "Combined A.")
        set_approval(mem_dir, rid, True)

        result = _run_cli(
            "--yaml-dir", str(pattern_dir),
            "--memory-dir", str(mem_dir),
            "--output-dir", str(tmp_path / "out"),
            "--format", "sharegpt",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        out_dir = tmp_path / "out"
        train = out_dir / "train_sharegpt.jsonl"
        assert train.exists()
        lines = [json.loads(ln) for ln in train.read_text().splitlines() if ln.strip()]
        assert len(lines) >= 1


# ---------------------------------------------------------------------------
# prepare_training_data.main() direct call
# ---------------------------------------------------------------------------

class TestMainDirectCall:
    """Call main() directly (not via subprocess) to maximize coverage of the function body."""

    def test_csv_direct_call(self, tmp_path, tmp_csv):
        import sys as _sys
        from data.prepare_training_data import main

        rows = [{"question": f"Direct Q{i}?", "answer": f"Direct A{i}."} for i in range(8)]
        csv_path = tmp_csv(rows)
        out_dir = tmp_path / "direct_out"

        _sys.argv = [
            "prepare_training_data",
            "--csv", str(csv_path),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]
        rc = main()
        assert rc == 0
        assert (out_dir / "train_sharegpt.jsonl").exists()

    def test_yaml_direct_call(self, tmp_path, sample_yaml_pattern):
        import sys as _sys
        from data.prepare_training_data import main

        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        out_dir = tmp_path / "yaml_out"

        _sys.argv = [
            "prepare_training_data",
            "--yaml-dir", str(pattern_dir),
            "--output-dir", str(out_dir),
            "--format", "both",
        ]
        rc = main()
        assert rc == 0
        assert (out_dir / "train_sharegpt.jsonl").exists()
        assert (out_dir / "train_alpaca.jsonl").exists()

    def test_no_source_returns_error(self, tmp_path):
        import sys as _sys
        from data.prepare_training_data import main

        _sys.argv = [
            "prepare_training_data",
            "--output-dir", str(tmp_path / "empty"),
        ]
        rc = main()
        assert rc == 1

    def test_csv_text_column_pipeline(self, tmp_path, tmp_csv):
        """CSV with 'text' column (not Q&A) runs through chunking + heuristic."""
        import sys as _sys
        from data.prepare_training_data import main

        rows = [
            {"text": "Hypertension Overview\n\nHypertension management combines lifestyle and medication."},
            {"text": "The Hypertension function computes a cumulative sum over the specified window."},
        ]
        csv_path = tmp_csv(rows)
        out_dir = tmp_path / "text_out"

        _sys.argv = [
            "prepare_training_data",
            "--csv", str(csv_path),
            "--output-dir", str(out_dir),
            "--format", "alpaca",
        ]
        rc = main()
        assert rc == 0

    def test_yaml_with_no_multiturn(self, tmp_path, sample_yaml_pattern):
        import sys as _sys
        from data.prepare_training_data import main

        pattern_dir = tmp_path / "patterns"
        pattern_dir.mkdir()
        (pattern_dir / "hypertension.yaml").write_text(sample_yaml_pattern, encoding="utf-8")
        out_dir = tmp_path / "out_no_mt"

        _sys.argv = [
            "prepare_training_data",
            "--yaml-dir", str(pattern_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
            "--no-multiturn",
        ]
        rc = main()
        assert rc == 0
        assert not (out_dir / "train_multiturn.jsonl").exists()

    def test_memory_dir_direct_call(self, tmp_path):
        """Direct main() call with --memory-dir exercises lines 318-342."""
        import sys as _sys
        from data.prepare_training_data import main
        from data.conversation_memory import log_interaction, set_approval

        mem_dir = tmp_path / "memory"
        rid1 = log_interaction(mem_dir, "Direct Mem Q1?", "Direct Mem A1.")
        rid2 = log_interaction(mem_dir, "Direct Mem Q2?", "Direct Mem A2.")
        log_interaction(mem_dir, "Unapproved Q?", "Unapproved A.")
        set_approval(mem_dir, rid1, True)
        set_approval(mem_dir, rid2, True)

        out_dir = tmp_path / "mem_out"
        _sys.argv = [
            "prepare_training_data",
            "--memory-dir", str(mem_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]
        rc = main()
        assert rc == 0
        assert (out_dir / "train_sharegpt.jsonl").exists()

    def test_memory_dir_with_malformed_lines(self, tmp_path):
        """Memory dir with malformed JSON lines is handled gracefully (lines 340-341)."""
        import sys as _sys
        import json as _json
        from data.prepare_training_data import main

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        jsonl = mem_dir / "interactions.jsonl"
        # Write one valid approved + one invalid line
        jsonl.write_text(
            'not valid json\n'
            + _json.dumps({"approved": True, "question": "Q?", "answer": "A."}) + "\n",
            encoding="utf-8",
        )

        out_dir = tmp_path / "mem_out"
        _sys.argv = [
            "prepare_training_data",
            "--memory-dir", str(mem_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]
        rc = main()
        assert rc == 0  # malformed line is skipped, valid line still produces output

    def test_memory_dir_empty_qa_skipped(self, tmp_path):
        """Approved records with empty question or answer are skipped."""
        import sys as _sys
        import json as _json
        from data.prepare_training_data import main

        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        jsonl = mem_dir / "interactions.jsonl"
        # Empty question → skipped; valid record → included
        jsonl.write_text(
            _json.dumps({"approved": True, "question": "", "answer": "A."}) + "\n"
            + _json.dumps({"approved": True, "question": "Q?", "answer": "Valid A."}) + "\n",
            encoding="utf-8",
        )

        out_dir = tmp_path / "mem_out"
        _sys.argv = [
            "prepare_training_data",
            "--memory-dir", str(mem_dir),
            "--output-dir", str(out_dir),
            "--format", "sharegpt",
        ]
        rc = main()
        assert rc == 0
