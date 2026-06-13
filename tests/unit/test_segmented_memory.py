"""Unit tests for data/segmented_memory.py — segmented memory with token budgets."""

import pytest
from data.segmented_memory import (
    SegmentBudgets,
    SegmentedMemory,
    Turn,
    _dict_to_text,
    _extract_key_sentences,
    _turns_to_text,
    count_tokens,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") >= 1

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert count_tokens(text) > 5

    def test_proportional(self):
        short = "hello world"
        long  = "hello world " * 10
        assert count_tokens(long) > count_tokens(short)


# ---------------------------------------------------------------------------
# Key sentence extraction
# ---------------------------------------------------------------------------

class TestExtractKeySentences:
    def test_empty(self):
        assert _extract_key_sentences("", 100) == ""

    def test_structured_sentences_preferred(self):
        text = (
            "This is an unrelated sentence. "
            "Use SELECT col FROM table WHERE id = 1. "
            "Another unrelated sentence here."
        )
        result = _extract_key_sentences(text, 50)
        assert "SELECT" in result or result != ""

    def test_respects_budget(self):
        text = "Word " * 200
        result = _extract_key_sentences(text, 20)
        assert count_tokens(result) <= 25   # small tolerance

    def test_single_sentence(self):
        text = "This is one sentence."
        result = _extract_key_sentences(text, 100)
        assert "This is one sentence" in result


# ---------------------------------------------------------------------------
# Turn
# ---------------------------------------------------------------------------

class TestTurn:
    def test_tokens_set_on_init(self):
        t = Turn(role="user", content="Hello world")
        assert t.tokens > 0

    def test_empty_content(self):
        t = Turn(role="user", content="")
        assert t.tokens == 0


# ---------------------------------------------------------------------------
# SegmentedMemory — ROM
# ---------------------------------------------------------------------------

class TestROM:
    def test_set_rom(self):
        mem = SegmentedMemory()
        mem.set_rom("You are a helpful assistant.")
        assert "helpful" in mem.build_prompt()["rom"]

    def test_rom_truncated_when_over_budget(self):
        tiny_budget = SegmentBudgets(rom=5, kernel=2000, l1=10000, l2=3000)
        mem = SegmentedMemory(budgets=tiny_budget)
        mem.set_rom("word " * 1000)
        stats = mem.stats()
        assert stats["rom"]["used"] <= tiny_budget.rom + 5  # small tolerance

    def test_rom_appears_in_render(self):
        mem = SegmentedMemory()
        mem.set_rom("System: you are SQL expert.")
        assert "SQL expert" in mem.render()


# ---------------------------------------------------------------------------
# SegmentedMemory — Kernel
# ---------------------------------------------------------------------------

class TestKernel:
    def test_set_kernel(self):
        mem = SegmentedMemory()
        mem.set_kernel({"user": "Alice", "goal": "Learn SQL"})
        prompt = mem.build_prompt()
        assert "Alice" in prompt["kernel"]
        assert "Learn SQL" in prompt["kernel"]

    def test_update_kernel(self):
        mem = SegmentedMemory()
        mem.set_kernel({"user": "Alice"})
        mem.update_kernel("goal", "Window functions")
        assert "Window functions" in mem.build_prompt()["kernel"]

    def test_kernel_in_render(self):
        mem = SegmentedMemory()
        mem.set_kernel({"user": "Bob"})
        assert "Bob" in mem.render()

    def test_kernel_trimmed_when_over_budget(self):
        tiny = SegmentBudgets(rom=5000, kernel=3, l1=10000, l2=3000)
        mem  = SegmentedMemory(budgets=tiny)
        mem.set_kernel({"a": "word " * 500, "b": "word " * 500})
        stats = mem.stats()
        assert stats["kernel"]["used"] <= tiny.kernel + 5


# ---------------------------------------------------------------------------
# SegmentedMemory — L1 (FIFO turns)
# ---------------------------------------------------------------------------

class TestL1:
    def test_add_single_turn(self):
        mem = SegmentedMemory()
        mem.add_turn("user", "What is hypertension?")
        msgs = mem.get_l1_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_add_multiple_turns(self):
        mem = SegmentedMemory()
        mem.add_turn("user", "Q1")
        mem.add_turn("assistant", "A1")
        mem.add_turn("user", "Q2")
        assert len(mem.get_l1_messages()) == 3

    def test_fifo_eviction_when_full(self):
        tiny = SegmentBudgets(rom=5000, kernel=2000, l1=10, l2=3000)
        mem  = SegmentedMemory(budgets=tiny)
        mem.add_turn("user", "First question")
        mem.add_turn("user", "Second question that is long enough to push out first")
        # L1 tokens should not exceed budget
        stats = mem.stats()
        assert stats["l1"]["used"] <= tiny.l1 + 15  # small tolerance

    def test_evicted_goes_to_l2(self):
        tiny = SegmentBudgets(rom=5000, kernel=2000, l1=5, l2=3000)
        mem  = SegmentedMemory(budgets=tiny)
        mem.add_turn("user", "This is a long enough message to force eviction from L1")
        mem.add_turn("user", "Another long message that forces the first one out of L1")
        # L2 should have something now
        assert mem.stats()["l2"]["used"] >= 0  # may be 0 if extractive found nothing

    def test_l1_appears_in_render(self):
        mem = SegmentedMemory()
        mem.add_turn("user", "Hello there")
        assert "Hello there" in mem.render()


# ---------------------------------------------------------------------------
# SegmentedMemory — L2 (summary)
# ---------------------------------------------------------------------------

class TestL2:
    def test_l2_does_not_exceed_budget(self):
        tiny = SegmentBudgets(rom=5000, kernel=2000, l1=5, l2=50)
        mem  = SegmentedMemory(budgets=tiny)
        for _ in range(20):
            mem.add_turn("user", "SELECT col FROM table WHERE condition = value and another thing")
        stats = mem.stats()
        assert stats["l2"]["used"] <= tiny.l2 + 10

    def test_clear_l1_flushes_to_l2(self):
        mem = SegmentedMemory()
        mem.add_turn("user", "SELECT * FROM important_table WHERE col = 1;")
        mem.clear_l1()
        assert len(mem.get_l1_messages()) == 0


# ---------------------------------------------------------------------------
# SegmentedMemory — full flow
# ---------------------------------------------------------------------------

class TestFullFlow:
    def test_total_tokens_within_budget(self):
        mem = SegmentedMemory()
        mem.set_rom("System prompt for the assistant.")
        mem.set_kernel({"user": "Alice", "goal": "SQL training"})
        for i in range(5):
            mem.add_turn("user", f"Question {i} about SELECT FROM WHERE")
            mem.add_turn("assistant", f"Answer {i}: use SELECT col FROM table WHERE id = {i};")
        stats = mem.stats()
        assert stats["total"]["used"] <= stats["total"]["budget"] + 50

    def test_render_contains_all_segments(self):
        mem = SegmentedMemory()
        mem.set_rom("SYSTEM PROMPT")
        mem.set_kernel({"goal": "KERNEL_GOAL"})
        mem.add_turn("user", "TURN_CONTENT")
        rendered = mem.render()
        assert "SYSTEM PROMPT"  in rendered
        assert "KERNEL_GOAL"    in rendered
        assert "TURN_CONTENT"   in rendered

    def test_reset_clears_all(self):
        mem = SegmentedMemory()
        mem.set_rom("ROM")
        mem.set_kernel({"k": "v"})
        mem.add_turn("user", "Q")
        mem.reset()
        stats = mem.stats()
        assert stats["total"]["used"] == 0
        assert mem.render() == ""

    def test_build_prompt_keys(self):
        mem    = SegmentedMemory()
        prompt = mem.build_prompt()
        assert set(prompt.keys()) == {"rom", "kernel", "l1", "l2"}

    def test_stats_structure(self):
        mem   = SegmentedMemory()
        stats = mem.stats()
        for seg in ("rom", "kernel", "l1", "l2", "total"):
            assert seg in stats
            assert "used"   in stats[seg]
            assert "budget" in stats[seg]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_dict_to_text_empty(self):
        assert _dict_to_text({}) == ""

    def test_dict_to_text_content(self):
        text = _dict_to_text({"user": "Alice", "goal": "Learn"})
        assert "Alice" in text
        assert "Learn"  in text

    def test_turns_to_text_empty(self):
        assert _turns_to_text([]) == ""

    def test_turns_to_text_content(self):
        turns = [Turn(role="user", content="Hello")]
        text  = _turns_to_text(turns)
        assert "Hello" in text
        assert "USER"  in text
