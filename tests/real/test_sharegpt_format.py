"""Real tokenizer tests for train.sharegpt_format."""
import pytest

from train.sharegpt_format import format_sharegpt_messages, make_sharegpt_formatting_func

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_format_messages_no_template(tiny_lm_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_lm_dir))
    text = format_sharegpt_messages(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        tok,
    )
    assert "user: Hi" in text
    assert "assistant: Hello" in text


def test_batched_formatting_func(tiny_lm_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_lm_dir))
    fn = make_sharegpt_formatting_func(tok)
    out = fn(
        {
            "conversations": [
                [
                    {"role": "user", "content": "Q"},
                    {"role": "assistant", "content": "A"},
                ]
            ]
        }
    )
    assert len(out) == 1
    assert len(out[0]) > 0
