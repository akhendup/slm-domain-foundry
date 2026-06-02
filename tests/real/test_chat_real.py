"""CLI chat with a real merged tiny model."""
import sys

import pytest

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_run_demo_noninteractive(tiny_lm_dir, capsys):
    from app.chat import run_demo

    run_demo(tiny_lm_dir, interactive=False)
    assert "Q:" in capsys.readouterr().out


def test_main_with_model_dir(tiny_lm_dir):
    from app.chat import main

    argv = ["chat", "--model-dir", str(tiny_lm_dir)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old
