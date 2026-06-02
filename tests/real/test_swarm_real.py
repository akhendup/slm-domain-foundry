"""SwarmManager with a real tiny model on disk."""
import pytest

from app.swarm import SwarmManager

pytestmark = [pytest.mark.real, pytest.mark.unit]


def test_load_and_generate_one(tiny_lm_dir):
    swarm = SwarmManager()
    status = swarm.load("tiny", tiny_lm_dir)
    assert "Loaded" in status
    out = swarm.generate_one("tiny", [{"role": "user", "content": "Say hi"}])
    assert isinstance(out, str)
    swarm.unload("tiny")


def test_generate_all(tiny_lm_dir):
    swarm = SwarmManager()
    swarm.load("a", tiny_lm_dir)
    results = swarm.generate_all([{"role": "user", "content": "Test"}])
    assert "a" in results
    assert isinstance(results["a"], str)
