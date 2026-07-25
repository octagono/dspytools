"""Tests for ResourceBudget — hard limits on optimization resources."""

from __future__ import annotations

import time

import pytest

from dspytools.gfl.budget import ResourceBudget


def test_budget_spend_tokens():
    b = ResourceBudget()
    b.spend_tokens(50000)
    assert b.tokens_spent == 50000
    b.spend_tokens(25000)
    assert b.tokens_spent == 75000


def test_budget_summary():
    b = ResourceBudget(max_iterations=5)
    b.spend_tokens(100000)
    summary = b.summary
    assert summary["tokens_spent"] == 100000
    assert "llm_calls_used" in summary
    assert "elapsed_seconds" in summary


def test_budget_default():
    b = ResourceBudget()
    assert b.tokens_spent == 0
    assert b.remaining_llm == 100


def test_default_limits():
    b = ResourceBudget()
    assert b.max_llm_calls == 100
    assert b.max_wall_seconds == 300
    assert b.max_agents_generated == 10
    assert b.max_iterations == 20


def test_check_no_raise():
    b = ResourceBudget()
    b.check()


# ── Stress tests ──────────────────────────────────────────────────────


def test_exceed_llm_calls_raises():
    """Calling check() past max_llm_calls raises RuntimeError."""
    b = ResourceBudget(max_llm_calls=3)
    b.check()
    b.check()
    b.check()
    with pytest.raises(RuntimeError, match="LLM call budget exceeded"):
        b.check()


def test_wall_time_exceeded_raises():
    """max_wall_seconds=0 causes immediate RuntimeError."""
    b = ResourceBudget(max_wall_seconds=0)
    time.sleep(0.01)
    with pytest.raises(RuntimeError, match="Wall time budget exceeded"):
        b.check()


def test_production_preset():
    """PRODUCTION preset has higher limits."""
    preset = ResourceBudget.PRODUCTION
    assert preset["max_llm_calls"] == 200
    assert preset["max_wall_seconds"] == 600
    assert preset["max_agents_generated"] == 20


def test_light_preset():
    """LIGHT preset has lower limits."""
    preset = ResourceBudget.LIGHT
    assert preset["max_llm_calls"] == 50
    assert preset["max_wall_seconds"] == 180
    assert preset["max_agents_generated"] == 5


def test_budget_from_production_preset():
    """ResourceBudget constructed from PRODUCTION preset."""
    b = ResourceBudget(**ResourceBudget.PRODUCTION)
    assert b.max_llm_calls == 200
    assert b.max_wall_seconds == 600


def test_remaining_llm_decreases():
    """remaining_llm reflects calls used."""
    b = ResourceBudget(max_llm_calls=10)
    assert b.remaining_llm == 10
    b.check()
    assert b.remaining_llm == 9


def test_elapsed_seconds_monotonic():
    """elapsed_seconds increases over time."""
    b = ResourceBudget()
    e1 = b.elapsed_seconds
    time.sleep(0.01)
    e2 = b.elapsed_seconds
    assert e2 >= e1


def test_summary_after_spend():
    """Summary reflects token spend."""
    b = ResourceBudget()
    b.spend_tokens(5000)
    b.spend_tokens(3000)
    s = b.summary
    assert s["tokens_spent"] == 8000


def test_zero_tokens():
    """Zero token spend is valid."""
    b = ResourceBudget()
    b.spend_tokens(0)
    assert b.tokens_spent == 0
