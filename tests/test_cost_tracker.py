"""Test cost tracking and lineage (Features #1, #3)."""

from dspytools.core.cost_tracker import PRICING, CompileCost, TokenUsage


def test_token_usage_estimate():
    usage = TokenUsage.estimate("deepseek/deepseek-v4-flash", 500000, 500000)
    assert usage.model == "deepseek/deepseek-v4-flash"
    assert usage.total_tokens == 1000000
    assert usage.cost_estimate > 0
    assert usage.cost_estimate < 1.0  # < $1 for 1M tokens


def test_token_usage_default_pricing():
    usage = TokenUsage.estimate("unknown-model", 1000000, 0)
    assert usage.cost_estimate >= 0


def test_compile_cost_aggregation():
    c = CompileCost(compile_id="test1", optimizer="mipro")
    c.add_call("deepseek/deepseek-v4-flash", 500000, 500000)
    c.add_call("Qwen/Qwen2.5-Coder-3B-Instruct-AWQ", 1000000, 0)
    c.finish()

    assert c.total_tokens == 2000000
    assert c.teacher_tokens == 1000000
    assert c.student_tokens == 1000000
    assert c.total_cost > 0
    assert c.elapsed_seconds >= 0


def test_compile_cost_summary():
    c = CompileCost(compile_id="test2", optimizer="gepa")
    c.add_call("deepseek/deepseek-v4-flash", 100000, 100000)
    c.finish()

    summary = c.summary()
    assert summary["compile_id"] == "test2"
    assert summary["optimizer"] == "gepa"
    assert summary["total_tokens"] == 200000
    assert "total_cost_usd" in summary
    assert summary["calls"] == 1


def test_local_model_cost_zero():
    usage = TokenUsage.estimate("Qwen/Qwen2.5-Coder-3B-Instruct-AWQ", 1000000, 1000000)
    assert usage.cost_estimate == 0.0


def test_pricing_contains_expected_keys():
    assert "deepseek/deepseek-v4-flash" in PRICING
    assert "Qwen/Qwen2.5-Coder-3B-Instruct-AWQ" in PRICING
    assert "default" in PRICING
