"""Tests for the optimizer registry factory in dspytools.commands.compile.

Tests the _OPTIMIZER_SPECS registry and exact_match_metric helper.
Does NOT register Click commands or invoke any optimizers.
"""

from dspytools.commands.compile import (
    _OPTIMIZER_SPECS,
    compile_cmd,
)
from dspytools.core.metrics import exact_match_metric


def test_all_specs_registered():
    """Every spec in the registry should have a corresponding Click subcommand."""
    registered_names = set(compile_cmd.commands.keys())
    for name in _OPTIMIZER_SPECS:
        assert name in registered_names, (
            f"Optimizer '{name}' not registered as a compile subcommand"
        )


def test_spec_names():
    """Verify all expected optimizers are in the registry."""
    expected = {
        "knn",
        "mipro",
        "gepa",
        "copro",
        "simba",
        "bootstrap-few-shot",
        "bootstrap-few-shot-random",
        "bootstrap-few-shot-optuna",
        "labeled-few-shot",
        "infer-rules",
    }
    assert set(_OPTIMIZER_SPECS.keys()) == expected


def test_each_spec_has_required_keys():
    """Every spec must have cls_name and cls_lambda."""
    for name, spec in _OPTIMIZER_SPECS.items():
        assert "cls_name" in spec, f"'{name}' missing cls_name"
        assert "cls_lambda" in spec, f"'{name}' missing cls_lambda"
        assert callable(spec["cls_lambda"]), f"'{name}' cls_lambda is not callable"


def test_specs_have_valid_names():
    """cls_name should look like a DSPy class name."""
    for name, spec in _OPTIMIZER_SPECS.items():
        cls_name = spec["cls_name"]
        assert cls_name[0].isupper(), f"'{name}' cls_name '{cls_name}' not PascalCase"
        assert " " not in cls_name, f"'{name}' cls_name contains spaces"


def test_metric_deterministic():
    """Verify the exact match metric works correctly and deterministically."""

    class FakePred:
        pass

    class FakeExample:
        pass

    p = FakePred()
    e = FakeExample()

    # Exact match
    p.output = "hello"
    e.output = "hello"
    assert exact_match_metric()(e, p) == 1.0

    # No match
    p.output = "world"
    assert exact_match_metric()(e, p) == 0.0


def test_metric_missing_attributes():
    """Metric returns 1.0 when both outputs default to '' (equal defaults)."""

    class Empty:
        pass

    e = Empty()
    p = Empty()
    # Both getattr(..., "output", "") return "" → equal → 1.0
    assert exact_match_metric()(e, p) == 1.0


def test_metric_missing_one_side():
    """Metric returns 0.0 when one side is missing output and the other is not."""

    class WithOutput:
        pass

    class WithoutOutput:
        pass

    e = WithOutput()
    e.output = "hello"
    p = WithoutOutput()
    # e has "hello", p defaults to "" → not equal → 0.0
    assert exact_match_metric()(e, p) == 0.0
    # Also test the reverse
    assert exact_match_metric()(p, e) == 0.0
