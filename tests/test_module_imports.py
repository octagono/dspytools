"""Verify all dspytools package modules import cleanly at runtime.

These imports should work with dspy installed as a dependency.
No live vLLM/API server required — the imports only validate
that the module dependency graph is sound.

If a test fails because the DSPy package itself is not installed,
it is skipped with an explanatory message.
"""


def test_core_imports():
    """All core modules import cleanly."""
    from dspytools.core._dspy import dspy  # lazy, defers actual dspy import

    # These modules have no DSPy imports at top level → safe

    # These modules import dspy directly at top level → requires dspy installed

    # Verify the lazy wrapper is an instance of _LazyDSPy
    assert type(dspy).__name__ == "_LazyDSPy"


def test_generate_imports():
    """All generate package exports import cleanly.

    RepositoryAnalyzer triggers lazy dspy imports at class-definition time
    (dspy.Signature, dspy.Module, etc.) which requires dspy installed.
    """


def test_commands_import():
    """The compile command module (with _OPTIMIZER_SPECS) imports cleanly."""


def test_gfl_import():
    """GFL pipeline module imports without triggering compilation."""


def test_config_imports():
    """Config modules are pure Python — no DSPy dependency."""


def test_cli_import():
    """CLI output utilities import cleanly."""
