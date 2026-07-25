"""vcrpy cassette infrastructure for hermetic LLM testing.

Records real LLM HTTP calls to YAML cassette files, then replays them
on subsequent runs — zero API cost, deterministic CI.

Usage in tests:

    import vcrpy

    @pytest.mark.vcr
    def test_predict():
        from dspytools.core._dspy import dspy
        # First run: records to tests/cassettes/test_predict.yaml
        # Subsequent runs: replays from cassette (no LM call)
        result = dspy.Predict("question -> answer")(question="What is 2+2?")
        assert result.answer == "4"

To re-record cassettes (when prompts change):
    rm tests/cassettes/*.yaml
    pytest tests/ -k vcr  # re-records from live LM
"""

from __future__ import annotations

import os
from pathlib import Path

CASSETTE_DIR = Path(__file__).parent / "cassettes"


def get_cassette_path(test_name: str) -> Path:
    """Return the cassette file path for a test."""
    return CASSETTE_DIR / f"{test_name}.yaml"


def should_record() -> bool:
    """Whether to record new cassettes (vs replay existing).

    Set VCR_RECORD=1 to force recording. Otherwise replays existing cassettes.
    """
    return os.environ.get("VCR_RECORD", "0") == "1"
