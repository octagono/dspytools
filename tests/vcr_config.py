"""vcrpy cassette infrastructure for hermetic LLM testing.

Records real LLM HTTP calls to YAML cassette files, then replays them
on subsequent runs — zero API cost, deterministic CI.

The working fixtures live in ``tests/conftest.py``:

- ``vcr_instance`` (session) — a configured ``vcr.VCR`` that records/replays
  dspy LM HTTP calls. Set ``VCR_RECORD=1`` to force re-recording.
- ``student_lm`` — a ``dspy.LM`` pointed at the local model
  (``DSPYTOOLS_TEST_LM_URL`` / ``DSPYTOOLS_TEST_LM_MODEL``, defaults to the
  llama-cpp server on :8080). Returns None when no LM is reachable.

Usage in tests (see tests/test_optimizer_behavior.py):

    def test_predict(student_lm, vcr_instance):
        from dspytools.core._dspy import dspy
        with vcr_instance.use_cassette("test_predict.yaml"):
            with dspy.context(lm=student_lm):
                result = dspy.Predict("question -> answer")(question="What is 2+2?")
        assert result.answer.strip()

First run records to tests/cassettes/test_predict.yaml; subsequent runs replay
it with no LM call.

To re-record cassettes (when prompts change):
    rm tests/cassettes/*.yaml
    VCR_RECORD=1 pytest tests/test_optimizer_behavior.py
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
