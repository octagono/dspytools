"""Tests for the JSON registry (dspytools.core.registry).

Uses monkeypatch to redirect compiled_dir to a temp directory,
avoiding side effects on the real compiled/ directory.
"""

import tempfile
from pathlib import Path

import pytest

from dspytools.core.registry import (
    delete_run,
    get_run,
    list_compiled_runs,
    register_run,
    save_run_index,
)


@pytest.fixture
def tmp_registry(monkeypatch):
    """Redirect compiled_dir to a temp directory and return the path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Ensure the path exists (TemporaryDirectory already creates it)
        monkeypatch.setattr("dspytools.core.registry.compiled_dir", lambda: tmp_path)
        yield tmp_path
    # cleanup handled by TemporaryDirectory context manager


# ── Tests ──────────────────────────────────────────────────────────────────


def test_register_and_list(tmp_registry):
    """Register 2 runs, list them, verify both are present."""
    register_run("run-alpha", {"optimizer": "mipro", "score": 0.85})
    register_run("run-beta", {"optimizer": "gepa", "score": 0.92})

    runs = list_compiled_runs()
    assert len(runs) == 2

    ids = {r["id"] for r in runs}
    assert ids == {"run-alpha", "run-beta"}

    # Verify metadata is preserved
    by_id = {r["id"]: r for r in runs}
    assert by_id["run-alpha"]["optimizer"] == "mipro"
    assert by_id["run-beta"]["optimizer"] == "gepa"


def test_get_run(tmp_registry):
    """Register a run, get it back by ID."""
    register_run("run-42", {"optimizer": "knn", "k": 3})

    found = get_run("run-42")
    assert found is not None
    assert found["id"] == "run-42"
    assert found["optimizer"] == "knn"
    assert found["k"] == 3

    # Non-existent run returns None
    assert get_run("does-not-exist") is None


def test_delete_run(tmp_registry):
    """Register, delete, verify removal."""
    register_run("to-delete", {"optimizer": "test"})
    assert len(list_compiled_runs()) == 1

    deleted = delete_run("to-delete")
    assert deleted is True

    runs = list_compiled_runs()
    assert len(runs) == 0

    # Deleting non-existent run returns False
    assert delete_run("already-gone") is False


def test_empty_index(tmp_registry):
    """List when no runs exist returns empty list."""
    runs = list_compiled_runs()
    assert runs == []


def test_save_load_roundtrip(tmp_registry):
    """save_run_index → list_compiled_runs preserves data."""
    data = [
        {"id": "r1", "optimizer": "mipro", "score": 0.80},
        {"id": "r2", "optimizer": "gepa", "score": 0.85},
    ]
    save_run_index(data)

    loaded = list_compiled_runs()
    assert loaded == data


def test_register_without_id(tmp_registry):
    """Verify that the run_id is stored as 'id' in the index."""
    register_run("no-duplicate", {"foo": "bar"})
    run = get_run("no-duplicate")
    assert run is not None
    assert run["id"] == "no-duplicate"
    assert run["foo"] == "bar"
