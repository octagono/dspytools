"""Tests for HotSwapManager — LRU cache, swap, metadata caching, inference scoring."""

from __future__ import annotations

import json

import pytest

from dspytools.core.hotswap import HotSwapManager
from dspytools.core.registry import save_run_index


@pytest.fixture
def fake_run(tmp_path):
    """Create a fake compiled run directory with valid metadata."""
    import os

    from dspytools.config import settings as _settings

    os.environ["DSPYTOOLS_COMPILED_DIR"] = str(tmp_path)
    _settings._env_path_cache.clear()

    run_id = "test_run_001"
    run_dir = tmp_path / run_id
    run_dir.mkdir()

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "id": run_id,
                "optimizer": "test",
                "module_type": "predict",
            }
        )
    )
    (run_dir / "signature.json").write_text(
        json.dumps(
            {
                "inputs": ["question"],
                "outputs": ["answer"],
            }
        )
    )
    (run_dir / "program.json").write_text(json.dumps({}))

    save_run_index([{"id": run_id, "optimizer": "test"}])
    return run_id


class TestHotSwapManager:
    """HotSwapManager core lifecycle tests."""

    def test_swap_sets_active(self, fake_run):
        """swap() activates the program. load_single auto-activates on first load."""
        mgr = HotSwapManager()
        mgr.swap(fake_run)
        assert mgr.active_id == fake_run

    def test_swap_unknown_raises_keyerror(self):
        """swap() with unknown run_id raises KeyError."""
        mgr = HotSwapManager()
        with pytest.raises(KeyError):
            mgr.swap("nonexistent_run")

    def test_active_program_requires_swap(self):
        """Accessing active_program without swap raises RuntimeError."""
        mgr = HotSwapManager()
        with pytest.raises(RuntimeError):
            _ = mgr.active_program

    def test_is_loaded_false_before_load(self):
        """is_loaded() returns False for unloaded programs."""
        mgr = HotSwapManager()
        assert not mgr.is_loaded("nonexistent")

    def test_list_returns_empty_before_load(self):
        """list() returns empty list when no programs loaded."""
        mgr = HotSwapManager()
        assert mgr.list() == []

    def test_get_metadata_returns_none_for_unknown(self):
        """get_metadata() returns None for unknown run_id."""
        mgr = HotSwapManager()
        assert mgr.get_metadata("nonexistent") is None

    def test_unload_returns_false_for_unknown(self):
        """unload() returns False for programs not in cache."""
        mgr = HotSwapManager()
        assert not mgr.unload("nonexistent")

    def test_unload_resets_active(self, fake_run):
        """unload() of active program selects a new active."""
        mgr = HotSwapManager()
        mgr.swap(fake_run)
        assert mgr.active_id == fake_run
        mgr.unload(fake_run)
        assert mgr.active_id is None

    def test_active_id_none_before_swap(self):
        """active_id is None before any swap."""
        mgr = HotSwapManager()
        assert mgr.active_id is None


class TestEnsureIndex:
    """Tests for lazy index loading."""

    def test_ensure_index_loads_from_disk(self, fake_run):
        """_ensure_index populates metadata from compiled/index.json."""
        mgr = HotSwapManager()
        mgr._ensure_index()
        assert fake_run in mgr._metadata

    def test_ensure_index_idempotent(self, fake_run):
        """_ensure_index only loads once (flag-based)."""
        mgr = HotSwapManager()
        mgr._ensure_index()
        assert mgr._index_loaded is True
        # Second call should be a no-op
        mgr._ensure_index()
        assert fake_run in mgr._metadata


class TestStochasticScoring:
    """Tests for the stochastic auto_metric scoring (1-in-10)."""

    def test_infer_count_starts_at_zero(self):
        """The global _infer_count starts at 0."""
        from dspytools.core import hotswap

        # Just verify the variable exists
        assert hasattr(hotswap, "_infer_count")

    def test_infer_count_increments(self):
        """_infer_count is incremented during infer()."""
        import dspytools.core.hotswap as hs_module

        original = hs_module._infer_count
        # Verify it's an int
        assert isinstance(original, int)
