"""pytest conftest — isolates test artifacts from production data.

Sets DSPYTOOLS_*_DIR env vars to pytest temp directories, and clears
the _env_path_cache so the first call picks up the env override.
Prevents test runs from trampling production compiled/index.json.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Generator
from pathlib import Path
from urllib.parse import urlsplit

import pytest


@pytest.fixture(autouse=True)
def _isolate_dspytools_dirs(tmp_path: Path) -> Generator[None, None, None]:
    """Point all dspytools config paths to a temp directory via env vars.

    Uses env vars (read by _env_path() in settings.py) instead of
    mock.patch, because many modules import compiled_dir directly at
    module level and mock.patch.object doesn't reach those references.
    Clears _env_path_cache so env override takes effect immediately.
    """
    from dspytools.config import settings as _settings

    test_home = tmp_path / "dspytools"

    # Create temp dirs matching production structure
    paths = {
        "DSPYTOOLS_COMPILED_DIR": test_home / "compiled",
        "DSPYTOOLS_DATA_DIR": test_home / "data",
        "DSPYTOOLS_MODULES_DIR": test_home / "modules",
        "DSPYTOOLS_SIGNATURES_DIR": test_home / "signatures",
        "DSPYTOOLS_AGENTS_DIR": test_home / "agents",
        "DSPYTOOLS_PROJECT_CONFIG_DIR": test_home / ".dspytools",
    }

    _saved_env = {}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        _saved_env[key] = os.environ.get(key)
        os.environ[key] = str(path)

    # Create empty index.json for compiled dir
    index_file = paths["DSPYTOOLS_COMPILED_DIR"] / "index.json"
    if not index_file.exists():
        index_file.write_text("[]")

    # Clear the env path cache so first call picks up env vars
    _saved_cache = dict(getattr(_settings, "_env_path_cache", {}))
    _settings._env_path_cache.clear()

    # Invalidate registry index cache
    from dspytools.core.registry import _invalidate_index as _inv_idx

    _inv_idx()

    yield

    # Restore env vars
    for key, val in _saved_env.items():
        if val is not None:
            os.environ[key] = val
        else:
            os.environ.pop(key, None)

    # Restore env path cache
    _settings._env_path_cache.clear()
    _settings._env_path_cache.update(_saved_cache)

    # Restore registry cache
    _inv_idx()


# ── Shared mock fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    """Mock Redis/FalkorDB for tests needing graph or cache isolation.

    Patches get_graph_client at EVERY module that uses ``from ... import get_graph_client``,
    because the ``from`` import creates a local binding that isn't affected by
    patching the source module.  Keep this list in sync with ``grep -rn
    'from dspytools.graph.client import get_graph_client' src/``.

    Returns (redis_mock, graph_mock) for assertions.
    """
    from unittest.mock import MagicMock, patch

    # All modules that do ``from dspytools.graph.client import get_graph_client``
    _GC_TARGETS = [
        "dspytools.graph.client.get_graph_client",
        "dspytools.graph.cache.get_graph_client",
        "dspytools.graph.skill_graph.get_graph_client",
        "dspytools.memory.manager.get_graph_client",
        "dspytools.mcp.tools.get_graph_client",
        "dspytools.commands.graph.get_graph_client",
    ]
    r = MagicMock()
    g = MagicMock()

    def _setup_mock(patcher):
        mock_val = patcher.start()
        mock_val.return_value.redis.return_value = r
        mock_val.return_value.graph.return_value = g
        # cache_breaker() returns a real pass-through breaker so wrapped
        # operations (breaker.call(fn, ...)) actually invoke fn in tests.
        mock_val.return_value.cache_breaker.return_value.call.side_effect = (
            lambda fn, *a, **kw: fn(*a, **kw)
        )
        return mock_val

    patchers = [patch(t) for t in _GC_TARGETS]
    for p in patchers:
        _setup_mock(p)

    yield r, g

    for p in patchers:
        p.stop()


@pytest.fixture
def mock_embedder():
    """Mock the shared embedder to return a 768-dim vector.

    Patches get_embedder at EVERY module that uses ``from ... import get_embedder``.
    """
    from unittest.mock import MagicMock, patch

    _EMB_TARGETS = [
        "dspytools.graph.cache.get_embedder",
        "dspytools.memory.manager.get_embedder",
    ]
    m = MagicMock()
    m.return_value = [0.1] * 768

    patchers = [patch(t) for t in _EMB_TARGETS]
    for p in patchers:
        mock_val = p.start()
        mock_val.return_value = m

    yield m

    for p in patchers:
        p.stop()


@pytest.fixture
def mock_falkordb_skill_graph(mock_redis):
    """Mock FalkorDBSkillGraph with a pre-configured mock backend."""
    from dspytools.graph.skill_graph import FalkorDBSkillGraph

    sg = FalkorDBSkillGraph()
    yield sg, mock_redis[1]


@pytest.fixture
def mock_drift_monitor(tmp_path: Path):
    """Create an isolated DriftMonitor for testing."""
    from dspytools.core.drift_monitor import DriftMonitor

    return DriftMonitor(state_file=str(tmp_path / ".dspytools" / "test_drift.json"))


@pytest.fixture
def mock_analysis_cache(tmp_path: Path):
    """Create an isolated AnalysisCache."""
    from dspytools.generate.cache import AnalysisCache

    d = tmp_path / "analysis_cache"
    d.mkdir(parents=True, exist_ok=True)
    return AnalysisCache(_cache_path=str(d), ttl=86400.0)


@pytest.fixture
def mock_semantic_cache(mock_redis, mock_embedder):
    """Create a SemanticCache with mocked Redis + embedder.

    Returns (cache, redis_mock, embedder_mock).
    """
    from dspytools.graph.cache import SemanticCache

    c = SemanticCache(name="test_cache", ttl_seconds=60, distance_threshold=0.15)
    embedder_mock = mock_embedder
    return c, mock_redis[0], embedder_mock


@pytest.fixture
def mock_graph(mock_redis: tuple):
    """Mock graph for test_graph_commands.py.

    Patches FalkorDBSkillGraph and get_graph_client so that graph CLI
    command callbacks operate on mocks without a real database.

    Returns dict with keys:
      - skill_graph: MagicMock for FalkorDBSkillGraph instance
      - falkordb: MagicMock for the FalkorDB graph object
      - redis: MagicMock for Redis client
    """
    from unittest.mock import MagicMock, patch

    skill_graph = MagicMock()
    _, falkordb = mock_redis

    with patch("dspytools.commands.graph.FalkorDBSkillGraph") as mock_sg_class:
        mock_sg_class.return_value = skill_graph
        with patch("dspytools.graph.skill_graph.FalkorDBSkillGraph") as mock_sg2:
            mock_sg2.return_value = skill_graph

            yield {
                "skill_graph": skill_graph,
                "falkordb": falkordb,
                "redis": mock_redis[0],
            }


# ── Hermetic LM fixtures ───────────────────────────────────────────────────
#
# DSPy 3.5's lm15 engine speaks HTTP through bundled stdlib raw-socket
# transports (``dspy/_vendor/lm15/transports/_http11.py``), which vcrpy cannot
# hook.  Recorded cassettes are dead infrastructure for this stack and were
# removed, so tests use ``dspy.utils.DummyLM`` or a stub
# ``complete(Request) -> Response`` engine (``dspy.LM(..., engine=...)``).
# ``probe_lm_endpoint()`` is the only liveness check, and nothing here probes
# or connects unless a test explicitly asks for it.


def probe_lm_endpoint(url: str | None = None, timeout: float = 0.25) -> bool:
    """Whether the configured OpenAI-compatible endpoint accepts a TCP connection.

    Fast and local: one ``socket.create_connection`` with a 0.25s timeout
    against ``DSPYTOOLS_TEST_LM_URL`` (default local llama-cpp on :8080).
    Never raises and never hangs — a closed port, bad URL or DNS failure is
    simply ``False``.
    """
    target = url or os.environ.get("DSPYTOOLS_TEST_LM_URL", "http://127.0.0.1:8080/v1")
    try:
        parts = urlsplit(target)
        if not parts.hostname:
            return False
        port = parts.port or (443 if parts.scheme == "https" else 80)
        with socket.create_connection((parts.hostname, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


@pytest.fixture
def dummy_lm():
    """Factory for a hermetic scripted LM — ``dspy.utils.DummyLM``.

    ``DummyLM(answers)`` accepts a list (consumed in order) or a dict keyed by
    text expected in the final prompt message.  Zero network, zero cost.
    """
    from dspytools.core._dspy import dspy

    return dspy.utils.DummyLM


@pytest.fixture
def student_lm(dummy_lm):
    """Hermetic stand-in for the old live llama.cpp student LM.

    Kept under the historical name so existing call sites keep working; it is a
    ``DummyLM`` now (DSPy 3.5 made cassette replay impossible).
    """
    return dummy_lm([{"answer": "4"}])


@pytest.fixture
def live_lm():
    """Opt-in live LM for manual debugging. ``None`` on every normal run.

    Two gates: ``DSPYTOOLS_LIVE_LM=1`` must be set AND the configured endpoint
    must accept a TCP connection (0.25s probe).  Nothing is constructed and
    nothing is contacted otherwise, so the suite stays hermetic.
    """
    if os.environ.get("DSPYTOOLS_LIVE_LM") != "1":
        return None
    if not probe_lm_endpoint():
        return None

    from dspytools.core._dspy import dspy

    return dspy.LM(
        f"openai/{os.environ.get('DSPYTOOLS_TEST_LM_MODEL', 'unsloth/Qwen3.5-9B-GGUF')}",
        api_base=os.environ.get("DSPYTOOLS_TEST_LM_URL", "http://127.0.0.1:8080/v1"),
        api_key=os.environ.get("DSPYTOOLS_TEST_LM_KEY", "not-needed"),
        cache=False,
        temperature=0.0,
        max_tokens=4096,
    )
