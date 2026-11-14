"""Integrity tests for the MCP subsystem.

One test group per defect fixed after the DSPy 3.5.0 audit:

1. `list_optimizers` / `dspytools://optimizers` / `compile_optimizer` share one
   optimizer-name source, and CLI-only families are reported separately;
2. every LM-touching handler guards the LM and the server configures one at
   startup (a fresh stdio server used to raise `No LM is loaded`);
3. the dispatcher separates an unknown tool from a handler `ValueError`;
4. `compile_optimizer` copies the shared spec params and splits the holdout
   before constructing the optimizer (Golden Rule 3);
5. `_last_lm_output` reads DSPy's `outputs` history key;
6. `inspect_history` uses DSPy's `file=` sink and never swaps `sys.stdout`;
7. `lora_unload_adapter` reports the truth (llama.cpp has no unload endpoint);
8. memory cache invalidation uses a glob that matches;
9. an external MCP tool with `is_error` raises instead of returning text;
10. pool operations are time-bounded and sessions are never dropped on a timer;
11. per-server connection failures are surfaced, not swallowed;
12. the process-wide HotSwap singleton is reused;
13. `agent_run` honours the configured adapter, `max_iters` and MCP schemas;
14. no tool fabricates a score or a "healthy" check it did not measure;
15. all in-band errors use `_error`'s `{"error", "detail"}` shape;
16. specs declare no `outputSchema` (the server never sent them);
17. importing the server does not import MLflow.

Everything is hermetic: `dspy.utils.DummyLM`, fake sessions/results, and
monkeypatched graph/redis/LM entry points. No network, no Redis, no FalkorDB.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from contextlib import nullcontext

import pytest

from dspytools.core._dspy import dspy
from dspytools.core.setup import LMRegistry
from dspytools.mcp import loader as mcp_loader
from dspytools.mcp import server as mcp_server
from dspytools.mcp import tools as mcp_tools


def _payload(text: str) -> dict:
    return json.loads(text)


class _LMShim:
    """Stand-in for the `dspy` module where only `settings` matters.

    `_ensure_lm_configured`, `_last_lm_output`, `inspect_history` and the
    adapter lookup all read `dspy.settings`; patching the whole module would
    break `dspy.Tool`/`dspy.ReAct`, so everything else delegates.
    """

    def __init__(self, **settings: object) -> None:
        self.settings = types.SimpleNamespace(**settings)

    def __getattr__(self, name: str) -> object:
        return getattr(dspy, name)


# ── 1. Optimizer names: one source of truth ───────────────────────────────


def test_list_and_compile_agree_on_optimizer_names(monkeypatch):
    calls: list[dict] = []

    class _FakeOpt:
        def compile(self, student, trainset):
            calls.append({"trainset": trainset})
            return student

    def _factory(metric, **kwargs):
        return _FakeOpt()

    specs = {
        "demo-opt": {"cls_lambda": _factory, "params": {"marker": 1}},
        "other-opt": {"cls_lambda": _factory, "params": {}},
    }
    monkeypatch.setattr(mcp_tools, "_OPTIMIZER_SPECS_CACHE", specs)
    monkeypatch.setattr(mcp_tools, "load_module_by_name", lambda name: object())
    monkeypatch.setattr(
        mcp_tools,
        "load_trainset",
        lambda path: [
            dspy.Example(input=str(i), output="o").with_inputs("input")
            for i in range(5)
        ],
    )
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    catalog = _payload(mcp_tools.tool_list_optimizers())
    assert catalog["optimizers"] == ["demo-opt", "other-opt"]

    for name in catalog["optimizers"]:
        result = _payload(mcp_tools.tool_compile_optimizer(name, "m", "d.json"))
        assert result["status"] == "compiled", result
        assert result["optimizer"] == name

    # A CLI-only family is reported separately and refused — never listed as
    # callable and never resolved by compile_optimizer.
    assert "finetune" in catalog["cli_only"]
    assert "finetune" not in catalog["optimizers"]
    refused = _payload(mcp_tools.tool_compile_optimizer("finetune", "m", "d.json"))
    assert "not callable" in refused["error"]
    assert "dspytools compile finetune" in refused["detail"]


def test_real_catalog_matches_optimizer_specs_and_resource(monkeypatch):
    catalog = _payload(mcp_tools.tool_list_optimizers())
    specs = mcp_tools._optimizer_specs()

    assert catalog["optimizers"] == sorted(specs)
    assert set(catalog["optimizers"]) & set(catalog["cli_only"]) == set()
    # The legacy snake_case spelling is no longer advertised as callable.
    assert "bootstrap_few_shot" not in catalog["optimizers"]
    assert "bootstrap-few-shot" in catalog["optimizers"]
    # The `dspytools://optimizers` resource serves the same payload.
    assert mcp_tools.optimizer_catalog() == catalog


# ── 2. LM bootstrap ───────────────────────────────────────────────────────


def test_create_mcp_server_configures_lm_at_startup(monkeypatch):
    calls: list[str] = []

    def _fake_ensure() -> None:
        calls.append("configured")

    monkeypatch.setattr(mcp_server, "_ensure_lm_configured", _fake_ensure)
    mcp_server.create_mcp_server()
    assert calls == ["configured"]


def test_create_mcp_server_survives_lm_failure(monkeypatch):
    def _boom() -> None:
        raise RuntimeError("no LM")

    monkeypatch.setattr(mcp_server, "_ensure_lm_configured", _boom)
    assert mcp_server.create_mcp_server() is not None


def test_tool_infer_bootstraps_lm_when_none_is_set(monkeypatch):
    seen: dict[str, object] = {}
    shim = _LMShim(lm=None)
    monkeypatch.setattr(mcp_tools, "dspy", shim)

    def _fake_setup() -> None:
        shim.settings.lm = dspy.utils.DummyLM([{"answer": "x"}])

    monkeypatch.setattr(mcp_tools, "setup_dspy", _fake_setup)

    class _Mgr:
        def infer(self, **inputs):
            seen["lm"] = shim.settings.lm
            return {"output": "ok"}

    monkeypatch.setattr(mcp_tools, "_get_hotswap", lambda: _Mgr())

    payload = _payload(mcp_tools.tool_infer(question="q"))
    assert payload["status"] == "ok"
    # The guard ran first: an LM existed by the time the handler inferred.
    assert seen["lm"] is shim.settings.lm
    assert seen["lm"] is not None


@pytest.mark.parametrize(
    "handler_name",
    [
        "tool_infer",
        "tool_challenger_solver",
        "tool_spin_optimize",
        "tool_meta_prompt_learn",
        "tool_trace2skill_evolve",
        "tool_lse_explore",
        "tool_opsd_purify",
        "tool_compile_optimizer",
        "tool_gfl_run_halving",
        "tool_agent_run",
        "tool_generate_llms_txt",
        "tool_evaluate",
        "tool_stream_infer",
    ],
)
def test_lm_touching_handlers_report_lm_failure_in_band(monkeypatch, handler_name):
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=None))
    monkeypatch.setattr(
        mcp_tools,
        "setup_dspy",
        lambda: (_ for _ in ()).throw(RuntimeError("no LM reachable")),
    )

    handler = getattr(mcp_tools, handler_name)
    # Every one of these handlers is guarded: the payload is the documented
    # error shape, not a raised exception.
    payload = _payload(handler())
    assert payload["error"] == "no LM reachable"
    assert "dspytools configure lm set student" in payload["detail"]


# ── 3. Dispatch classification ────────────────────────────────────────────


def test_unknown_tool_is_not_confused_with_handler_value_error(monkeypatch):
    def _raises_value_error(**kwargs):
        raise ValueError("dspy rejected the arguments")

    monkeypatch.setitem(
        mcp_tools.BUILTIN_TOOLS,
        "boom_tool",
        {"description": "boom", "inputSchema": {}, "handler": _raises_value_error},
    )

    unknown = _payload(mcp_server._dispatch("nope", {}))
    assert unknown["error"] == "tool_not_found"
    assert "available_tools" in unknown

    failed = _payload(mcp_server._dispatch("boom_tool", {}))
    assert failed["error"] == "invalid_request"
    assert failed["tool"] == "boom_tool"
    assert failed["message"] == "dspy rejected the arguments"


def test_dispatch_maps_runtime_and_unexpected_failures(monkeypatch):
    def _runtime(**kwargs):
        raise RuntimeError("env down")

    def _odd(**kwargs):
        raise KeyError("weird")

    monkeypatch.setitem(
        mcp_tools.BUILTIN_TOOLS,
        "runtime_tool",
        {"description": "rt", "inputSchema": {}, "handler": _runtime},
    )
    monkeypatch.setitem(
        mcp_tools.BUILTIN_TOOLS,
        "odd_tool",
        {"description": "odd", "inputSchema": {}, "handler": _odd},
    )

    assert _payload(mcp_server._dispatch("runtime_tool", {}))["error"] == (
        "execution_failed"
    )
    assert _payload(mcp_server._dispatch("odd_tool", {}))["error"] == "internal_error"


# ── 4. compile_optimizer: params copy + holdout split ────────────────────


def test_compile_optimizer_copies_params_and_splits_holdout(monkeypatch):
    spec_params: dict[str, object] = {"marker": "unchanged"}
    captured: dict[str, object] = {}
    teacher = dspy.utils.DummyLM([{"answer": "reflection"}])

    class _FakeOpt:
        def compile(self, student, trainset):
            captured["trainset"] = trainset
            return student

    def _factory(metric, **kwargs):
        captured["kwargs"] = kwargs
        return _FakeOpt()

    specs = {
        "demo-opt": {
            "cls_lambda": _factory,
            "params": spec_params,
            "needs_teacher": True,
        }
    }
    monkeypatch.setattr(mcp_tools, "_OPTIMIZER_SPECS_CACHE", specs)
    monkeypatch.setattr(mcp_tools, "load_module_by_name", lambda name: object())
    examples = [
        dspy.Example(input=str(i), output="o").with_inputs("input") for i in range(10)
    ]
    monkeypatch.setattr(mcp_tools, "load_trainset", lambda path: examples)
    monkeypatch.setattr(LMRegistry, "get_teacher", classmethod(lambda cls: teacher))
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    splits: list[tuple[int, str]] = []
    gate = mcp_tools.get_holdout_gate()
    real_split = gate.split

    def _spy(trainset, compile_id=""):
        splits.append((len(trainset), compile_id))
        return real_split(trainset, compile_id=compile_id)

    monkeypatch.setattr(gate, "split", _spy)

    payload = _payload(mcp_tools.tool_compile_optimizer("demo-opt", "m", "d.json"))

    assert payload["status"] == "compiled"
    # The process-wide spec table is untouched: no teacher (or its api_key)
    # leaks into later calls or into the CLI.
    assert spec_params == {"marker": "unchanged"}
    assert captured["kwargs"]["reflection_lm"] is teacher
    assert captured["kwargs"]["marker"] == "unchanged"
    # Holdout split happened first, and only the train split was compiled on.
    assert len(splits) == 1
    assert splits[0][0] == len(examples)
    assert len(captured["trainset"]) == payload["trainset_size"] < len(examples)
    assert payload["holdout_size"] > 0


def test_compile_optimizer_rejects_a_tiny_trainset(monkeypatch):
    monkeypatch.setattr(
        mcp_tools,
        "_OPTIMIZER_SPECS_CACHE",
        {"demo-opt": {"cls_lambda": lambda metric, **kw: None, "params": {}}},
    )
    monkeypatch.setattr(mcp_tools, "load_module_by_name", lambda name: object())
    monkeypatch.setattr(mcp_tools, "load_trainset", lambda path: [])
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    payload = _payload(mcp_tools.tool_compile_optimizer("demo-opt", "m", "d.json"))
    assert "2 examples are required" in payload["detail"]


# ── 5. _last_lm_output ────────────────────────────────────────────────────


def test_last_lm_output_reads_outputs(monkeypatch):
    lm = types.SimpleNamespace(
        history=[
            {
                "messages": [{"role": "user", "content": "hi"}],
                "prompt": "hi",
                "outputs": [{"text": "RAW MODEL OUTPUT"}],
                "timestamp": "now",
            }
        ]
    )
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=lm))
    assert mcp_tools._last_lm_output() == "RAW MODEL OUTPUT"


def test_last_lm_output_tolerates_object_outputs(monkeypatch):
    lm = types.SimpleNamespace(
        history=[{"outputs": [types.SimpleNamespace(text="OBJ TEXT")]}]
    )
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=lm))
    assert mcp_tools._last_lm_output() == "OBJ TEXT"


def test_last_lm_output_tolerates_plain_string_outputs(monkeypatch):
    """The real lm15 shape: `outputs` is a list of raw strings."""
    lm = types.SimpleNamespace(
        history=[
            {
                "messages": [{"role": "user", "content": "hi"}],
                "prompt": "hi",
                "outputs": ["<answer>\n42\n</answer>"],
                "response": types.SimpleNamespace(text="<answer>\n42\n</answer>"),
                "timestamp": "now",
            }
        ]
    )
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=lm))
    assert mcp_tools._last_lm_output() == "<answer>\n42\n</answer>"


def test_last_lm_output_empty_without_history(monkeypatch):
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=None))
    assert mcp_tools._last_lm_output() == ""


# ── 6. inspect_history sink ──────────────────────────────────────────────


def test_inspect_history_uses_a_file_sink(monkeypatch):
    calls: list[dict] = []

    def _fake_inspect_history(n=1, file=None):
        calls.append({"n": n, "file": file})
        assert file is not None, "must not fall back to sys.stdout"
        file.write("HISTORY")

    shim = _LMShim()
    shim.inspect_history = _fake_inspect_history
    monkeypatch.setattr(mcp_tools, "dspy", shim)

    stdout_before = sys.stdout
    assert mcp_tools.tool_inspect_history(n=2) == "HISTORY"
    assert calls and calls[0]["n"] == 2
    # sys.stdout is never swapped, so an exception cannot leave it broken.
    assert sys.stdout is stdout_before


# ── 7. LoRA unload honesty ───────────────────────────────────────────────


def test_lora_unload_reports_unsupported_not_unloaded():
    payload = _payload(mcp_tools.tool_lora_unload_adapter("demo"))
    assert payload["status"] == "unsupported"
    assert "no LoRA unload endpoint" in payload["detail"]
    assert payload["name"]


# ── 8. Cache invalidation glob ───────────────────────────────────────────


class _RecordingCache:
    """Minimal cache that honours `keys(pattern)` then `delete(key)`."""

    def __init__(self) -> None:
        self.data = {"memory_all:u": "1", "memory_stats:u": "2", "list_programs": "3"}
        self.deleted: list[str] = []

    def keys(self, pattern: str = "*"):
        return list(self.data)

    def delete(self, key: str) -> None:
        self.deleted.append(key)
        self.data.pop(key, None)


def test_invalidate_memory_glob_deletes_entries(monkeypatch):
    cache = _RecordingCache()
    monkeypatch.setattr(mcp_tools, "_resolved_cache", cache)

    mcp_tools._invalidate("memory_*")

    assert sorted(cache.deleted) == ["memory_all:u", "memory_stats:u"]
    assert "list_programs" in cache.data


def test_invalidate_uses_the_fallback_cache_too(monkeypatch):
    cache = mcp_tools._FallbackCache()
    cache.set("memory_all:u", "1")
    cache.set("memory_stats:u", "2")
    cache.set("list_programs", "3")
    monkeypatch.setattr(mcp_tools, "_resolved_cache", cache)

    mcp_tools._invalidate("memory_*")

    assert cache.get("memory_all:u") is None
    assert cache.get("memory_stats:u") is None
    assert cache.get("list_programs") == "3"


def test_memory_delete_invalidates_with_a_glob(monkeypatch):
    cache = _RecordingCache()
    monkeypatch.setattr(mcp_tools, "_resolved_cache", cache)
    monkeypatch.setitem(mcp_tools.BUILTIN_TOOLS, "unused", {"handler": lambda: ""})

    class _Mgr:
        def delete(self, memory_id):
            return True

    monkeypatch.setattr(mcp_tools, "get_memory_manager", lambda: _Mgr())

    mcp_tools.tool_memory_delete("mem-1")
    assert sorted(cache.deleted) == ["memory_all:u", "memory_stats:u"]


# ── 9. is_error raises ───────────────────────────────────────────────────


class _ResultSession:
    def __init__(self, *, is_error: bool) -> None:
        self.is_error = is_error
        self.calls: list[tuple] = []

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return types.SimpleNamespace(
            is_error=self.is_error,
            content=[types.SimpleNamespace(text="tool output")],
        )


def _sync_tool(session) -> object:
    return mcp_loader._make_sync_tool(
        name="demo_tool",
        description="demo",
        input_schema={"type": "object", "properties": {}},
        session=session,
    )


def test_sync_tool_returns_text_on_success():
    session = _ResultSession(is_error=False)
    tool = _sync_tool(session)
    assert tool() == "tool output"
    assert session.calls == [("demo_tool", None)]


def test_sync_tool_raises_on_is_error():
    tool = _sync_tool(_ResultSession(is_error=True))
    with pytest.raises(mcp_loader.MCPToolError) as excinfo:
        tool()
    assert excinfo.value.tool_name == "demo_tool"
    assert "tool output" in str(excinfo.value)


# ── 10. Pool timeouts and session lifetime ───────────────────────────────


def test_run_coro_times_out_instead_of_hanging():
    import asyncio

    loop = mcp_loader._MCPEventLoop()
    loop.start()
    gate = asyncio.Event()  # created on the test thread; never set

    async def _hang():
        await gate.wait()

    try:
        with pytest.raises(TimeoutError):
            loop.run_coro(_hang(), timeout=0.05)
    finally:
        # Let the cancelled task settle before the loop thread stops, so the
        # test does not leak "Task was destroyed but it is pending".
        try:
            loop.run_coro(asyncio.sleep(0), timeout=5)
        finally:
            loop.stop()


def test_get_tools_keeps_working_sessions_past_the_old_timer(monkeypatch):
    pool = mcp_loader.MCPSessionPool()
    connected: list[str] = []

    def _connect(config_path):
        connected.append(config_path)
        return ["tool"]

    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_connect_all", _connect)
    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_disconnect_all", lambda: None)
    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_tools", [])

    assert pool.get_tools("cfg") == ["tool"]
    # Pretend a long time passed: no keepalive sweep reconnects a live pool.
    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_last_connect", 0.0)
    assert pool.get_tools("cfg") == ["tool"]
    assert connected == ["cfg"]
    # refresh() is the documented way to force a reconnect.
    pool.refresh()
    assert pool.get_tools("cfg") == ["tool"]
    assert connected == ["cfg", "cfg"]


# ── 11. Partial-connection surfacing ─────────────────────────────────────


class _FailingCM:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aenter__(self):
        raise self.error

    async def __aexit__(self, *_exc):
        return None


def test_failed_servers_are_recorded_and_described(tmp_path, monkeypatch):
    cfg = tmp_path / ".mcp.json"
    cfg.write_text(
        json.dumps({"mcpServers": {"broken": {"command": "nope-mcp"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mcp_loader,
        "stdio_client",
        lambda params: _FailingCM(FileNotFoundError("missing binary")),
    )

    pool = mcp_loader.MCPSessionPool()
    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_disconnect_all", lambda: None)
    monkeypatch.setattr(mcp_loader.MCPSessionPool, "_tools", [])

    tools = pool.get_tools(str(cfg))
    assert tools == []
    assert pool.failed_servers() == [
        {"server": "broken", "error": "FileNotFoundError: missing binary"}
    ]
    assert "broken" in pool.describe_failures()


def test_mcp_tools_command_reports_failed_servers(monkeypatch):
    from click.testing import CliRunner

    from dspytools.commands.mcp import mcp_tools as mcp_tools_cmd

    monkeypatch.setattr(
        mcp_loader, "load_mcp_tools_sync", lambda cfg_path=".mcp.json": ([], [])
    )
    monkeypatch.setattr(
        mcp_loader.MCPSessionPool,
        "failed_servers",
        classmethod(lambda cls: [{"server": "broken", "error": "boom"}]),
    )

    result = CliRunner().invoke(mcp_tools_cmd, ["--config", "x.json"])

    combined = result.output + getattr(result, "stderr", "")
    assert result.exit_code == 0
    assert "broken" in combined
    assert "failed to connect" in combined


# ── 12. HotSwap singleton ────────────────────────────────────────────────


def test_stream_infer_reuses_the_module_hotswap_singleton(monkeypatch):
    created: list[object] = []

    class _Mgr:
        active_id = "prog"

        def __init__(self) -> None:
            created.append(self)

        def swap(self, program_id):
            self.active_id = program_id

        def infer(self, **inputs):
            return {"output": "hello world"}

    monkeypatch.setattr(mcp_tools, "HotSwapManager", _Mgr)
    monkeypatch.setattr(mcp_tools, "_hotswap", None)
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    first = _payload(mcp_tools.tool_stream_infer("hi"))
    second = _payload(mcp_tools.tool_stream_infer("hi", program_id="other"))

    assert first["program_id"] == "prog"
    assert second["program_id"] == "other"
    # One manager for the process: swaps and the LRU cache are process state.
    assert len(created) == 1


# ── 13. agent_run: adapter, max_iters, MCP schemas ───────────────────────


def test_agent_run_honours_adapter_limits_and_mcp_schemas(monkeypatch):
    sentinel_adapter = object()
    captured: dict[str, object] = {}

    class _FakeReAct:
        def __init__(self, signature, tools=None, max_iters=None):
            captured["tools"] = tools
            captured["max_iters"] = max_iters

        def __call__(self, **kwargs):
            return types.SimpleNamespace(
                answer="42", termination_reason="submit", history=None
            )

    def _fake_context(**kwargs):
        captured["context"] = kwargs
        return nullcontext()

    shim = _LMShim(adapter=sentinel_adapter)
    shim.ReAct = _FakeReAct
    shim.context = _fake_context
    monkeypatch.setattr(mcp_tools, "dspy", shim)
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)
    monkeypatch.setitem(
        mcp_tools.BUILTIN_TOOLS,
        "demo_tool",
        {
            "description": "demo tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            "handler": lambda query, limit=5: "ok",
        },
    )

    payload = _payload(
        mcp_tools.tool_agent_run("q", tools='["demo_tool"]', max_iters=3)
    )

    assert payload["answer"] == "42"
    assert captured["max_iters"] == 3
    # The configured adapter is scoped, never a hardcoded BAMLAdapter.
    assert captured["context"]["adapter"] is sentinel_adapter
    tool = captured["tools"][0]
    assert tool.name == "demo_tool"
    # The MCP inputSchema survives instead of being re-derived from a signature.
    assert tool.arg_desc["query"].startswith("Search query")
    assert tool.args["limit"]["default"] == 5


def test_agent_run_schema_advertises_tools_and_max_iters():
    schema = mcp_tools.BUILTIN_TOOLS["agent_run"]["inputSchema"]
    assert set(schema["properties"]) == {"question", "tools", "max_iters"}
    assert schema["required"] == ["question"]


# ── 14. No fabricated measurements ───────────────────────────────────────


def test_doctor_reports_a_measured_dspy_check(monkeypatch):
    monkeypatch.setattr(mcp_tools, "dspy", _LMShim(lm=None))
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    def _offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(mcp_tools.urllib.request, "urlopen", _offline)
    monkeypatch.setattr(
        mcp_tools._sp,
        "run",
        lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout=""),
    )

    payload = _payload(mcp_tools.tool_doctor())
    assert payload["checks"]["dspy"] is False
    assert payload["details"]["dspy_version"]

    shim = _LMShim(lm=dspy.utils.DummyLM([{"answer": "x"}]))
    monkeypatch.setattr(mcp_tools, "dspy", shim)
    assert _payload(mcp_tools.tool_doctor())["checks"]["dspy"] is True


def test_lse_explore_skips_instead_of_fabricating_scores(monkeypatch):
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)
    monkeypatch.setattr(
        mcp_tools,
        "load_module_by_name",
        lambda name: pytest.fail("must not load a module without a trainset"),
    )

    payload = _payload(mcp_tools.tool_lse_explore("mod"))
    assert payload["status"] == "skipped"
    assert "nodes" not in payload
    assert not hasattr(mcp_tools, "_dummy_trainset")


def test_lse_explore_scores_with_the_evaluator(monkeypatch):
    measured: list[float] = []

    class _Compiled:
        pass

    class _Opt:
        def compile(self, student, trainset):
            return _Compiled()

    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)
    monkeypatch.setattr(mcp_tools, "load_module_by_name", lambda name: object())
    monkeypatch.setattr(mcp_tools, "_make_optimizer_instance", lambda name: _Opt())

    def _fake_measure(program, devset):
        measured.append(len(devset))
        return 0.25

    monkeypatch.setattr(mcp_tools, "_measure", _fake_measure)

    trainset_json = json.dumps(
        [{"input": str(i), "output": "o"} for i in range(6)],
    )
    payload = _payload(mcp_tools.tool_lse_explore("mod", trainset_json, max_depth=1))

    node = payload["nodes"][0]
    assert node["score"] == 0.25
    assert node["scored_on"].startswith("holdout[")
    assert measured  # every score came from a real evaluation


def test_meta_prompt_learn_does_not_invent_a_score(monkeypatch):
    import dspytools.gfl.paper_optimizers as paper

    class _FakeMeta:
        def meta_learn(self, task_programs, dev_sets, num_iterations=5):
            return {"final_meta_prompt": "be helpful", "iterations": []}

    monkeypatch.setattr(paper, "MetaPromptOptimizer", _FakeMeta)
    monkeypatch.setattr(mcp_tools, "_ensure_lm_configured", lambda: None)

    payload = _payload(mcp_tools.tool_meta_prompt_learn())
    assert payload["final_score"] is None
    assert payload["status"] == "not_evaluated"
    assert "no task programs" in payload["reason"]


# ── 15. Uniform error shape ──────────────────────────────────────────────


def test_tool_errors_share_the_error_detail_shape(monkeypatch):
    monkeypatch.setattr(
        mcp_tools,
        "_get_hotswap",
        lambda: types.SimpleNamespace(get_metadata=lambda pid: None),
    )
    metadata = _payload(mcp_tools.tool_get_program_metadata("missing"))
    lora = _payload(
        mcp_tools.tool_lora_load_adapter("nope", path="/definitely/not/here")
    )

    for payload in (metadata, lora):
        assert set(payload) <= {"error", "detail"}
        assert payload["error"]
        assert "status" not in payload
        assert "message" not in payload


# ── 16/17. Spec hygiene and lazy imports ─────────────────────────────────


def test_no_spec_declares_an_output_schema():
    offenders = [n for n, s in mcp_tools.BUILTIN_TOOLS.items() if "outputSchema" in s]
    assert offenders == []


def test_importing_the_server_does_not_import_mlflow():
    code = "import sys; import dspytools.mcp.server; print('mlflow' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
