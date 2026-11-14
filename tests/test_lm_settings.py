"""LM settings + cache policy tests (DSPy 3.5).

Everything is offline: ``load_config``/``load_env`` are patched, ``dspy.configure``
and ``dspy.configure_cache`` are captured, and LM calls go through a stub engine
(``complete(Request) -> Response``).  No socket is opened and no config file is
read or written outside ``tmp_path``.

Coverage:

1. ``setup_dspy`` applies the persisted ``cfg["dspy"]`` runtime settings.
2. Unset keys are never injected (DSPy's own defaults stand).
3. Persisted cache policy is applied with ``restrict_pickle=True`` always.
4. ``cache disable`` turns caching off and never loosens pickle restrictions.
5. Explicit-model LMs are cached by default (``cache=None`` passthrough).
6. The setup fast path is invalidated by a temperature change.
7. ``cache clear`` removes both dspytools' and DSPy's cache directories.
8. The ``response_format`` retry reaches the LM without ``response_format``.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from dspytools.core._dspy import dspy


# ── Hermetic scaffolding ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No config file, no .env, no cached LM instances, no "already configured"."""
    import dspytools.core.setup as setup_mod

    state: dict = {"cfg": {}}
    monkeypatch.setattr(setup_mod, "load_config", lambda: state["cfg"])
    monkeypatch.setattr(setup_mod, "load_env", lambda: {})
    monkeypatch.setattr(setup_mod, "merge_environ", lambda env=None: None)
    monkeypatch.setattr(setup_mod, "_setup_configured", False)
    setup_mod.LMRegistry.clear()
    yield state
    setup_mod.LMRegistry.clear()


class _RecordingEngine:
    """Stub sync engine: records the canonical request, returns fixed text."""

    def __init__(self, text: str = "stub answer") -> None:
        self.text = text
        self.requests: list = []

    def complete(self, request):
        self.requests.append(request)
        return dspy.lm15.Response(
            id=None,
            model=request.model,
            message=dspy.lm15.Message.assistant([dspy.lm15.TextPart(self.text)]),
            finish_reason="stop",
            usage=dspy.lm15.Usage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


def _stub_lm(text: str = "stub answer") -> dspy.LM:
    return dspy.LM(
        "openai/gpt-4o", engine=_RecordingEngine(text), cache=False, model_type="chat"
    )


def _run_setup_dspy(monkeypatch, state, *, cfg=None, model=None):
    """Call ``setup_dspy()`` hermetically; return the ``dspy.configure`` kwargs."""
    import dspytools.core.setup as setup_mod

    state["cfg"] = {} if cfg is None else cfg
    if model is None:
        monkeypatch.setattr(
            setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: _stub_lm())
        )
    # A previous call in the same test would otherwise take the fast path.
    monkeypatch.setattr(setup_mod, "_setup_configured", False)
    captured: dict = {}
    monkeypatch.setattr(dspy, "configure", lambda **kwargs: captured.update(kwargs))
    setup_mod.setup_dspy(model=model)
    return captured


# ── 1-2. persisted runtime settings ───────────────────────────────────────


def test_persisted_runtime_settings_are_forwarded(monkeypatch, _hermetic):
    captured = _run_setup_dspy(
        monkeypatch,
        _hermetic,
        cfg={"dspy": {"max_errors": 3, "num_threads": 2, "track_usage": True}},
    )

    assert captured["max_errors"] == 3
    assert captured["num_threads"] == 2
    assert captured["track_usage"] is True


def test_persisted_settings_land_in_dspy_settings(monkeypatch, _hermetic):
    import dspytools.core.setup as setup_mod

    _hermetic["cfg"] = {"dspy": {"max_errors": 3}}
    monkeypatch.setattr(
        setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: _stub_lm())
    )
    previous = {
        key: getattr(dspy.settings, key, None)
        for key in ("lm", "adapter", "max_errors")
    }
    try:
        setup_mod.setup_dspy()
        assert dspy.settings.max_errors == 3
    finally:
        dspy.configure(**previous)


def test_unset_settings_are_not_injected(monkeypatch, _hermetic):
    captured = _run_setup_dspy(
        monkeypatch, _hermetic, cfg={"dspy": {"adapter": "chat"}}
    )

    assert set(captured) == {"lm", "adapter"}
    # The adapter/model_type/cache keys are consumed here, never forwarded.
    assert "model_type" not in captured
    assert "cache" not in captured


def test_warn_type_is_not_forced(monkeypatch, _hermetic):
    """No config, no explicit argument -> no warn_on_type_mismatch injection."""
    captured = _run_setup_dspy(monkeypatch, _hermetic)
    assert "warn_on_type_mismatch" not in captured

    captured = _run_setup_dspy(
        monkeypatch, _hermetic, cfg={"dspy": {"warn_on_type_mismatch": True}}
    )
    assert captured["warn_on_type_mismatch"] is True


# ── 3-4. cache policy ─────────────────────────────────────────────────────


def test_cache_config_is_applied_with_restricted_pickle(monkeypatch, _hermetic):
    import dspytools.core.setup as setup_mod

    calls: dict = {}
    monkeypatch.setattr(dspy, "configure_cache", lambda **kw: calls.update(kw))

    setup_mod._apply_cache_config(
        {
            "dspy": {
                "cache": {
                    "enable_disk_cache": False,
                    "enable_memory_cache": False,
                    "disk_size_limit_bytes": 1234,
                }
            }
        }
    )

    assert calls == {
        "enable_disk_cache": False,
        "enable_memory_cache": False,
        "restrict_pickle": True,
        "disk_size_limit_bytes": 1234,
    }


def test_cache_config_is_skipped_when_absent(monkeypatch, _hermetic):
    import dspytools.core.setup as setup_mod

    calls: list = []
    monkeypatch.setattr(dspy, "configure_cache", lambda **kw: calls.append(kw))

    setup_mod._apply_cache_config({})
    setup_mod._apply_cache_config({"dspy": {"adapter": "baml"}})

    assert calls == []


def test_setup_applies_persisted_cache_policy(monkeypatch, _hermetic):
    calls: dict = {}
    monkeypatch.setattr(dspy, "configure_cache", lambda **kw: calls.update(kw))

    _run_setup_dspy(
        monkeypatch,
        _hermetic,
        cfg={"dspy": {"cache": {"enable_disk_cache": False}}},
    )

    assert calls["enable_disk_cache"] is False
    assert calls["restrict_pickle"] is True


def test_cli_cache_disable_persists_without_loosening_pickle(monkeypatch):
    import dspytools.commands.configure as configure_mod
    from dspytools.commands.configure import configure_cmd

    cfg: dict = {}
    saved: dict = {}
    monkeypatch.setattr(configure_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(configure_mod, "save_user_config", lambda c: saved.update(c))

    result = CliRunner().invoke(configure_cmd, ["cache", "disable"])

    assert result.exit_code == 0, result.output
    assert saved["dspy"]["cache"] == {
        "enable_disk_cache": False,
        "enable_memory_cache": False,
    }
    assert "restrict_pickle" not in saved["dspy"]["cache"]


def test_cli_cache_enable_persists_safe_types(monkeypatch):
    import dspytools.commands.configure as configure_mod
    from dspytools.commands.configure import configure_cmd

    cfg: dict = {}
    saved: dict = {}
    monkeypatch.setattr(configure_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(configure_mod, "save_user_config", lambda c: saved.update(c))

    result = CliRunner().invoke(
        configure_cmd,
        [
            "cache",
            "enable",
            "--safe-type",
            "dspy.adapters.types.reasoning.Reasoning",
        ],
    )

    assert result.exit_code == 0, result.output
    cache_cfg = saved["dspy"]["cache"]
    assert cache_cfg["enable_disk_cache"] is True
    assert cache_cfg["enable_memory_cache"] is True
    assert cache_cfg["safe_types"] == ["dspy.adapters.types.reasoning:Reasoning"]


def test_cli_cache_enable_rejects_bad_safe_type(monkeypatch):
    import dspytools.commands.configure as configure_mod
    from dspytools.commands.configure import configure_cmd

    monkeypatch.setattr(configure_mod, "load_config", lambda: {})
    monkeypatch.setattr(configure_mod, "save_user_config", lambda c: None)

    result = CliRunner().invoke(
        configure_cmd, ["cache", "enable", "--safe-type", "no_such_module:Nope"]
    )

    assert result.exit_code != 0
    assert "cannot resolve safe type" in result.output


# ── 5-6. LM caching defaults and fast path ────────────────────────────────


def test_explicit_model_lm_is_cached_by_default(monkeypatch, _hermetic):
    captured = _run_setup_dspy(monkeypatch, _hermetic, model="openai/gpt-4o")

    assert captured["lm"] is not None
    assert captured["lm"].cache is True


def test_explicit_cache_false_is_honoured(monkeypatch, _hermetic):
    import dspytools.core.setup as setup_mod

    monkeypatch.setattr(dspy, "configure", lambda **kwargs: None)
    lm = setup_mod.setup_dspy(model="openai/gpt-4o", cache=False)

    assert lm.cache is False


def test_fast_path_is_invalidated_by_temperature(monkeypatch, _hermetic):
    import dspytools.core.setup as setup_mod

    monkeypatch.setattr(dspy, "configure", lambda **kwargs: None)
    first = setup_mod.setup_dspy()
    again = setup_mod.setup_dspy()
    hot = setup_mod.setup_dspy(temperature=0.9)

    assert again is first
    assert hot is not first
    assert hot.kwargs["temperature"] == 0.9


# ── 7. cache clear ────────────────────────────────────────────────────────


def test_cache_clear_removes_both_cache_dirs(monkeypatch, tmp_path):
    import dspytools.config.settings as settings_mod
    from dspytools.commands.configure import configure_cmd

    analysis = tmp_path / "analysis"
    analysis.mkdir(parents=True)
    (analysis / "thresholds.json").write_text("{}", encoding="utf-8")

    lm_cache = tmp_path / "lm_cache"
    lm_cache.mkdir()
    for i in range(3):
        (lm_cache / f"entry{i}.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(settings_mod, "cache_dir", lambda: analysis)
    monkeypatch.setattr(dspy.clients, "DISK_CACHE_DIR", str(lm_cache))

    result = CliRunner().invoke(configure_cmd, ["cache", "clear"])

    assert result.exit_code == 0, result.output
    assert f"analysis cache cleared: {analysis} (1 entries removed)" in result.output
    assert f"LM cache cleared: {lm_cache} (3 entries removed)" in result.output
    assert not analysis.exists()
    assert not lm_cache.exists()


def test_cache_clear_reports_unset_lm_cache(monkeypatch, tmp_path):
    import dspytools.config.settings as settings_mod
    from dspytools.commands.configure import configure_cmd

    monkeypatch.setattr(settings_mod, "cache_dir", lambda: tmp_path / "analysis")
    monkeypatch.setattr(dspy.clients, "DISK_CACHE_DIR", None)

    result = CliRunner().invoke(configure_cmd, ["cache", "clear"])

    assert result.exit_code == 0, result.output
    assert "DISK_CACHE_DIR is unset" in result.output


def test_cache_show_reports_persisted_state(monkeypatch, tmp_path):
    import dspytools.commands.configure as configure_mod
    import dspytools.config.settings as settings_mod
    from dspytools.commands.configure import configure_cmd

    cfg: dict = {
        "dspy": {"cache": {"enable_disk_cache": False, "enable_memory_cache": True}}
    }
    monkeypatch.setattr(configure_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(settings_mod, "cache_dir", lambda: tmp_path / "analysis")
    monkeypatch.setattr(dspy.clients, "DISK_CACHE_DIR", str(tmp_path / "lm"))

    result = CliRunner().invoke(configure_cmd, ["cache", "show"])

    assert result.exit_code == 0, result.output
    assert "enable_disk_cache: False" in result.output
    assert "enable_memory_cache: True" in result.output
    assert "restrict_pickle: True" in result.output


# ── 8. response_format retry ──────────────────────────────────────────────


class _ResponseFormatRejectingEngine:
    """Rejects requests carrying response_format, accepts plain chat requests."""

    supported_params = {"response_format", "temperature", "max_tokens"}
    supports_response_schema = True
    supports_function_calling = False

    def __init__(self) -> None:
        self.requests: list = []

    def complete(self, request):
        self.requests.append(request)
        if request.config.response_format is not None:
            raise dspy.utils.exceptions.LMError(
                "This response_format type is unavailable now", model=request.model
            )
        return dspy.lm15.Response(
            id=None,
            model=request.model,
            message=dspy.lm15.Message.assistant("[[ ## a ## ]]\n42"),
            finish_reason="stop",
            usage=dspy.lm15.Usage(input_tokens=0, output_tokens=0, total_tokens=0),
        )


def test_response_format_retry_hits_lm_without_response_format(monkeypatch, _hermetic):
    from dspy.adapters.baml_adapter import BAMLAdapter

    _run_setup_dspy(monkeypatch, _hermetic)  # installs the JSONAdapter.__call__ patch

    engine = _ResponseFormatRejectingEngine()
    lm = dspy.LM("openai/gpt-4o", engine=engine, cache=False, model_type="chat")
    signature = dspy.Signature(
        {"q": (str, dspy.InputField()), "a": (str, dspy.OutputField())}
    )

    outputs = BAMLAdapter(use_native_function_calling=False)(
        lm, {"temperature": 0.0}, signature, [], {"q": "the question"}
    )

    assert outputs == [{"a": "42"}]
    assert len(engine.requests) == 2
    # First attempt asked for structured output; the retry did not.
    assert engine.requests[0].config.response_format is not None
    assert engine.requests[1].config.response_format is None


def test_response_format_retry_not_used_without_response_format(monkeypatch, _hermetic):
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.utils.exceptions import LMError

    class _FailingEngine:
        supported_params = {"temperature", "max_tokens"}
        supports_response_schema = False
        supports_function_calling = False

        def __init__(self) -> None:
            self.requests: list = []

        def complete(self, request):
            self.requests.append(request)
            raise LMError("boom", model=request.model)

    _run_setup_dspy(monkeypatch, _hermetic)

    engine = _FailingEngine()
    lm = dspy.LM("openai/gpt-4o", engine=engine, cache=False, model_type="chat")
    signature = dspy.Signature(
        {"q": (str, dspy.InputField()), "a": (str, dspy.OutputField())}
    )

    with pytest.raises(LMError):
        JSONAdapter()(lm, {"temperature": 0.0}, signature, [], {"q": "x"})

    # No response_format was ever set, so there is no retry.
    assert len(engine.requests) == 1
