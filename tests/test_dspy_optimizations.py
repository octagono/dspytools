"""DSPy 3.5 optimization features: retry taxonomy, real token accounting,
parallel tool calls and CLI token streaming.

Everything is offline: ``dspy.utils.DummyLM`` or a stub ``complete(Request) ->
Response`` engine, ``setup_dspy()`` patched out of the command layer, and no
config/env/socket access.  ``time.sleep`` is always monkeypatched, so the retry
tests finish instantly.

Coverage:

1. ``core.retry`` retries DSPy's transient LM errors (honouring a capped
   ``retry_after``) and never retries context-window/auth/billing errors.
2. ``setup_dspy()`` enables ``track_usage`` by default (config still wins) so
   ``Prediction.get_lm_usage()`` reports real tokens; ``core.cost_tracker``
   records measured usage without breaking its estimate-based API.
3. ``--parallel-tools`` clones the active adapter with provider-side parallel
   tool calling on, only inside its own context, and fails loudly when the LM
   cannot do native function calling.
4. ``run predict|cot --stream`` prints output-field chunks and still renders the
   result panel; a non-streaming adapter falls through to the panel.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from dspytools.commands import agent as agent_mod
from dspytools.commands import run as run_mod
from dspytools.core import cost_tracker
from dspytools.core import retry as retry_mod
from dspytools.core._dspy import dspy

# ── Hermetic scaffolding ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_setup_dspy(monkeypatch):
    """Never read config or build a real LM from the command layer."""
    monkeypatch.setattr(run_mod, "setup_dspy", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent_mod, "setup_dspy", lambda *args, **kwargs: None)


@pytest.fixture
def _restore_settings():
    """Restore ``dspy``'s global config after a test mutates it."""
    from dspy.dsp.utils.settings import main_thread_config

    saved = dict(main_thread_config)
    yield
    main_thread_config.clear()
    main_thread_config.update(saved)


@pytest.fixture(autouse=True)
def _instant_sleep(monkeypatch):
    """Record retry delays instead of waiting for them."""
    delays: list[float] = []

    def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(retry_mod.time, "sleep", fake_sleep)
    return delays


class _UsageEngine:
    """Stub sync engine reporting fixed token counts."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.requests: list = []

    def complete(self, request):
        self.requests.append(request)
        return dspy.lm15.Response(
            id=None,
            model=request.model,
            message=dspy.lm15.Message.assistant([dspy.lm15.TextPart(self.text)]),
            finish_reason="stop",
            usage=dspy.lm15.Usage(
                input_tokens=self.prompt_tokens,
                output_tokens=self.completion_tokens,
                total_tokens=self.prompt_tokens + self.completion_tokens,
            ),
        )


def _usage_lm(text: str = "<answer>\nParis\n</answer>", prompt=120, completion=42):
    engine = _UsageEngine(text, prompt, completion)
    return dspy.LM("openai/stub-model", engine=engine, cache=False), engine


class _FunctionCallingDummy(dspy.utils.DummyLM):
    """DummyLM that advertises native function calling."""

    @property
    def supports_function_calling(self) -> bool:
        return True


class _NoParallelAdapter(dspy.ChatAdapter):
    """Adapter honouring DSPy 3.5's older constructor (no parallel flag)."""

    def __init__(self, callbacks=None, use_native_function_calling: bool = False):
        super().__init__(
            callbacks=callbacks, use_native_function_calling=use_native_function_calling
        )


# ═══════════════════════════════════════════════════════════════════════════
#  1. Retry taxonomy
# ═══════════════════════════════════════════════════════════════════════════


def test_retry_defaults_cover_dspy_transient_errors():
    types = retry_mod.retryable_exceptions()

    for transient in (
        dspy.LMRateLimitError,
        dspy.LMTimeoutError,
        dspy.LMServerError,
        dspy.LMTransportError,
        dspy.LMLockTimeoutError,
        TimeoutError,
        ConnectionError,
        OSError,
    ):
        assert issubclass(transient, types), transient.__name__


def test_retry_never_includes_context_window_or_auth_errors():
    types = retry_mod.retryable_exceptions()

    for never in (
        dspy.ContextWindowExceededError,
        dspy.LMInvalidRequestError,
        dspy.LMAuthError,
        dspy.LMBillingError,
    ):
        assert not issubclass(never, types), never.__name__
    assert not issubclass(dspy.LMUnexpectedError, types)


def test_lm_exception_taxonomy_is_read_from_the_installed_dspy():
    """The classifier is upstream's, not a hardcoded copy."""
    from dspy.utils import exceptions as lm_exc

    assert set(retry_mod.retryable_exceptions()) >= set(lm_exc._RETRYABLE_LM_ERRORS)
    known = {value for value in vars(lm_exc).values() if isinstance(value, type)}
    assert set(retry_mod.non_retryable_exceptions()) <= known
    assert lm_exc.LMError not in retry_mod.non_retryable_exceptions()


def test_retry_honours_retry_after_and_caps_it(_instant_sleep):
    calls: list[int] = []

    @retry_mod.retry(max_retries=2, base_delay=2.0, max_delay=10.0)
    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise dspy.LMRateLimitError("slow down", retry_after=7.5)
        if len(calls) == 2:
            raise dspy.LMRateLimitError("still slow", retry_after=999.0)
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3
    assert _instant_sleep == [7.5, 10.0]


def test_retry_falls_back_to_exponential_backoff_without_a_hint(_instant_sleep):
    calls: list[int] = []

    @retry_mod.retry(max_retries=2, base_delay=2.0, max_delay=60.0, backoff_factor=2.0)
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise dspy.LMTransportError("connection reset")
        return "ok"

    assert flaky() == "ok"
    assert _instant_sleep == [2.0, 4.0]


def test_context_window_error_is_not_retried_even_when_tuple_is_widened():
    calls: list[int] = []

    @retry_mod.retry(max_retries=3, retryable_exceptions=(Exception,))
    def boom():
        calls.append(1)
        raise dspy.ContextWindowExceededError(model="openai/stub-model")

    with pytest.raises(dspy.ContextWindowExceededError):
        boom()
    assert len(calls) == 1


def test_auth_error_is_not_retried():
    calls: list[int] = []

    @retry_mod.retry(max_retries=3)
    def boom():
        calls.append(1)
        raise dspy.LMAuthError("bad key", status=401)

    with pytest.raises(dspy.LMAuthError):
        boom()
    assert len(calls) == 1


def test_non_dspy_exception_propagates_immediately():
    calls: list[int] = []

    @retry_mod.retry(max_retries=3)
    def boom():
        calls.append(1)
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        boom()
    assert len(calls) == 1


def test_compile_with_retry_reports_stats_and_honours_retry_after(_instant_sleep):
    calls: list[int] = []

    def flaky_compile():
        calls.append(1)
        if len(calls) == 1:
            raise dspy.LMServerError("provider 500", retry_after=3.0)
        return "compiled"

    result, stats = retry_mod.compile_with_retry(
        flaky_compile, max_retries=2, base_delay=2.0
    )

    assert result == "compiled"
    assert stats["attempts"] == 2
    assert stats["retries"] == 1
    assert stats["total_delay"] == 3.0
    assert _instant_sleep == [3.0]


def test_compile_with_retry_keeps_retrying_runtime_errors():
    calls: list[int] = []

    def oom_then_ok():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory")
        return "compiled"

    result, stats = retry_mod.compile_with_retry(oom_then_ok, max_retries=1)
    assert result == "compiled"
    assert stats["retries"] == 1


def test_retry_after_seconds_ignores_invalid_hints():
    assert retry_mod.retry_after_seconds(dspy.LMRateLimitError("x")) is None
    assert retry_mod.retry_after_seconds(ValueError("x")) is None
    assert (
        retry_mod.retry_after_seconds(dspy.LMRateLimitError("x", retry_after=-1))
        is None
    )
    assert (
        retry_mod.retry_after_seconds(dspy.LMRateLimitError("x", retry_after=True))
        is None
    )
    assert (
        retry_mod.retry_after_seconds(dspy.LMRateLimitError("x", retry_after=2)) == 2.0
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Real token accounting
# ═══════════════════════════════════════════════════════════════════════════


def _run_setup_dspy(monkeypatch, cfg: dict):
    """Call ``setup_dspy()`` hermetically; return the ``dspy.configure`` kwargs."""
    import dspytools.core.setup as setup_mod

    monkeypatch.setattr(setup_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(setup_mod, "load_env", dict)
    monkeypatch.setattr(setup_mod, "merge_environ", lambda env=None: None)
    monkeypatch.setattr(setup_mod, "_setup_configured", False)
    lm, _ = _usage_lm()
    monkeypatch.setattr(
        setup_mod.LMRegistry, "get_or_default", classmethod(lambda cls: lm)
    )
    captured: dict = {}
    monkeypatch.setattr(dspy, "configure", lambda **kwargs: captured.update(kwargs))
    setup_mod.setup_dspy()
    return captured


def test_setup_dspy_enables_usage_tracking_by_default(monkeypatch, _restore_settings):
    captured = _run_setup_dspy(monkeypatch, {"dspy": {}})

    # Forwarded as a real configure() kwarg rather than written into DSPy's
    # private settings dict, so the public API path is the only one used.
    assert captured["track_usage"] is True


def test_config_can_still_disable_usage_tracking(monkeypatch, _restore_settings):
    from dspy.dsp.utils.settings import main_thread_config

    main_thread_config.pop("track_usage", None)
    captured = _run_setup_dspy(monkeypatch, {"dspy": {"track_usage": False}})

    assert captured["track_usage"] is False


def test_prediction_reports_real_usage_when_tracking_is_on():
    lm, _ = _usage_lm()
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(), track_usage=True):

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        prediction = dspy.Predict(QA)(question="capital of France?")

    usage = prediction.get_lm_usage()
    assert usage and "openai/stub-model" in usage
    entry = usage["openai/stub-model"]
    assert entry["prompt_tokens"] > 0
    assert entry["completion_tokens"] > 0


def test_usage_tracking_off_reports_none():
    lm, _ = _usage_lm()
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(), track_usage=False):

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        prediction = dspy.Predict(QA)(question="capital of France?")

    assert prediction.get_lm_usage() is None
    assert cost_tracker.usage_from_result(prediction) == {}


def test_add_result_records_measured_usage():
    cost = cost_tracker.CompileCost(compile_id="run-1", optimizer="gepa")
    usage = {"openai/stub-model": {"prompt_tokens": 120, "completion_tokens": 42}}

    assert cost.add_result({"unused": True}, lm=None) == 0
    added = cost.add_usage(usage)

    assert added == 1
    assert cost.total_tokens == 162
    assert cost.calls[0].measured is True
    summary = cost.summary()
    assert summary["measured_calls"] == 1
    assert summary["estimated_calls"] == 0


def test_add_result_falls_back_to_lm_history():
    lm, _ = _usage_lm(prompt=10, completion=5)
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(), track_usage=False):

        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        prediction = dspy.Predict(QA)(question="hi")

    cost = cost_tracker.CompileCost(compile_id="run-2", optimizer="mipro")
    assert cost.add_result(prediction, lm) == 1
    assert cost.total_tokens > 0
    assert cost.calls[0].measured is True


def test_cached_call_records_nothing():
    class Cached:
        history = [{"model": "openai/stub-model", "usage": None, "cost": None}]

    cost = cost_tracker.CompileCost(compile_id="run-3", optimizer="mipro")
    assert cost.add_result(object(), Cached()) == 0
    assert cost.total_tokens == 0
    assert cost.calls == []


def test_zero_usage_entries_are_skipped():
    cost = cost_tracker.CompileCost(compile_id="run-4", optimizer="simba")
    assert cost.add_usage({"openai/stub-model": {"prompt_tokens": 0}}) == 0
    assert cost.calls == []


def test_provider_reported_cost_wins_over_pricing():
    usage = cost_tracker.TokenUsage.from_usage(
        "openai/stub-model", {"prompt_tokens": 100, "completion_tokens": 100}, cost=1.25
    )
    assert usage.cost_estimate == 1.25
    assert usage.measured is True


def test_estimate_api_is_unchanged():
    usage = cost_tracker.TokenUsage.estimate(
        "deepseek/deepseek-v4-flash", 500_000, 500_000
    )
    assert usage.total_tokens == 1_000_000
    assert usage.measured is False

    cost = cost_tracker.CompileCost(compile_id="run-5", optimizer="mipro")
    cost.add_call("deepseek/deepseek-v4-flash", 100, 50)
    assert cost.summary()["calls"] == 1
    assert cost.summary()["estimated_calls"] == 1


def test_usage_from_result_tolerates_non_predictions():
    assert cost_tracker.usage_from_result(None) == {}
    assert cost_tracker.usage_from_result({"answer": "4"}) == {}
    assert cost_tracker.usage_from_lm_history(object()) is None


# ═══════════════════════════════════════════════════════════════════════════
#  3. Parallel tool calls
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "adapter",
    [
        dspy.ChatAdapter(),
        dspy.JSONAdapter(),
        dspy.XMLAdapter(),
    ],
)
def test_parallel_adapter_clone_sets_both_flags(adapter):
    clone = run_mod._parallel_tools_adapter(adapter)

    assert clone is not adapter
    assert type(clone) is type(adapter)
    assert clone.parallel_tool_calls is True
    assert clone.use_native_function_calling is True
    # The source adapter is untouched.
    assert adapter.parallel_tool_calls is None


def test_parallel_adapter_clone_preserves_the_configured_baml_flavour():
    from dspy.adapters.baml_adapter import BAMLAdapter

    adapter = BAMLAdapter(use_native_function_calling=False)
    clone = run_mod._parallel_tools_adapter(adapter)

    assert type(clone) is BAMLAdapter
    assert clone.parallel_tool_calls is True
    assert clone.use_native_function_calling is True


def test_parallel_adapter_rejects_an_adapter_without_the_option():
    with pytest.raises(run_mod.click.ClickException) as err:
        run_mod._parallel_tools_adapter(_NoParallelAdapter())
    assert "parallel_tool_calls" in str(err.value)


def test_parallel_adapter_requires_a_configured_adapter():
    with pytest.raises(run_mod.click.ClickException):
        run_mod._parallel_tools_adapter(None)


def test_parallel_context_is_a_noop_when_disabled():
    before = dspy.settings.adapter
    with run_mod._parallel_tools_context(False):
        assert dspy.settings.adapter is before
    assert dspy.settings.adapter is before


def test_parallel_context_requires_function_calling():
    lm = dspy.utils.DummyLM([{"answer": "4"}])
    assert lm.supports_function_calling is False
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        with pytest.raises(run_mod.click.ClickException) as err:
            with run_mod._parallel_tools_context(True):
                pass
    assert "native function calling" in str(err.value)


def test_parallel_context_swaps_the_adapter_only_inside_the_scope():
    lm = _FunctionCallingDummy([{"answer": "4"}])
    outer = dspy.ChatAdapter()
    with dspy.context(lm=lm, adapter=outer):
        with run_mod._parallel_tools_context(True):
            inner = dspy.settings.adapter
            assert inner is not outer
            assert inner.parallel_tool_calls is True
            assert inner.use_native_function_calling is True
        assert dspy.settings.adapter is outer


class _AdapterRecorder(dspy.Module):
    """Records the active adapter at call time, returns a Prediction."""

    def __init__(self, sink: list):
        super().__init__()
        self.sink = sink

    def forward(self, **kwargs):
        self.sink.append(dspy.settings.adapter)
        return dspy.Prediction(answer="4", termination_reason="submit", history=None)


def _invoke(args: list[str], lm) -> object:
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        return CliRunner().invoke(run_mod.run_cmd, args)


def test_run_react_parallel_flag_reaches_the_adapter(monkeypatch):
    sink: list = []
    monkeypatch.setattr(run_mod.dspy, "ReAct", lambda *a, **k: _AdapterRecorder(sink))
    lm = _FunctionCallingDummy([{"answer": "4"}])

    result = _invoke(
        ["react", "question -> answer", "-i", "question=2+2", "--parallel-tools"], lm
    )

    assert result.exit_code == 0, result.output
    assert len(sink) == 1
    assert sink[0].parallel_tool_calls is True
    assert sink[0].use_native_function_calling is True
    assert "Parallel tool calls" in result.output


def test_run_react_without_the_flag_keeps_the_global_adapter(monkeypatch):
    sink: list = []
    monkeypatch.setattr(run_mod.dspy, "ReAct", lambda *a, **k: _AdapterRecorder(sink))
    lm = _FunctionCallingDummy([{"answer": "4"}])

    result = _invoke(["react", "question -> answer", "-i", "question=2+2"], lm)

    assert result.exit_code == 0, result.output
    assert sink[0].parallel_tool_calls is None
    assert "Parallel tool calls" not in result.output


def test_react_v2_exposes_the_same_flag(monkeypatch):
    sink: list = []
    monkeypatch.setattr(run_mod.dspy, "ReAct", lambda *a, **k: _AdapterRecorder(sink))
    lm = _FunctionCallingDummy([{"answer": "4"}])

    result = _invoke(
        ["react-v2", "question -> answer", "-i", "question=2+2", "--parallel-tools"],
        lm,
    )

    assert result.exit_code == 0, result.output
    assert sink[0].parallel_tool_calls is True


def test_agent_run_parallel_flag_reaches_the_adapter(monkeypatch, tmp_path):
    sink: list = []
    monkeypatch.setattr(run_mod.dspy, "ReAct", lambda *a, **k: _AdapterRecorder(sink))
    monkeypatch.setattr(agent_mod, "agents_dir", lambda: tmp_path)
    monkeypatch.setattr(agent_mod, "load_mcp_tools_sync", lambda cfg: ([], []))
    (tmp_path / "demo.json").write_text(
        '{"name": "demo", "signature": "question -> answer", "max_iters": 3, '
        '"mcp_config": ".mcp.json", "tool_count": 0, "tool_names": []}',
        encoding="utf-8",
    )
    lm = _FunctionCallingDummy([{"answer": "4"}])

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = CliRunner().invoke(
            agent_mod.agent_cmd, ["run", "demo", "2+2", "--parallel-tools"]
        )

    assert result.exit_code == 0, result.output
    assert len(sink) == 1
    assert sink[0].parallel_tool_calls is True


# ═══════════════════════════════════════════════════════════════════════════
#  4. CLI token streaming
# ═══════════════════════════════════════════════════════════════════════════


def _dummy_lm(**answers) -> dspy.utils.DummyLM:
    return dspy.utils.DummyLM([answers] * 4)


def test_predict_stream_prints_chunks_before_the_result_panel():
    lm = _dummy_lm(answer="Paris is the capital")
    with dspy.context(lm=lm):
        result = CliRunner().invoke(
            run_mod.run_cmd,
            [
                "predict",
                "question -> answer",
                "-i",
                "question=capital of France",
                "--adapter",
                "xml",
                "--stream",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Streaming:" in result.output
    streamed = result.output.index("Paris is the capital")
    panel = result.output.index("Result")
    assert streamed < panel
    # The final panel is still rendered after the stream.
    assert result.output.count("Paris is the capital") >= 2


def test_cot_stream_targets_the_last_output_field():
    module = dspy.ChainOfThought("question -> answer")
    assert run_mod._stream_output_field(module) == "answer"

    lm = _dummy_lm(plan="think", answer="Paris")
    with dspy.context(lm=lm):
        result = CliRunner().invoke(
            run_mod.run_cmd,
            [
                "cot",
                "question -> answer",
                "-i",
                "question=capital",
                "--adapter",
                "xml",
                "--stream",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Streaming: field 'answer'" in result.output
    assert result.output.index("Paris") < result.output.index("Result")


def test_stream_is_off_by_default():
    lm = _dummy_lm(answer="Paris")
    with dspy.context(lm=lm):
        result = CliRunner().invoke(
            run_mod.run_cmd,
            [
                "predict",
                "question -> answer",
                "-i",
                "question=capital",
                "--adapter",
                "xml",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Streaming:" not in result.output
    assert result.output.count("Paris") == 1


def test_non_streaming_adapter_still_renders_the_result_panel():
    """ChatAdapter's single chunk carries no terminator: no hang, panel renders."""
    lm = _dummy_lm(answer="Paris")
    with dspy.context(lm=lm):
        result = CliRunner().invoke(
            run_mod.run_cmd,
            [
                "predict",
                "question -> answer",
                "-i",
                "question=capital",
                "--adapter",
                "chat",
                "--stream",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "no streamed chunks" in result.output
    assert "Result" in result.output
    assert "Paris" in result.output


def test_stream_requires_an_output_field():
    class OnlyInput(dspy.Signature):
        question: str = dspy.InputField()

    module = dspy.Predict(OnlyInput)
    assert run_mod._stream_output_field(module) is None
    with pytest.raises(run_mod.click.ClickException):
        run_mod._run_streaming(module, {}, adapter=dspy.ChatAdapter())
