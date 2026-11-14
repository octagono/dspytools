"""LM setup, DSPy configuration, and API key loading.

Optimization 9: LMRegistry singleton — shares LM instances across modules,
ensuring DSPy's LM cache hits across multiple module calls.
Optimization 23: setup_dspy() guard — skips full init if already configured.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, get_args

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.config.env import load_env, merge_environ
from dspytools.config.settings import load_config

# Optimization 23: Track whether setup_dspy() has been called
_setup_configured = False
_baml_patched = False

# `model_type="auto"` routes OpenAI-compatible models to the /v1/responses
# endpoint.  Local llama.cpp servers only serve /v1/chat/completions, so "chat"
# is the project default.  Override with `dspy.model_type` in config
# ("auto"/"responses") for models that need it.
_DEFAULT_MODEL_TYPE = "chat"


def _model_type(
    cfg: dict, explicit: str | None = None, *, embedding: bool = False
) -> str | None:
    """Resolve the LM API family: explicit argument, then config, then "chat".

    LMRegistry is for chat/completion LMs; embeddings go through
    dspy.Embedder.  Embedding models therefore get no ``model_type`` default.
    """
    if explicit:
        return explicit
    configured = cfg.get("dspy", {}).get("model_type")
    if configured:
        return configured
    return None if embedding else _DEFAULT_MODEL_TYPE


def _reasoning_kwargs(section: dict) -> dict[str, Any]:
    """Translate thinking config into DSPy's ``reasoning_effort`` dial.

    DSPy rejects ``chat_template_kwargs`` (``LMUnsupportedFeatureError``: the
    canonical Request has no such field) and spells the reasoning dial
    ``reasoning_effort``, forwarded verbatim into the request body.

    - explicit ``reasoning_effort`` always wins;
    - legacy ``enable_thinking: false`` -> ``reasoning_effort="none"``;
    - legacy ``enable_thinking: true`` -> no kwarg (server default keeps
      thinking on);
    - ``enable_thinking`` absent -> nothing sent.
    """
    if section.get("reasoning_effort") is not None:
        return {"reasoning_effort": section["reasoning_effort"]}
    enable_thinking = section.get("enable_thinking")
    if enable_thinking is None:
        return {}
    return {} if enable_thinking else {"reasoning_effort": "none"}


# Keys of ``cfg["dspy"]`` that map 1:1 onto a ``dspy.configure()`` setting.
# Anything else in that section (``adapter``, ``model_type``, ``cache``,
# ``profile``) is consumed by this module directly and is never forwarded.
_RUNTIME_SETTING_KEYS = (
    "track_usage",
    "num_threads",
    "max_errors",
    "async_max_workers",
    "disable_history",
    "max_history_size",
    "allow_tool_async_sync_conversion",
    "provide_traceback",
    "warn_on_type_mismatch",
)


def resolve_safe_types(specs: Any) -> list[type]:
    """Resolve ``"module:Class"`` / ``"module.Class"`` spellings into classes.

    Used for ``dspy.configure_cache(safe_types=...)`` — the only widening knob
    on restricted cache deserialization.  Raises ``ValueError`` on a bad spec.
    """
    from importlib import import_module

    resolved: list[type] = []
    for spec in specs:
        separator = ":" if ":" in spec else "."
        module_name, _, attr = spec.rpartition(separator)
        if not module_name or not attr:
            raise ValueError(
                f"safe type must be 'module:Class' or 'module.Class', got {spec!r}"
            )
        try:
            obj = getattr(import_module(module_name), attr)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"cannot resolve safe type {spec!r}: {exc}") from exc
        if not isinstance(obj, type):
            raise ValueError(f"safe type {spec!r} is not a class")
        resolved.append(obj)
    return resolved


def _apply_cache_config(cfg: dict) -> None:
    """Apply persisted ``cfg["dspy"]["cache"]`` through ``dspy.configure_cache``.

    No section, no call: DSPy's own cache defaults stay in force.  Restricted
    deserialization is a project contract, so ``restrict_pickle`` is always
    True here — cache "disable" turns caching off, it never loosens this.
    """
    cache_cfg = cfg.get("dspy", {}).get("cache")
    if not cache_cfg:
        return

    kwargs: dict[str, Any] = {
        "enable_disk_cache": bool(cache_cfg.get("enable_disk_cache", True)),
        "enable_memory_cache": bool(cache_cfg.get("enable_memory_cache", True)),
        "restrict_pickle": True,
    }
    if cache_cfg.get("disk_size_limit_bytes") is not None:
        kwargs["disk_size_limit_bytes"] = int(cache_cfg["disk_size_limit_bytes"])
    if cache_cfg.get("safe_types"):
        kwargs["safe_types"] = resolve_safe_types(cache_cfg["safe_types"])
    dspy.configure_cache(**kwargs)


def _runtime_settings(cfg: dict) -> dict[str, Any]:
    """``cfg["dspy"]`` entries that are real ``dspy.configure()`` settings.

    Only keys actually present are returned: an unset key must not be injected,
    so DSPy's own default for it stays in force.

    ``track_usage`` is the one setting with a project default (``True`` — see
    :func:`_default_track_usage`).  It is applied through the settings object
    instead of being injected here, so the "never forward an unset key"
    contract for ``dspy.configure`` kwargs stays exact.
    """
    section = cfg.get("dspy", {})
    return {k: section[k] for k in _RUNTIME_SETTING_KEYS if k in section}


def _default_track_usage() -> None:
    """Turn DSPy usage tracking on unless config already set it.

    ``core.cost_tracker`` records *measured* tokens from
    ``Prediction.get_lm_usage()``, which DSPy only populates when
    ``settings.track_usage`` is true (verified against dspy 3.5.0: the getter
    returns ``None`` when tracking is off).  Without this default every cost
    report would silently fall back to ``PRICING`` estimates.

    Written straight into ``main_thread_config`` — the dict ``configure()``
    itself writes and every ``dspy.context`` copies.  A direct assignment is
    used deliberately: it neither injects an extra ``dspy.configure`` kwarg
    (which would break the "unset keys stay unset" contract) nor trips the
    configure-owner-thread ``RuntimeError`` when setup runs off the owner
    thread.  ``dspy.configure(track_usage=...)`` from config always wins
    because it is forwarded as a normal runtime setting.
    """
    import sys as _sys

    _cfg_mod = _sys.modules.get("dspy.dsp.utils.settings")
    if _cfg_mod is not None:
        _cfg_mod.main_thread_config["track_usage"] = True


def _lm_cache_key(model: str, lm_kw: dict[str, Any]) -> str:
    """Cache key covering every setting that changes generated output.

    Sharing LM instances across modules is what makes DSPy's own cache hit
    (Optimization 9), but two differently-configured LMs must never share one
    — e.g. ``reasoning_effort="none"`` vs. no dial at all.  ``api_key`` is
    excluded: it does not change output and does not belong in a dict key.
    """
    behavior = {k: v for k, v in lm_kw.items() if k not in ("api_key",)}
    return f"{model}|{json.dumps(behavior, sort_keys=True, default=str)}"


class LMRegistry:
    """Singleton LM instance registry.

    Optimization 9: Sharing LM instances ensures DSPy's built-in cache
    (when cache=True) actually hits across calls from different modules.
    Optimization 26: Teacher LM cached like student LM.
    """

    _instances: dict[str, dspy.LM] = {}
    _default: dspy.LM | None = None
    _teacher: dspy.LM | None = None  # Optimization 26: cached teacher

    @classmethod
    def get(
        cls,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> dspy.LM:
        """Get or create a cached LM instance.

        Keyed by model plus every generation setting that changes output.
        LMRegistry is for chat/completion LMs; embeddings go through
        dspy.Embedder, so models whose name contains "embed" get no
        ``model_type`` default injected.
        """
        resolved_model = model or "openai/gpt-4o"

        resolved_cache = kwargs.get("cache")
        lm_kw: dict[str, Any] = {
            "model": resolved_model,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 4096),
            # None means "unspecified" — keep DSPy's cached-by-default behavior.
            "cache": True if resolved_cache is None else resolved_cache,
        }
        model_type = _model_type(
            load_config(),
            kwargs.get("model_type"),
            embedding="embed" in resolved_model.lower(),
        )
        if model_type:
            lm_kw["model_type"] = model_type
        if api_key:
            lm_kw["api_key"] = api_key
        if api_base:
            lm_kw["api_base"] = api_base
        # Pass through extra LM kwargs (e.g. reasoning_effort for thinking models)
        for k, v in kwargs.items():
            if k not in lm_kw and k not in (
                "model",
                "api_base",
                "api_key",
                "model_type",
            ):
                lm_kw[k] = v

        key = _lm_cache_key(resolved_model, lm_kw)
        if key in cls._instances:
            return cls._instances[key]

        lm = dspy.LM(**lm_kw)  # type: ignore[arg-type]
        cls._instances[key] = lm
        return lm

    @classmethod
    def get_or_default(cls) -> dspy.LM:
        """Return the default LM or create one from config."""
        if cls._default is not None:
            return cls._default

        cfg = load_config()

        # Prefer student model from config
        student_cfg = cfg.get("lm", {}).get("student", {})
        if student_cfg:
            api_key = student_cfg.get("api_key")
            api_base = student_cfg.get("api_base")
            # Auto-supply dummy key for local LLM — the OpenAI-compatible client requires non-empty
            if (
                not api_key
                and api_base
                and ("localhost" in api_base or "127.0.0.1" in api_base)
            ):
                api_key = "sk-local"
            lm = cls.get(
                model=student_cfg.get("model"),
                api_base=api_base,
                api_key=api_key,
                temperature=student_cfg.get("temperature", 0.2),
                max_tokens=student_cfg.get("max_tokens", 4096),
                model_type=student_cfg.get("model_type"),
                **_reasoning_kwargs(student_cfg),
            )
            cls._default = lm
            return lm

        # Fall back to any configured model
        default_model = cfg.get("lm", {}).get("default")
        if default_model:
            lm = cls.get(model=default_model)
            cls._default = lm
            return lm

        # Absolute fallback
        lm = cls.get(model="openai/gpt-4o", cache=False)
        cls._default = lm
        return lm

    @classmethod
    def get_teacher(cls) -> dspy.LM | None:
        """Return the teacher LM from config, or auto-configure from .env.

        Optimization 26: Caches the teacher LM instance — avoids re-reading
        config and .env on every call.
        """
        if cls._teacher is not None:
            return cls._teacher

        cfg = load_config()
        teacher_cfg = cfg.get("lm", {}).get("teacher", {})
        if teacher_cfg:
            teacher_api_base = teacher_cfg.get("api_base")
            teacher_api_key = teacher_cfg.get("api_key")
            # Auto-supply dummy key for local LLM — keeps the remote provider key
            # from being sent to a local endpoint
            if (
                not teacher_api_key
                and teacher_api_base
                and ("localhost" in teacher_api_base or "127.0.0.1" in teacher_api_base)
            ):
                teacher_api_key = "sk-local"

            cls._teacher = cls.get(
                model=teacher_cfg.get("model"),
                api_base=teacher_api_base,
                api_key=teacher_api_key,
                temperature=teacher_cfg.get("temperature", 0.0),
                max_tokens=teacher_cfg.get("max_tokens", 4096),
                cache=True,
                model_type=teacher_cfg.get("model_type"),
                **_reasoning_kwargs(teacher_cfg),
            )
            return cls._teacher

        # Auto-config from .env: DeepSeek V4 Pro
        env = load_env()
        deepseek_key = env.get("DEEPSEEK_API_KEY") or os.environ.get(
            "DEEPSEEK_API_KEY", ""
        )
        if not deepseek_key:
            return None

        merge_environ(env)
        cls._teacher = cls.get(
            model="deepseek/deepseek-v4-pro",
            api_base="https://api.deepseek.com/v1",
            api_key=deepseek_key,
            temperature=0.0,
            max_tokens=4096,
            cache=True,
        )
        return cls._teacher

    @classmethod
    def clear(cls) -> None:
        cls._instances.clear()
        cls._default = None
        cls._teacher = None  # Optimization 26: clear teacher cache


def setup_dspy(
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.2,
    cache: bool | None = None,
    warn_type: bool | None = None,
) -> dspy.LM:
    """Configure DSPy with an LM. Uses LMRegistry singleton.

    Optimization 23: Returns cached LM immediately if already configured
    (unless caller explicitly passes model/api_base/api_key overrides, or a
    temperature/max_tokens/cache that differs from the previous call).

    Also applies the persisted ``cfg["dspy"]`` runtime settings and cache
    policy — only the keys actually present, so DSPy's defaults stand for the
    rest.  The one exception is ``track_usage``, which defaults to on so
    ``Prediction.get_lm_usage()`` reports real tokens for
    ``core.cost_tracker``; config can turn it off.
    """
    global _setup_configured

    # Optimization 23: Fast path — already configured with default params
    if (
        _setup_configured
        and model is None
        and api_base is None
        and api_key is None
        and cache is None
        and temperature == 0.2
        and max_tokens == 4096
    ):
        return LMRegistry.get_or_default()

    env = load_env()
    merge_environ(env)

    # When no model specified, use the student model from config (auto-resolves api_base + api_key)
    if model is None:
        lm = LMRegistry.get_or_default()
        # Apply any explicit temperature/max_tokens/cache overrides using same api_base/api_key
        if temperature != 0.2 or max_tokens != 4096 or cache is not None:
            cfg = load_config()
            student = cfg.get("lm", {}).get("student", {})
            lm = LMRegistry.get(
                model=lm.model,
                api_base=student.get("api_base"),
                api_key=student.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
                cache=cache,
                model_type=student.get("model_type"),
                **_reasoning_kwargs(student),
            )
    else:
        if api_key is None and model:
            provider = model.split("/")[0]
            var = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(var) or ""

        # Auto-resolve api_base from configured student model
        if api_base is None and model:
            cfg = load_config()
            student = cfg.get("lm", {}).get("student", {})
            if student.get("model") == model:
                api_base = student.get("api_base")
            registry = cfg.get("lm", {}).get("registry", {})
            for m, opts in registry.items():
                if m == model and opts.get("api_base"):
                    api_base = opts["api_base"]
                    break

        # Auto-supply dummy key for local models (LLM/LMStudio) — the client requires non-empty
        if (
            not api_key
            and api_base
            and ("localhost" in api_base or "127.0.0.1" in api_base)
        ):
            api_key = "sk-local"

        lm = LMRegistry.get(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )

    # Use BAMLAdapter (compact Pydantic schema) with native function calling
    # disabled.  Qwen3.5-9B 7B + llama-cpp-server doesn't support the OpenAI tool_calls
    # API response format — it returns function calls as plain text in `content`.
    # With use_native_function_calling=False, DSPy renders tool descriptions as
    # text prompts and parses tool calls from the model's text output.
    #
    # BAMLAdapter uses compact schema syntax (simplified Pydantic) rather than
    # full JSON Schema, which helps small 7B models understand output structures.
    #
    # One fix applied: dspy.Reasoning is a str-like type (its "str-like"
    # property is noted in the docstring), but BAMLAdapter's _render_type_str
    # detects it as a Pydantic BaseModel and expands it to {"content": string}
    # with a "# Reasoning type" docstring comment.  Qwen 7B reads the comment
    # and outputs "reasoning" instead of "next_thought" as the JSON key, causing
    # AdapterParseError.  The fix: patch BAMLAdapter's _render_type_str to
    # render Reasoning as plain "string".
    # Optimization: patch only once per process — _setup_configured guards
    # the fast path, but explicit model overrides also reach this code.
    global _baml_patched
    if not _baml_patched:
        from dspy.adapters import baml_adapter as _baml_mod

        _orig_render = _baml_mod._render_type_str

        def _patched_render(annotation, depth=0, indent=0, seen_models=None):
            # dspy.Reasoning is a str-like type — render as plain string
            # instead of expanding the Pydantic schema.
            if (
                annotation is not str
                and hasattr(annotation, "__name__")
                and annotation.__name__ == "Reasoning"
            ):
                return "string"
            return _orig_render(annotation, depth, indent, seen_models)

        _baml_mod._render_type_str = _patched_render

        # DeepSeek response_format fix: JSONAdapter.__call__ (which BAMLAdapter
        # inherits) sets lm_kwargs["response_format"] to a Pydantic model or
        # {"type": "json_object"} and passes it to the LM.  DeepSeek rejects
        # this with "This response_format type is unavailable now" on some
        # API key tiers.  The LMError propagates up through ChatAdapter and
        # kills the call.  Fix: catch that LMError and retry once through a
        # plain ChatAdapter, which never sets response_format and parses text
        # sections instead of structured output.
        #
        # The retry must be a different adapter instance: any call back through
        # JSONAdapter/BAMLAdapter re-enters _call_preprocess ->
        # _prepare_response_format, which re-injects response_format and would
        # re-send the request the provider just rejected.
        from dspy.adapters.json_adapter import JSONAdapter as _JSONAdapter
        from dspy.utils.exceptions import LMError

        _orig_json_call = _JSONAdapter.__call__

        def _patched_json_call(self, lm, lm_kwargs, signature, demos, inputs):
            try:
                return _orig_json_call(self, lm, lm_kwargs, signature, demos, inputs)
            except LMError:
                # Only response_format-related failures are retryable.
                if lm_kwargs.get("response_format") is None:
                    raise
                fallback_kwargs = {
                    k: v for k, v in lm_kwargs.items() if k != "response_format"
                }
                return dspy.ChatAdapter()(lm, fallback_kwargs, signature, demos, inputs)

        _JSONAdapter.__call__ = _patched_json_call

        # dspy.ReAct + Reasoning-field fix: ReAct's react signature has output
        # fields [next_thought (dspy.Reasoning), tool_calls (dspy.ToolCalls)].
        # With native function calling, the provider returns tool_calls in the
        # API response and reasoning in reasoning_content — the JSON body the
        # model emits contains tool_calls but often omits next_thought.
        # JSONAdapter.parse() strict-requires every output field key, so the
        # parse fails and ReAct's tool loop dies before executing tools.
        # Fix: patch JSONAdapter.parse() to treat missing dspy.Reasoning-typed
        # output fields as optional (they are filled from reasoning_content or
        # defaulted to None in _call_postprocess). Non-Reasoning missing fields
        # still raise.
        def _parse_with_reasoning_tolerance(self, signature, completion):
            import json_repair as _json_repair
            import regex as _regex
            from dspy.adapters.types.reasoning import Reasoning
            from dspy.adapters.utils import parse_value
            from dspy.utils.exceptions import AdapterParseError

            parsed = _json_repair.loads(completion)
            if not isinstance(parsed, dict):
                _pattern = r"\{(?:[^{}]|(?R))*\}"
                _m = _regex.search(_pattern, completion, _regex.DOTALL)
                if _m:
                    parsed = _json_repair.loads(_m.group(0))
            if not isinstance(parsed, dict):
                # BAMLAdapter renders output fields as a `[[ ## field ## ]]`
                # section template (see format_field_structure) and small local
                # models (Qwen) often echo that template back verbatim instead
                # of emitting a JSON object.  JSONAdapter.parse() can't handle
                # it, so fall back to ChatAdapter-style section parsing, which
                # extracts `[[ ## field ## ]]` sections by header.
                _chat_fields: dict[str, str] = {}
                _header_re = _regex.compile(r"\[\[ ## (\w+) ## \]\]")
                _section = None
                _buf: list[str] = []
                for _line in completion.splitlines():
                    _mh = _header_re.match(_line.strip())
                    if _mh:
                        if _section is not None and _section in signature.output_fields:
                            _chat_fields[_section] = "\n".join(_buf).strip()
                        _section = _mh.group(1)
                        _buf = []
                        _rest = _line[_mh.end() :].strip()
                        if _rest:
                            _buf.append(_rest)
                    else:
                        _buf.append(_line)
                if _section is not None and _section in signature.output_fields:
                    _chat_fields[_section] = "\n".join(_buf).strip()
                if _chat_fields:
                    parsed = {}
                    for _k, _v in _chat_fields.items():
                        if _k in signature.output_fields:
                            try:
                                parsed[_k] = parse_value(
                                    _v, signature.output_fields[_k].annotation
                                )
                            except Exception:
                                parsed[_k] = _v
            if not isinstance(parsed, dict):
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    message="LM response cannot be serialized to a JSON object.",
                )

            parsed = {k: v for k, v in parsed.items() if k in signature.output_fields}
            for k, v in parsed.items():
                if k in signature.output_fields:
                    parsed[k] = parse_value(v, signature.output_fields[k].annotation)

            # Stock JSONAdapter.parse ends with
            # apply_output_field_defaults(), which fills any omitted optional
            # output field (declared default / default factory / None-allowing
            # annotation). Our override must keep that behavior so defaulted or
            # nullable non-Reasoning fields no longer raise when the model omits
            # them — matching stock semantics before the strict check below.
            # Feature-detected: if the helper moves/renames in a future DSPy,
            # degrade gracefully (skip default-filling) instead of hard-crashing
            # setup on an unexpected version.
            try:
                from dspy.adapters.utils import apply_output_field_defaults
            except ImportError:
                apply_output_field_defaults = None
            if apply_output_field_defaults is not None:
                parsed = apply_output_field_defaults(signature, parsed)

            # Reasoning fields may be absent — populated from reasoning_content
            # or defaulted in _call_postprocess. Other fields must all be present.
            # A ChainOfThought rationale is typed as `dspy.Reasoning`
            # (str-like, but not a str instance); tolerate it bare or wrapped
            # in an Optional/Union annotation.
            required = {
                k
                for k in signature.output_fields
                if signature.output_fields[k].annotation is not Reasoning
                and Reasoning not in get_args(signature.output_fields[k].annotation)
            }
            if required and not required.issubset(set(parsed.keys())):
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed,
                )
            return parsed

        _JSONAdapter.parse = _parse_with_reasoning_tolerance

        # BAMLAdapter inherits JSONAdapter.parse(), but patch it explicitly as
        # well so ReAct's Reasoning fields are tolerated even if that changes.
        from dspy.adapters.baml_adapter import BAMLAdapter as _BAMLAdapter2

        _BAMLAdapter2.parse = _parse_with_reasoning_tolerance
        _baml_patched = True

    # Read configured adapter from config, default to BAMLAdapter
    cfg = load_config()
    adapter_type = str(cfg.get("dspy", {}).get("adapter", "baml")).strip().lower()

    # Always lazily import to avoid startup cost; BAMLAdapter is already imported
    # via the patch above, so reuse its module reference.
    from dspy.adapters.baml_adapter import BAMLAdapter as _BAMLAdapter

    # Build only the selected adapter (not all four) — the others would be
    # constructed and immediately discarded.  "baml" is the project's tuned
    # default; "chat"/"json"/"xml" are the stock DSPy adapters.  BAML, Chat and
    # XML are built with native function calling OFF because the local
    # llama.cpp server returns tool calls as plain text, not as API tool_calls
    # (the stock ChatAdapter()/XMLAdapter() already default to off).  JSON keeps
    # stock behavior (native FC on) since it is the structured-output path.
    if adapter_type == "chat":
        adapter = dspy.ChatAdapter()
    elif adapter_type == "json":
        adapter = dspy.JSONAdapter()
    elif adapter_type == "xml":
        adapter = dspy.XMLAdapter()
    else:
        # Default (and fallback for unknown/stale values): BAMLAdapter
        adapter = _BAMLAdapter(use_native_function_calling=False)

    # Persisted runtime settings + cache policy.  Only keys present in config
    # are forwarded — never synthesize a value the user did not set.
    _apply_cache_config(cfg)
    runtime_settings = _runtime_settings(cfg)
    if warn_type is not None:
        runtime_settings["warn_on_type_mismatch"] = warn_type

    # Usage tracking defaults ON so Prediction.get_lm_usage() is populated for
    # core.cost_tracker; a persisted "track_usage" wins via runtime_settings.
    if "track_usage" not in runtime_settings:
        _default_track_usage()

    try:
        dspy.configure(lm=lm, adapter=adapter, **runtime_settings)
    except RuntimeError:
        # DSPy thread-safety: once dspy.settings.configure() is called from one
        # thread, only that thread may call it again.  If another thread already
        # owns settings (e.g. MLflow autolog in a worker, or a prior test suite
        # run), calling configure() here raises RuntimeError.
        # Fall back: set values directly in main_thread_config (Settings.__setattr__
        # delegates to configure(), but main_thread_config is a plain dotdict so
        # direct assignment bypasses the thread-safety check).
        import sys as _sys

        _cfg_mod = _sys.modules.get("dspy.dsp.utils.settings")
        if _cfg_mod is not None:
            _cfg_mod.main_thread_config["lm"] = lm
            _cfg_mod.main_thread_config["adapter"] = adapter
            _cfg_mod.main_thread_config.update(runtime_settings)

    # Optimization 2: Lazy-load — create HotSwapManager but defer load_all() to first swap/infer.
    # The index is loaded lazily on first metadata access. This avoids reading 60+ JSON files
    # at CLI startup when the user only needs 1 program.
    from dspytools.core.hotswap import HotSwapManager

    HotSwapManager()

    _setup_configured = True  # Optimization 23: Mark as configured
    return lm
