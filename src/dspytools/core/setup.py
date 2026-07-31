"""LM setup, DSPy configuration, and API key loading.

Optimization 9: LMRegistry singleton — shares LM instances across modules,
ensuring DSPy's LM cache hits across multiple module calls.
Optimization 23: setup_dspy() guard — skips full init if already configured.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.config.env import load_env, merge_environ
from dspytools.config.settings import load_config

# Optimization 23: Track whether setup_dspy() has been called
_setup_configured = False
_baml_patched = False


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

        Keyed by (model, api_base) tuple to reuse connections.
        """
        resolved_model = model or "openai/gpt-4o"
        key = f"{resolved_model}|{api_base or ''}"

        if key in cls._instances:
            return cls._instances[key]

        lm_kw: dict[str, Any] = {
            "model": resolved_model,
            "temperature": kwargs.get("temperature", 0.2),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "cache": kwargs.get("cache", True),
        }
        if api_key:
            lm_kw["api_key"] = api_key
        if api_base:
            lm_kw["api_base"] = api_base
        # Pass through extra LM kwargs (e.g. chat_template_kwargs for thinking models)
        for k, v in kwargs.items():
            if k not in lm_kw and k not in ("model", "api_base", "api_key"):
                lm_kw[k] = v

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
            # Auto-supply dummy key for local LLM — LiteLLM requires non-empty
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
                **(
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": student_cfg["enable_thinking"]
                        }
                    }
                    if "enable_thinking" in student_cfg
                    else {}
                ),
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
            # Auto-supply dummy key for local LLM — prevents litellm from
            # picking up DEEPSEEK_API_KEY from env for local endpoints
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
                **(
                    {
                        "chat_template_kwargs": {
                            "enable_thinking": teacher_cfg["enable_thinking"]
                        }
                    }
                    if "enable_thinking" in teacher_cfg
                    else {}
                ),
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
    cache: bool = False,
    warn_type: bool = False,
) -> dspy.LM:
    """Configure DSPy with an LM. Uses LMRegistry singleton.

    Optimization 23: Returns cached LM immediately if already configured
    (unless caller explicitly passes model/api_base/api_key overrides).
    """
    global _setup_configured

    # Optimization 23: Fast path — already configured with default params
    if (
        _setup_configured
        and model is None
        and api_base is None
        and api_key is None
        and not cache
    ):
        return LMRegistry.get_or_default()

    env = load_env()
    merge_environ(env)

    # When no model specified, use the student model from config (auto-resolves api_base + api_key)
    if model is None:
        lm = LMRegistry.get_or_default()
        # Apply any explicit temperature/max_tokens overrides using same api_base/api_key
        if temperature != 0.2 or max_tokens != 4096 or cache:
            cfg = load_config()
            student = cfg.get("lm", {}).get("student", {})
            lm = LMRegistry.get(
                model=lm.model,
                api_base=student.get("api_base"),
                api_key=student.get("api_key"),
                temperature=temperature,
                max_tokens=max_tokens,
                cache=cache,
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

        # Auto-supply dummy key for local models (LLM/LMStudio) — LiteLLM requires non-empty
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
        # kills the call.  Fix: patch JSONAdapter.__call__ to catch LMError
        # where response_format was set, strip it, and retry via the plain
        # ChatAdapter path (text-based parsing, no structured output).
        from dspy.adapters.json_adapter import JSONAdapter as _JSONAdapter
        from dspy.utils.exceptions import LMError

        _orig_json_call = _JSONAdapter.__call__

        def _patched_json_call(self, lm, lm_kwargs, signature, demos, inputs):
            try:
                return _orig_json_call(self, lm, lm_kwargs, signature, demos, inputs)
            except LMError:
                # If response_format was set and the LM rejected it, retry
                # without it using the ChatAdapter text-parsing path.
                if lm_kwargs.get("response_format") is not None:
                    lm_kwargs_copy = dict(lm_kwargs)
                    lm_kwargs_copy.pop("response_format", None)
                    return super(_JSONAdapter, self).__call__(
                        lm, lm_kwargs_copy, signature, demos, inputs
                    )
                raise

        _JSONAdapter.__call__ = _patched_json_call

        # ReActV2 + Reasoning-field fix: ReActV2's react signature has output
        # fields [next_thought (dspy.Reasoning), tool_calls (dspy.ToolCalls)].
        # With native function calling, the provider returns tool_calls in the
        # API response and reasoning in reasoning_content — the JSON body the
        # model emits contains tool_calls but often omits next_thought.
        # JSONAdapter.parse() strict-requires every output field key, so the
        # parse fails and ReActV2's tool loop dies before executing tools.
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

            # Reasoning fields may be absent — populated from reasoning_content
            # or defaulted in _call_postprocess. Other fields must all be present.
            required = {
                k
                for k in signature.output_fields
                if signature.output_fields[k].annotation is not Reasoning
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

        # BAMLAdapter overrides parse() with its own strict key check — patch it
        # the same way so ReActV2's Reasoning fields are tolerated there too.
        from dspy.adapters.baml_adapter import BAMLAdapter as _BAMLAdapter2

        _BAMLAdapter2.parse = _parse_with_reasoning_tolerance
        _baml_patched = True

    # Read configured adapter from config, default to BAMLAdapter
    cfg = load_config()
    adapter_type = cfg.get("dspy", {}).get("adapter", "baml")

    adapters = {}
    # Always lazily import to avoid startup cost; BAMLAdapter is already imported
    # via the patch above, so reuse its module reference.
    from dspy.adapters.baml_adapter import BAMLAdapter as _BAMLAdapter

    adapters["baml"] = _BAMLAdapter(use_native_function_calling=False)

    if adapter_type != "baml":
        adapters["chat"] = dspy.ChatAdapter()
        adapters["json"] = dspy.JSONAdapter()
        adapters["xml"] = dspy.XMLAdapter()

    adapter = adapters.get(adapter_type) or adapters["baml"]

    try:
        dspy.configure(
            lm=lm,
            adapter=adapter,
            warn_on_type_mismatch=warn_type,
        )
    except RuntimeError as _exc:
        # DSPy v3.3.0b1 thread-safety: once dspy.settings.configure() is called
        # from one thread, only that thread may call it again.  If another thread
        # already owns settings (e.g. MLflow autolog in a worker, or a prior test
        # suite run), calling configure() here raises RuntimeError.
        # Fall back: set values directly in main_thread_config (Settings.__setattr__
        # delegates to configure(), but main_thread_config is a plain dotdict so
        # direct assignment bypasses the thread-safety check).
        import sys as _sys

        _cfg_mod = _sys.modules.get("dspy.dsp.utils.settings")
        if _cfg_mod is not None:
            _cfg_mod.main_thread_config["lm"] = lm
            _cfg_mod.main_thread_config["adapter"] = adapter
            _cfg_mod.main_thread_config["warn_on_type_mismatch"] = warn_type

    # Optimization 2: Lazy-load — create HotSwapManager but defer load_all() to first swap/infer.
    # The index is loaded lazily on first metadata access. This avoids reading 60+ JSON files
    # at CLI startup when the user only needs 1 program.
    from dspytools.core.hotswap import HotSwapManager

    HotSwapManager()

    # DSPy 3.x compatibility: TypedPredictor was removed, replaced by Predict
    # with typed signatures. AvatarOptimizer and other legacy code reference it.
    if not hasattr(dspy, "TypedPredictor"):
        dspy.TypedPredictor = dspy.Predict

    _setup_configured = True  # Optimization 23: Mark as configured
    return lm
