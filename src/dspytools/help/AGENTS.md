# DOX — help

## Purpose

Self-optimizing `--help` for the dspytools CLI. Replaces static Click help text with a DSPy program that generates contextual help using a local llama-cpp-server LLM. The compiled program is auto-loaded from cache on first `--help` invocation; auto-compilation is deferred to avoid blocking startup.

## Ownership

This directory owns the full help pipeline: CLI introspection (context gathering), DSPy module definition, auto-compilation and caching, and the `SelfOptimizingCLI` override that plugs into Click's help dispatch.

## Local Contracts

### `__init__.py` — HelpManager + SelfOptimizingCLI

- `SelfOptimizingCLI.get_help()` first attempts `HelpManager.get_answer()`. If it returns a non-None answer, print it with `rich.Panel` and return `""`. Otherwise fall back to `click.Group.get_help()`.
- `HelpManager.get_answer()` must **never** call `ctx.get_help()` — doing so creates a recursive loop (`get_help` → `get_answer` → `get_help` → ...). Return `None` instead.
- `HelpManager.get_answer()` must call `dspy.configure(lm=LMRegistry.get_or_default())` before invoking the DSPy module for inference. The cache load (`module.load()`) happens in `_init_module` / `compile_if_needed` and is separate from LM config.
- `HelpManager.init(cli)` must be called before any help request to register the root Click group.

### `module.py` — HelpModule

- `HelpModule` is a `dspy.Module` using `dspy.ChainOfThought` with signature `command, subcommands, examples -> answer: str`.
- The forward method passes three string inputs (`command`, `subcommands`, `examples`) and returns a `dspy.Prediction` with an `answer` attribute.
- No hardcoded command knowledge — all context comes from the caller (via `context.py`).
- Uses `from dspytools.core._dspy import dspy` for lazy DSPy import; do not `import dspy` directly.

### `context.py` — CLI introspection

- `get_all_commands(cli)` introspects the Click command tree and returns a dict keyed by command name with metadata (description, subcommands, options, examples, help_text).
- `build_trainset_from_cli(cli)` builds a list of `dspy.Example` objects with `command`, `subcommands`, `examples` inputs and `answer` output. One example per top-level command and one per subcommand.
- `_format_subcommands()` and `_build_examples()` produce the string representations fed to the DSPy module.
- Internal helper `_build_sub_examples()` contains a small hardcoded table of task-specific examples for common commands (configure, compile, run, tool, agent).

### `optimize.py` — AutoCompiler

- `compile_if_needed()` **must return `None` when no cache exists**. It must not block on first `--help` by triggering a compile. Callers check for `None` and fall through to Click's default help.
- `force_compile()` triggers a full re-compile regardless of cache state. Used by `dspytools self optimize`.
- Cache path: `~/.config/dspytools/help_compiled.json`. Metadata path: `~/.config/dspytools/help_meta.json`.
- Two optimization modes:
  - **Quick** (default): `dspy.LabeledFewShot(k=min(4, len(trainset)))` — no LM calls during compile.
  - **Teacher** (`use_teacher=True`): `dspy.GEPA` with `LMRegistry.get_teacher()` as reflection LM. Uses `max_metric_calls=50, num_threads=1` to avoid litellm thread pool exhaustion on local models. Compatible with any Open AI-compatible LM including local 7B models (~10s compile). For production use, prefer DeepSeek-class remote models.
- GEPA splits 20% of the trainset as valset (seed 42) to suppress the "no valset" warning.
- `clear()` removes both cache and meta files.

### Shared constraints

- All files use `from dspytools.core._dspy import dspy` for lazy DSPy import.
- All files use `from __future__ import annotations`.

## Work Guidance

- Keep `HelpModule` generic — no command-specific logic. Command knowledge lives in `context.py` or the compiled program.
- If adding new fields to the DSPy signature, update both `HelpModule.forward()` and `context.py`'s `build_trainset_from_cli()` in lockstep.
- Cache invalidation: the compiled program is keyed by the Click command tree structure. If commands change significantly, run `dspytools self optimize` or delete `~/.config/dspytools/help_compiled.json`.
- The recursive-loop contract (`get_answer` must not call `get_help`) is critical. Any code path from `get_answer` that could reach Click's help formatter must return early.

## Verification

```
ruff check --fix --unsafe-fixes src/dspytools/help/
```

Must pass with zero errors.

## Child DOX Index

No subdirectories — this directory contains only four `.py` files and this `AGENTS.md`.
