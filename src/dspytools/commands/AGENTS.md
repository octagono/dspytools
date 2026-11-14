# DOX — commands

## Purpose

CLI subcommand groups that implement the `dspytools` command surface. Each `.py` file is a Click command group registered by `main.py` (SelfOptimizingCLI).

## Ownership

Each file owns its command group. No file-level cross-dependencies — commands may import from `dspytools.core.*`, `dspytools.config.*`, and other dspytools subpackages but must not import from sibling command files.

## Local Contracts

### All commands MUST

- Call `setup_dspy()` before any DSPy operation that touches LM calls or optimizer compile.
- Use `from dspytools.core._dspy import dspy` for lazy DSPy import — do not `import dspy` directly.
- Use `from dspytools.core.setup import LMRegistry` to access configured LMs (`LMRegistry.get_teacher()`, `LMRegistry.get_student()`).
- Follow the role split: **teacher** LM for optimization/reflection, **student** LM for inference.
- Accept `--label` on commands that produce run artifacts (create_run_dir, save_program, register_run).

### Async compile contract

- `compile.py` owns the async pattern: `submit` (returns job_id), `status` (poll), `list`, `cancel`.
- Sync compile commands (knn, mipro, gepa, copro, simba, etc.) are legacy fallbacks; new optimizer commands should prefer async submission via `CompileScheduler.submit()`.
- All async jobs run through `dspytools.core.scheduler.CompileScheduler`.

### File index

24 command files (alphabetical, 24 Click groups):

| File | Group | Scope |
|------|-------|-------|
| `agent.py` | `agent` | ReAct agent management — create, list, run agents. `agent run` accepts `--agent-type {react,react-v2}` (default `react`; `react-v2` is a back-compat alias — both build `dspy.ReAct`, the structured loop with `History` + typed `submit`) and reports `termination_reason` + turn count |
| `compare.py` | `compare` | A/B program comparison — side-by-side evaluation with bootstrap p-value significance |
| `compile.py` | `compile` | OptimizerRegistry factory generates 10 commands from `_OPTIMIZER_SPECS` (knn, mipro, gepa, copro, simba, bootstrap-few-shot/{random,optuna}, labeled-few-shot, infer-rules). 12 explicit commands: submit/status/list/cancel (async), cost (cost/lineage lookup + per-optimizer savings), better-together, ensemble, finetune, gfl (--halving, --auto-suggest, --validate, --optimizers), grpo, avatar, distill (multi-step/teacher-dependent). New: `flex` (async) — wraps a module in `dspy.Flex` + GEPA with `gepa_metric_cost_aware`; saves `module_type="flex"` for hotswap. GEPA paths expose `--gepa-kwarg` (constructor escape hatch, validated against `dspy.GEPA.__init__`), `--gepa-kwargs` (passthrough to `gepa.optimize`), `--use-mlflow`, and the graduated knobs `--seed`, `--use-merge`/`--no-use-merge`, `--component-selector`, `--perfect-score`, `--track-stats`, `--track-best-outputs`. Kwarg validation is **signature-derived** (`_gepa_init_params`, `_build_gepa`) — never a hardcoded version list; `max_reflection_cost` / `reflection_prompt_template` are rejected because they still raise upstream. `BootstrapFinetune` passes `train_kwargs={"n_epochs": ...}` (its `num_iters` kwarg is gone in 3.5); `avatar` fails fast because upstream `dspy.AvatarOptimizer` still calls the removed `dspy.TypedPredictor`. The `gfl` comparison output includes a **cost/quality table** (per-optimizer est. $ and $/score via the shared `_estimate_optimizer_cost` helper — token multipliers × dataset size × DeepSeek pricing from `core.cost_tracker`). |
| `configure.py` | `configure` | API key management (secure hidden-prompt input, --stdin pipe mode), LM registry (student/teacher), adapter selection (`baml`/`chat`/`json`/`xml`), DSPy runtime settings including `--model-type {chat,auto,responses,text}` (config `dspy.model_type`, default `chat`), cache control (`cache enable --safe-type module:Class` widening `configure_cache`'s restricted deserialization), shell completion (install/uninstall/status/show) |
| `data.py` | `data` | Dataset management — list, import, export, inspect datasets |
| `distill.py` | `distill` | LoRA distillation pipeline — run, list-frameworks, stats, prepare-colab, check (dependency verification) |
| `doctor.py` | `doctor` | System diagnostics — Python/dep/GPU/config/registry/LM/MCP health checks. Also checks dependency services (FalkorDB/Redis :6379, embedding server) with actionable start-up hints (docker compose command) when down; exits 1 on critical service failure. |
| `evaluate.py` | `evaluate` | Evaluation pipeline — run metrics against compiled programs on devsets |
| `export.py` | `export` | Program export/packaging — list, package, info subcommands |
| `generate.py` | `generate` | llms.txt generation — `llms-txt`, `batch`, `explore`, `warmup` (pre-seed analysis cache with known URL patterns) subcommands |
| `gfl.py` | `gfl` | GFL pipeline commands — loop, status, synthesize, meta-optimize, decompose, ab-test, consolidate, spin, lse, gepa, opsd |
| `graph.py` | `graph` | FalkorDB graph management — status, query, migrate, benchmark (p50/p95/p99 latency), cascade (BFS drift→recompile with --dry-run/--no-dry-run), program-lineage (ancestry chain, --visualize renders an ASCII evolution tree with per-step score deltas) |
| `inspect.py` | `inspect` | Program inspection — load and display compiled program structure |
| `lora.py` | `lora` | LoRA adapter management — load, unload, list, chat, test, health, discover, extract, evaluate (bootstrap p-value CI with auto drift registration), train |
| `mcp.py` | `mcp` | MCP server management — start/stop/status of MCP tool servers |
| `memory.py` | `memory` | FalkorDB-native persistent memory — add, search, get-all, delete subcommands |
| `module.py` | `module` | DSPy module CRUD — new, list, show, call, delete. `module new --type` supports Predict/ChainOfThought/ReAct/MultiChainComparison/Tool/Generate/Flex; legacy `ReActV2`/`react-v2`/`react_v2` strings resolve to `ReAct` via `_MODULE_TYPE_ALIASES` |
| `pipeline.py` | `pipeline` | Multi-module pipeline composition — compose, list, run subcommands |
| `run.py` | `run` | Hot-swap inference runner — run compiled programs with live module reload. Subcommands include `react` (structured `dspy.ReAct`: `termination_reason` + `History`, no trajectory dict), `react-v2` (legacy alias), `flex` (`dspy.Flex` with optional tools, `--interpreter-pool-size`, `--max-predictor-calls` — which counts nested/candidate LM calls in 3.5), `rlm` (`--interpreter-pool-size`, `--resume-history <path>` replaying a saved `dspy.History`), `parallel` (`--timeout`, `--straggler-limit`), and the legacy-named `pot`/`code-act` (upstream `CodeAct`/`ProgramOfThought` were **removed** in 3.5 with no alias — both execute on `dspy.RLM`). Result rendering uses `_result_fields()` because `Prediction._output_field_names` no longer exists. `run rlm` persists `result.history` after each successful run as `data_dir()/rlm_history_<YYYYmmdd_HHMMSS>.json` (skipped when empty) and replays it via `--resume-history`; `run rlm`/`run flex` share `_interpreter_factory(pool_size)` for opt-in `dspy.InterpreterPool` prewarming |
| `self.py` | `self` | Auto-evolve commands — self-optimization, router management, drift→recompile cascade (auto-fix), UCB exploration status/reset, watch daemon (--interval --alert-url, --auto-fix for autonomous recompile+hot-swap), distill auto-evaluate with rollback |
| `server.py` | `server` | Hot-swap server — start/stop/status the FastAPI inference server |
| `signature.py` | `signature` | DSPy signature CRUD — new, list, show, edit, delete |
| `skills.py` | `skills` | Skills system — list, install, create, export, ecosystem search (find/discover/categories) |
| `tool.py` | `tool` | DSPy tool registration — register, list, show, inspect, from-mcp, history |

## Work Guidance

- Keep Click command functions thin — delegate logic to `core.*` or subpackage modules.
- Use `click.Choice` for enumerated options, `click.Path` for file/dir arguments.
- Error messages should start with lowercase, no trailing period, prefixed with two spaces.
- Use `click.ClickException` for user-facing errors, not `sys.exit` or raw exceptions.
- New command groups get a new file; do not bloat existing files.
- Register new command groups in `main.py` `SelfOptimizingCLI` under the appropriate category.

### DSPy-native generator pattern (signature.py, module.py)

Both `signature.py` and `module.py` use the same generator architecture:

- **100% DSPy-native**: `_SignatureGeneratorDSPy` and `_ModuleGeneratorDSPy` are inline `dspy.ChainOfThought` modules that produce Python code via the LLM. There is no string-formatting fallback — the LLM **is** the generator.
- **The DSPy generator classes are compilable**: Both use `dspy.ChainOfThought` with descriptive signatures. They can be extracted, compiled with optimizers, and improved over time (e.g. `dspytools compile` on the command module).
- **No import from `modules/`**: Generators are defined inline in the command file, avoiding the `.gitignore`d `modules/` directory.
- **Reserved field names**: `instructions` and `output_fields` are reserved attributes in `dspy.Signature` — generator signatures use `task_instructions` instead.
- **Field parsing**: Bracket-aware field splitting (`list[str]`, `dict[str, Any]`) with type aliasing (`string → str`, etc.) and `=` description extraction. Used for parameter inference, not code generation.

### Bug-fix contracts (established during local-LLM validation)

| File | Contract |
|------|----------|
| `compile.py` | GFL output `result['trend']` may be a string (not float) — use `isinstance()` check before `:.4f` formatting |
| `configure.py` | `key set` defaults to hidden prompt when key argument omitted; `--stdin` flag for pipe-safe mode. `lm list` shows only Student + Teacher sections (no redundant Inference/Default lines). |
| `data.py` | Initialize `dataset = None` before loading; error+abort on nonexistent file path. `--rename old=new` supports field remapping. HuggingFace errors caught with descriptive message; bare names without org/ prefix rejected. |
| `evaluate.py` | Find `dspy.Module` subclass by scanning `dir(mod)` for `issubclass` — do not guess PascalCase |
| `gfl.py` | `--trainset`/`--tasks` options support both raw JSON strings and file paths (`Path(exists)` check) |
| `graph.py` | FalkorDB `list_skills()` may return `description: None` — always use `s.get("description") or ""` |
| `signature.py` | Must use `rsplit("->", 1)` not `split("->")` to handle arrows in descriptions. `instructions` and `output_fields` are reserved field names in `dspy.Signature` — avoid as signature fields. |
| `module.py` | `module call` accepts both `--inputs '{"key":"val"}'` (JSON) and `-i key=val` (KEY=VALUE) formats. `module new` defaults to teacher LM for code generation when `--model` not specified — reads `config["lm"]["teacher"]` from config. |
| `self.py` | `--teacher` defaults to `False` — local 7B models cannot handle GEPA's parallelism; `distill` auto-eval uses local `_llama_chat()` (fail-fast, no try/except) |

## Verification

No automated tests exist yet for command files. Manual smoke test:
```
dspytools --help
dspytools <group> --help
dspytools <group> <command> --help
```

## Child DOX Index

No subdirectories — this directory contains only flat `.py` files and this `AGENTS.md`.
