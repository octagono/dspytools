# DOX — evolve

- DOX is a highly performant AGENTS.md hierarchy installed here
- This child doc is owned by the root AGENTS.md at `/home/octagono/dev/dspytools/AGENTS.md`
- This doc does not weaken DOX

## Purpose

The `evolve/` directory implements the self-evolving agent system — continuous optimization that monitors quality, triggers re-compilation, and routes queries through a ReAct agent with CLI introspection tools. Covers the `self` CLI command group and the underlying SelfEvolve engine.

Eight source files across 2 directories:

### Top-level files

| File | Role |
|------|------|
| `__init__.py` | Public API — `SelfEvolve` facade (auto_optimize, ask, status), `get_router()` singleton |
| `router.py` | `RouterAgent` — ReAct agent with 6 built-in tools for program management and quality decisions |
| `metrics.py` | Quality metrics — `auto_metric` (heuristic 0.0–1.0), `gepa_metric`, `simple_metric`. All delegating to `core/metrics.py` SSOT. |
| `self_evolve.py` | Continuous self-evolution engine — `MorphologyTracker`, `KnowledgeTransfer`, `UCBExplorer`, `SkillGraph`, `SelfEvolveEngine` |

### `layers/` — Harness-so layers

| File | Role |
|------|------|
| `__init__.py` | Public API — re-exports `Action`, `ActionLayer`, `ContractLayer`, `ContractResult`, `ContractViolation`, `TrajectoryLayer` |
| `action.py` | H1 Action Layer — `Action` dataclass (invoke, bind, with_tools), `ActionLayer` manager (register, from_program, from_skill, from_compiled_run) |
| `contract.py` | H2 Contract Layer — `ContractLayer` (validate_inputs, validate_outputs, infer_schema), `ContractResult`, `ContractViolation` |
| `trajectory.py` | H3 Trajectory Layer — `TrajectoryLayer` (record, replay, diff, search, stats) with SQLite-backed execution trace storage |

## Ownership

This doc owns all files in `src/dspytools/evolve/` including the `layers/` subdirectory. No child DOX docs exist for `layers/` — its contracts are documented here.

`SelfEvolve` (`__init__.py`) is the primary interface consumed by `commands/self.py` and the GFL pipeline.

## Local Contracts

### `__init__.py` — SelfEvolve facade

- `SelfEvolve(quality_threshold=0.5)` — default threshold for triggering re-optimization.
- `auto_optimize() → dict` — monitors quality via `RouterAgent.evolve()`. Returns `{"should_recompile": bool, "action": str, "average_score": float, "message": str}`. When `should_recompile` is `True`, sets `action` to `"recompile_needed"`.
- `ask(question: str) → dict` — routes query through `RouterAgent.ask()`. Returns `{"answer": str, "score": float, "active_program": str, "needs_optimization": bool}`.
- `status → dict` — delegates to `RouterAgent.status`.
- `get_router() → RouterAgent` — module-level singleton. `_router` is `None` until first call.

### `router.py` — RouterAgent

- `RouterAgent(max_iters=10)` — wraps `dspy.ReAct("question -> answer", ...)` with 6 tools (text-based parsing for Qwen 7B compatibility).
- **6 built-in tools** (all `dspy.Tool` instances):
  - `list_programs()` → JSON list of loaded compiled programs
  - `swap_program(program_id)` → JSON `{status, active, previous}` or `{error}`
  - `infer(**inputs)` → JSON `{status, result}` or `{error}`
  - `list_optimizers()` → JSON array of optimizer names
  - `evaluate(content)` → JSON `{score}` via `auto_metric`
  - `compile_decision(program_id, score, threshold=0.5)` → JSON with `{needs_compile, action}`
- `ask(question) → dict` — runs agent, auto-evaluates quality, appends to `_quality_history`. `needs_optimization` is `True` when `score < 0.4`.
- `evolve() → dict` — computes average of last 10 quality samples. `should_recompile` is `True` when `avg_score < 0.5`.
- `status → dict` — returns `{programs_loaded, active_program, quality_samples, average_score}`.
- `_quality_history: list[dict]` — in-memory only (not persisted).
- `HotSwapManager` instance is created internally (`self.mgr`), used by tools. `load_all()` called in `__init__`.

### `metrics.py` — Quality metrics (re-exports from `core/metrics.py` SSOT)

- `auto_metric(content, target_format="markdown") → float (0.0–1.0)` — imported from `dspytools.core.metrics`. Heuristic content quality scoring.
- `gepa_metric(gold, pred, trace=None, ...) → float` — imported from `dspytools.core.metrics`. Extracts `pred.answer` or `pred.output`, delegates to `auto_metric`. Compatible with GEPA optimizer's progress bar interface.
- `simple_metric(example, prediction, trace=None) → float` — imported from `dspytools.core.metrics`. Same extraction logic. Compatible with BootstrapFewShot, MIPROv2, etc.

### `self_evolve.py` — Continuous self-evolution engine

Four learning components wired together by `SelfEvolveEngine`:

#### MorphologyTracker
- Persists to `settings.morphology_path()` (default: `~/.config/dspytools/morphology.json`). SSOT for path: `dspytools.config.settings.morphology_path()`.
- `_load()` is fail-fast — corrupted state raises immediately instead of silently resetting.
- `record(task_profile, pattern_type, success)` — updates `(count, success_count)` tuple per (profile, pattern).
- `best_pattern(task_profile) → str | None` — returns best pattern where `count >= 3` and success rate is highest.
- `profile_task(description, field_count, data_size) → str` — creates a `"{domain}_{size}_{field_count}f_{words}w"` profile string. Domain detection keywords: repo/documentation → documentation, classify → classification, generate/write → generation, else general. Size thresholds: <10 → sparse, <50 → moderate, else dense.

#### KnowledgeTransfer
- `find_similar_tasks(task_profile, max_results=3) → list[str]` — matches by domain prefix (first `_` segment).
- `transfer_patterns(target_profile) → dict[str, float]` — transfers best patterns from similar tasks. Same-domain weight = 1.0, cross-domain weight = 0.5.

#### UCBExplorer
- `trials: dict[str, tuple[int, float]]` — optimizer → `(trial_count, avg_score)`.
- 9 optimizers: `bootstrap_few_shot`, `mipro`, `gepa`, `copro`, `simba`, `labeled_few_shot`, `knn`, `better_together`, `grpo`.
- `record(optimizer, score, cost=0.0)` — incremental running average update. Optional `cost` tracks avg token cost per trial for cost-aware selection.
- `select(c=2.0, cost_weight=0.0) → str` — UCB1 formula: `avg + c * sqrt(log(total) / count)`. Untried optimizers get `ucb = inf` to prioritize exploration. `cost_weight > 0` penalizes expensive optimizers (cost-aware UCB).
- `costs: dict[str, float]` — optimizer → avg_cost_per_trial. Persists alongside trials in ucb_explorer.json. Backward compatible (old format auto-migrates).
- `exploitation_score → float` — ratio of tried optimizers / total optimizers.

#### SkillGraph
- SSOT: delegates to `FalkorDBSkillGraph` when FalkorDB is available, JSON file as fallback only.
- Persists to `~/.config/dspytools/skill_graph.json` (fallback only — FalkorDB is primary).
- `_get_falkordb()` — lazy-init `FalkorDBSkillGraph` backend. Returns `None` if FalkorDB unreachable (JSON fallback). Import is fail-fast; only runtime instantiation is wrapped.
- `add_dependency(skill, depends_on)` — dual-write: FalkorDB (SSOT) + JSON mirror.
- `get_dependents(skill) → list[str]` — queries FalkorDB first, JSON fallback.
- `transitive_dependents(skill) → list[str]` — queries FalkorDB first, JSON BFS fallback.
- `on_improvement(skill) → list[str]` — delegates to `transitive_dependents`.

#### SelfEvolveEngine
- `on_compile(task_profile, optimizer, score, success=True) → dict` — updates all four components. Uses per-component dirty flags so `_flush_dirty()` only persists components that actually changed (morphology, UCB, graph, scores each tracked independently). Calls `DriftMonitor.request_recompile()` for downstream skills in the skill graph (cascade recompile). Returns `{morphology, transferred, ucb_next, exploitation}`. **SSOT: stores optimization lessons in `MemoryManager` (FalkorDB-native) and records to `FalkorDBSkillGraph`.**
- `suggest_optimizer(task_profile) → str` — preference order: **Memory search** (past lessons from `MemoryManager`) > morphology best pattern > transfer best > UCB select.
- **LSE tree-guided evolution** (`evolve_context_lse`): integrated into the engine. Uses `LSESelfEvolveModule` (compilable f_ψ policy) + UCB tree search to evolve instruction contexts from performance history. Evaluates on a holdout set per paper Eq. 4.
- **Trace2Skill consolidation** (`consolidate_skills`): integrated into the engine. Wires the full 3-stage pipeline (Rollout → Analyze → Consolidate) with compilable DSPy modules. Results feed into the skill graph for transitive improvement chains.
- **Closed self-evolve cycle** (`auto_evolve_cycle`): runs the full loop — suggest optimizer → LSE-evolve context → compile → Trace2Skill consolidate → LoRA distill. The system continuously improves its own programs, contexts, and skill library.
- **Drift→recompile cascade**: `on_compile()` triggers `DriftMonitor.request_recompile()` for each downstream skill via `SkillGraph.transitive_dependents()`. CLI: `dspytools self auto-fix --no-dry-run --auto-fix`. MCP tool: `drift_auto_fix`.
- `suggest_optimizer(task_profile) → str` — preference order: morphology best pattern > transfer best > UCB select.
- `on_skill_improvement(skill) → list[str]` — delegates to `SkillGraph.on_improvement`.
- `add_skill_dependency(skill, depends_on)` — delegates to `SkillGraph.add_dependency`.
- `validate_and_deploy(candidate_program, program_id, holdout_set[, alpha=0.05, beta=0.2, max_evaluations=50]) → dict` — Gödel Agent pattern (arXiv 2410.04444) using Sequential Probability Ratio Test (Wald, 1945). Evaluates one example at a time, computes log-likelihood ratio between H₀ (p ≤ 0.50) and H₁ (p ≥ 0.65), accepts/rejects early when confidence thresholds are met. `alpha` controls Type I error (false deploy), `beta` controls Type II error (miss improvement), `max_evaluations` limits total cost. Returns `{accepted, candidate_score, p_value, n_evaluated, early_stop, reason, statistical_method}`.
- `self_validate(program_id, holdout[, alpha=0.05, beta=0.2]) → dict` — convenience wrapper: calls `suggest_optimizer()` then `validate_and_deploy()` with SPRT parameters.
- `archive_search(task_description, top_k=3) → list[dict]` — Meta Agent Search pattern (arXiv 2408.08435). Keyword-scans compiled registry for similar past programs. Returns top-k runs sorted by relevance.
- **Convergence guardrails** — Goodhart/repetition/detection to prevent metric cheating:
  - `detect_metric_cheating(outputs, repetition_threshold=0.85, min_samples=3) → dict` — static method. Checks if the most common output exceeds `repetition_threshold` fraction of all outputs (mode ratio). Returns `{cheating, max_repetition_ratio, n_samples, trigger}`. Needs at least `min_samples` before flagging.
  - `detect_output_degradation(scores, window=5, variance_threshold=0.01) → dict` — static method. Checks for score stagnation (variance < threshold) or oscillation (alternating signs with amplitude > 0.2 AND no net improvement). Returns `{degraded, variance, oscillating, mean, triggers}`.
  - `check_convergence(predictions) → dict` — unified check combining repetition + degradation. Caches last 50 predictions and 100 scores. Returns `{repetition_warning, degradation_warning, safe, max_repetition_ratio, score_variance}`.
  - `record_score(score)` — records a score for convergence tracking (capped at 100 entries).
  - `on_compile()` now calls `record_score()` before updating morphology/UCB/transfer.

#### Persistent state files
| File SSOT | Settings Accessor | Owner |
|-----------|-------------------|-------|
| `~/.config/dspytools/morphology.json` | `settings.morphology_path()` | MorphologyTracker |
| `~/.config/dspytools/skill_graph.json` | `settings.skill_graph_path()` | SkillGraph |
| `~/.config/dspytools/ucb_explorer.json` | `settings.ucb_explorer_path()` | UCBExplorer |
| `~/.config/dspytools/evolve_scores.json` | `settings.evolve_scores_path()` | SelfEvolveEngine |

All use `dspytools.core._io.write_json()` for atomic saves and `read_json()` for loading.
Load methods are **fail-fast** — corrupted state files raise immediately instead of silently resetting.
`SelfEvolveEngine._flush_dirty()` tracks per-component dirty flags and only persists components whose `record()` was called since the last flush — avoids redundant disk I/O when only one tracker changed.

### `layers/__init__.py` — Harness-so public API

- Re-exports: `Action`, `ActionLayer`, `ContractLayer`, `ContractResult`, `ContractViolation`, `TrajectoryLayer`.
- No additional logic — all contracts live in the respective module files.

### `layers/action.py` — H1 Action Layer

- `Action` is a dataclass with fields: `name`, `description`, `program`, `tools`, `input_schema`, `output_schema`, `tags`.
- `Action.invoke(**inputs) → dict` — executes `self.program(**inputs)`, returns structured output. Uses `_output_field_names` if available (DSPy Prediction), falls back to `toDict()`, then `{"output": str(result)}`.
- `Action.bind(program) → Action` — binds a DSPy program, returns self for chaining.
- `Action.with_tools(*tools) → Action` — appends tools, returns self for chaining.
- `Action.to_dict() → dict` — serializes metadata (no program/tools content).
- `ActionLayer` — manager collecting named `Action` instances.
- `ActionLayer.register(action)`, `get(name)`, `list_actions()`, `invoke(name, **inputs)`.
- `ActionLayer.from_program(name, program, ...) → Action` — creates and registers an action from any DSPy program/module.
- `ActionLayer.from_skill(skill_name) → Action | None` — creates action from a compiled skill (loads `program.json` via `SkillManager`). Returns `None` if skill absent or uncompiled.
- `ActionLayer.from_compiled_run(run_id) → Action | None` — creates action from `_load_program_from_run`. Returns `None` on missing run.

### `layers/contract.py` — H2 Contract Layer

- `ContractLayer.TYPES` — maps type name strings (`"str"`, `"int"`, etc.) to Python types.
- `validate_inputs(inputs, schema) → ContractResult` — checks required fields present and types match. `schema` keys: `inputs` (comma-separated field names), `types` (field → type name dict). Empty schema → always valid.
- `validate_outputs(outputs, schema) → ContractResult` — checks expected keys present and non-empty. `schema` keys: `outputs` (comma-separated field names). Empty schema → always valid.
- `infer_schema(signature_str) → dict` — parses `"inputs -> outputs"` DSPy signature string, extracts field names and type annotations, returns `{inputs, outputs, types}` dict. Fields without type annotation default to `"str"`.
- `ContractViolation(field, expected, actual, message)` — dataclass for a single violation.
- `ContractResult(valid, violations)` — dataclass with `valid: bool` and `violations: list[ContractViolation]`. Uses `field(default_factory=list)` for the violations default (Python 3.12 prohibits mutable defaults in dataclasses).

### `layers/trajectory.py` — H3 Trajectory Layer

- `TrajectoryLayer` is a class-level singleton via class methods and module-level `_lock` threading lock.
- **SQLite-backed**: Database at `~/.config/dspytools/trajectories.db`. Table `trajectories` with columns: `id`, `run_id`, `action_name`, `timestamp`, `inputs`, `outputs`, `score`, `metadata`. Indexed on `run_id` and `action_name`.
- `record(run_id, action_name, inputs, outputs, score=0.0, metadata=None) → int` — inserts a trace row, returns row ID. Inputs/outputs/metadata serialized via `json.dumps(..., default=str)`.
- `replay(run_id) → list[dict]` — returns all traces for a run, ordered by timestamp. Deserializes `inputs`, `outputs` JSON fields (non-destructively on parse error).
- `diff(run_a, run_b) → dict` — compares two runs: step count, avg score, winner, score delta. Winner is run with higher average score (ties go to `a`).
- `search(action_name=None, min_score=0.0, limit=20) → list[dict]` — filters by action name and minimum score, returns most recent first.
- `stats(action_name=None) → dict` — returns `{total_traces, average_score, recent}` from aggregate SQL queries.
- All public methods are thread-safe via `_lock`.

## Work Guidance

- `SelfEvolve` in `__init__.py` is the primary user-facing API. `SelfEvolveEngine` in `self_evolve.py` is the internal continuous learning engine. They serve different purposes — keep them separate.
- When adding tools to `RouterAgent`, add them to `_build_router_tools()` and return a `dspy.Tool` with a `name` and `desc`.
- Router tools return JSON strings — callers must `json.loads()` the result. Error states return `{"error": str}` format.
- `auto_metric` is intentionally heuristic and rule-based. Do not add learned scoring here — that belongs in the GFL pipeline.
- Quality thresholds: `score < 0.4` in `ask()` triggers `needs_optimization` flag; `avg_score < 0.5` in `evolve()` triggers `should_recompile`. These are separate thresholds for different purposes.
- `_quality_history` is in-memory only — lost on process restart. For persistence, route through `core/registry.py` or GFL's `QualityMonitor`.
- When adding new persistent state in `self_evolve.py`, follow the pattern: `DATA_PATH` class attribute, `_load()`/`save()` methods, silent error handling on load.
- UCB exploration constant `c=2.0` encourages exploration. Decrease for more exploitation-heavy behavior. Tune based on empirical results.
- Morphology tracker requires `count >= 3` min evidence before recommending a pattern from `best_pattern()`.

## Verification

- **SelfEvolve smoke test**: Instantiate `SelfEvolve()`, call `ask("list programs")`, assert `"answer"` is non-empty string, `"score"` is 0.0–1.0.
- **auto_optimize contract**: Call `auto_optimize()`, assert return dict has all four keys (`should_recompile`, `action`, `average_score`, `message`).
- **Router tools smoke test**: `list_programs()` returns valid JSON array; `evaluate("# Hello")` returns `score > 0.0`; `compile_decision("test", 0.3)` returns `needs_compile: true`.
- **auto_metric edge cases**: Empty string → 0.0; JSON echo (`{"repo_url": "..."}`) → 0.0; well-structured markdown → >0.5; short content <50 chars → 0.0.
- **UCB select test**: With no trials, `select()` returns first untried optimizer. After recording one trial, untried optimizers still get `ucb = inf` priority.
- **MorphologyTracker persistence**: Record a pattern, reload from disk, assert `best_pattern()` returns it.
- **SkillGraph transitive test**: `A → B → C`, call `on_improvement(A)`, assert `["B", "C"]` is returned in BFS order.
- **SelfEvolveEngine suggestion**: After recording a successful compile, `suggest_optimizer()` returns the recorded optimizer name.
- **Action.invoke contract**: Create an `Action` with a mock program returning a dataclass with `_output_field_names`, assert `invoke()` returns structured dict with those field names.
- **ActionLayer.from_skill**: With a valid compiled skill, `from_skill()` returns non-None Action; with uncompiled skill, returns `None`.
- **ContractLayer.validate_inputs**: Missing required field → `ContractResult(valid=False)`. Type mismatch → violation. Empty schema → `valid=True`.
- **ContractLayer.infer_schema**: `"question: str -> answer: str"` returns `{inputs: "question", outputs: "answer", types: {question: "str", answer: "str"}}`.
- **TrajectoryLayer round-trip**: `record()` then `replay()` returns a list with matching inputs/outputs.
- **TrajectoryLayer.diff**: Two runs with different scores — assert `winner` is the higher-scoring run.
- **TrajectoryLayer.search**: Record multiple traces with different scores, `search(min_score=0.5)` returns only traces above threshold.

## Child DOX Index

- `layers/` — Harness-so layers (Action, Contract, Trajectory). No AGENTS.md file — all contracts are documented in this parent doc.
