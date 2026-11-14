# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read root AGENTS.md
2. Identify files/folders you expect to touch
3. Walk from repo root to each target path, reading every AGENTS.md along the route
4. If a parent lists a child whose scope contains the path, read that child and continue
5. Use the nearest AGENTS.md as local contract, parent docs for repo-wide rules
6. If docs conflict, the closer doc controls local details; no child may weaken DOX

Do not rely on memory — re-read the applicable DOX chain in each session.

## Update After Editing

Every meaningful change requires a DOX pass. Update the closest owning AGENTS.md when a change affects purpose, scope, ownership, responsibilities, durable structure, contracts, workflows, rules, inputs/outputs, permissions, constraints, artifacts, or user preferences. Update parent docs when parent-level structure or child index changes. Update child docs when parent changes alter local rules. Small edits that do not change behavior may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide rules, global preferences, top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent documents direct children scope; closer docs are more specific and practical

## Child Doc Shape

Create a child AGENTS.md at a durable boundary with distinct purpose, rules, or quality standards. Default section order: Purpose, Ownership, Local Contracts, Work Guidance, Verification, Child DOX Index. Work Guidance and Verification reflect current project standards; leave empty if none exist yet.

## Style

Keep docs concise and current. Document stable contracts, not diary entries. Put broad rules in parent docs, concrete details in children. Prefer direct bullets. Delete stale notes instead of explaining history.

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

When the user requests a durable behavior change, record it here or in the relevant child AGENTS.md

## Golden Rules

1. **Lazy import**: always `from dspytools.core._dspy import dspy` — never `import dspy` directly
2. **Teacher LM only for optimization**: `LMRegistry.get_teacher()` only in GEPA/distill/finetune paths. Never for inference.
3. **Holdout never seen by optimizer**: `split_holdout()` before any compile call — `compile <optimizer>` splits `HoldoutGate` before constructing the optimizer and carves GEPA/MIPROv2 valsets out of the train split, never the holdout
4. **No try/except around imports**: all packages are hard dependencies
5. **Student LM (Qwen 7B local) for inference, Teacher (DeepSeek V4) for reflection only**
6. **Ruff 0 errors required before commit**
7. **`model_type="chat"` is the default**: llama.cpp serves only `/v1/chat/completions`; DSPy 3.5's `"auto"` routes to `/v1/responses`. Override with config `dspy.model_type`.
8. **Thinking is controlled by `reasoning_effort`, not `chat_template_kwargs`**: 3.5 rejects the latter on the canonical Request path. `enable_thinking: false` → `reasoning_effort="none"`; `true`/absent → omit the dial.
9. **No live LM in tests**: `dspy.utils.DummyLM` or a stub `complete(Request) -> Response` engine. vcrpy cannot intercept lm15's bundled raw-socket transports.

## DSPy 3.5 Contracts

- **Pin**: `dspy-ai==3.5.0` + `dspy[openai,numpy]>=3.5.0`. LiteLLM/OpenAI SDK are optional upstream; the `lm15` engine is bundled and is the only execution path this project uses.
- **`dspy.ReAct` is the structured agent** (history + typed `submit` tool). `dspy.ReActV2` survives as a deprecated subclass until 3.6; CLI `react-v2` / `ReActV2` strings alias to `dspy.ReAct`. Results carry `termination_reason` (`submit`/`forced_submit`) and `history` (`dspy.History.messages`); a failed forced submit raises.
- **Removed upstream**: `CodeAct`, `ProgramOfThought`, `dspy.experimental`, `TypedPredictor`, `Image.from_file`/`from_PIL`, `lm(messages=...)`, `Tool.format_as_litellm_function_call`. `dspytools run pot|code-act` map to `dspy.RLM`.
- **Adapter**: upstream default is `XMLAdapter(use_native_function_calling=True)`; `dspytools` always selects one explicitly, so `settings.adapter` is never `None`. The code default is `baml`; `dspy.adapter` accepts `baml`/`chat`/`json`/`xml` and the configured value wins (`dspytools configure dspy show`).
- **Reasoning fields**: `ChainOfThought` rationale is `dspy.Reasoning`/`None`, not `str`; it is str-like but fails `isinstance(x, str)` and `json.dumps`. Opt into text with `rationale_field_type=str`.
- **Metrics**: `core.metrics.exact_match_metric` normalizes both answers (strip/casefold/whitespace/surrounding punctuation) before comparing; `gepa_metric` returns `dspy.Prediction(score, feedback)` (use `gepa_metric_float` where the metric is summed, e.g. `dspy.BetterTogether`).
- **GRPO** (`dspy.teleprompt.grpo.GRPO`, not re-exported by `dspy`) is finetune-only and has no `lora`/`beta`/`max_steps` kwargs in 3.5 — `compile grpo` maps `--max-steps` to `num_train_steps` and fails fast on a local llama.cpp endpoint.
- **GEPA/Flex/RLM are graduated** (no `dspy.experimental`). GEPA kwargs are validated against the installed constructor — accept, never hardcode a version list.
- **Sampling**: `temperature` is not a DSPy setting (`DEFAULT_CONFIG` omits it; `Predict` reads `config`/`lm.kwargs`). Vary it with `core.setup.sampling_lm(lm, t)` → `lm.copy(temperature=t)`; `dspy.context(temperature=...)` is silently ignored.
- **Caches**: restricted deserialization is the default (canonical v2 keys invalidate old entries); `configure_cache(safe_types=...)` is the only widening knob.
- **Usage accounting**: `setup_dspy()` turns DSPy's `track_usage` on by default (a persisted `dspy.track_usage` still wins). Read real numbers from `Prediction.get_lm_usage()` / `lm.history[-1]["usage"|"cost"]` via `core.cost_tracker`; cached calls report `None` and are skipped, never counted as free work.
- **Retries**: `core.retry` retries upstream's own classifier (`dspy.utils.exceptions._RETRYABLE_LM_ERRORS`: rate-limit, timeout, server, transport, lock-timeout) plus stdlib transients, honours `retry_after`, and refuses context-window/auth/bad-request/config errors even if a caller widens the tuple. `LMUnexpectedError` is deliberately not retried — upstream does not classify unclassified failures as retryable.
- **Not adopted**: rich tool results (images/files from tools). The `chat`/chat-completions presets declare `tool_result_media="reject"` in lm15, so a media part in a tool result aborts the agent mid-turn. Revisit only with a media-capable route and `mcp/loader.py` emitting `dspy.Image` parts behind an explicit flag.

## Project Map

```
dspytools/
├── commands/  → CLI subcommands (compile, generate, skills, pipeline, export, memory, graph, ...)
├── core/       → Engine (hotswap, LMRegistry, registry, scheduler, MLflow, holdout, retry, drift_monitor, mojo_bridge, errors)
├── gfl/       → Generative Feedback Loop + paper optimizers
├── generate/  → llms.txt generation + sandbox pool
├── generators/ → DSPy-native code generators (SignatureGeneratorDSPy, ModuleGeneratorDSPy)
├── evolve/    → Self-evolving agent system + harness layers
├── graph/     → FalkorDB graph database + RedisVL semantic cache, cache_mojo_bridge, benchmark
├── memory/    → FalkorDB-native persistent memory layer
├── mcp/       → MCP server (65 tools, 8 resources, 3 prompts)
├── skills/    → Skills library (BM25 + embedding, bm25_mojo_bridge)
├── help/      → Self-optimizing --help
├── api/       → FastAPI server
├── config/    → Config management + .env
├── mojo_modules/ → Mojo hybrid acceleration — auto-compiled via mojo.importer at import time; SIMD acceleration needs full Mojo SDK (vector_utils, sprt, bm25)
└── cli/       → Rich output utilities
```

## Where to Look

| Need | Go to |
|------|-------|
| Architecture | `docs/architecture.md` |
| GFL Pipeline | `docs/gfl-pipeline.md` |
| Self-Evolve | `docs/self-evolve.md` |
| MCP Server | `docs/mcp-server.md` |
| LoRA Integration | `docs/lora-integration.md` |
| All commands | `dspytools --help` or `docs/index.md` |
| MLflow setup | `dspytools/core/mlflow_tracker.py` docstrings |
| One-command dev | `docs/dev-local.md` |
| Verify before ship | `scripts/verify.sh` |
| DOX tree | 13 child AGENTS.md in `src/dspytools/*/` |

## Child DOX Index

Root governs the dspytools project — DSPy 3.5.0 examples running on Qwen3.5-9B via llama-cpp-server with MCP-git agentic tools (dspy-ai==3.5.0 pin; Flex, ReAct, RLM, GEPA cost-aware optimization supported). Thirteen child AGENTS.md files in `src/dspytools/*/`:

| Directory | Scope |
|-----------|-------|
| `commands/` | 24 CLI commands (24 groups) |
| `core/` | Engine: hotswap, LMRegistry, registry, scheduler, MLflow, holdout, retry, cost_tracker, drift_monitor, loaders, metrics, output, _dspy, _io, _embedder, dspy_modules, errors, logging_config |
| `mcp/` | Unified MCP server, 65 tool handlers (all merged into BUILTIN_TOOLS), session pool |
| `generate/` | llms.txt generation: RepositoryAnalyzer, SandboxPool, quality scoring |
| `generators/` | DSPy-native code generators: SignatureGeneratorDSPy, ModuleGeneratorDSPy (compilable ChainOfThought modules) |
| `gfl/` | GFL pipeline (4-way, Successive Halving, Speculative Compile), 8 paper optimizers, ChallengerSolver, Trace2Skill |
| `evolve/` | SelfEvolveEngine (SPRT, Meta Agent Search), RouterAgent (ReAct), 3 harness layers |
| `graph/` | FalkorDB: client, skill_graph, cache, migrate (O(1) graph traversal, semantic cache) |
| `memory/` | FalkorDB-native: MemoryManager singleton (entity extraction, dedup, semantic search) |
| `help/` | Self-optimizing help: DSPy program, CLI introspection, auto-compile |
| `api/` | FastAPI server with hot-swap endpoints |
| `config/` | Config management: hot-reload ConfigCache, .env read/write |
| `skills/` | Agent Skills: BM25 + embedding hybrid search, lifecycle management |

## Quick Commands

```bash
pytest tests/ -q              # 490 hermetic tests, ~6s (DummyLM only; 20 Redis/FalkorDB failures when no container is up)
ruff check                     # zero errors required
dspytools --help               # explore all 24 commands
dspytools self auto-fix         # inspect and fix drift-degraded programs
./scripts/ci-local.sh          # full 5-stage CI pipeline on local GPU
./scripts/ci-local.sh --stage 1  # lint + smoke tests only
```

## CI/CD

Five-stage pipeline. See `docs/ci-cd-plan.md` for full architecture.

| Stage | Workflow | Trigger | Runner |
|-------|----------|---------|--------|
| 1. Lint + Smoke | `ci.yml` | PR/push | `ubuntu-latest` |
| 2. Integration | `integration.yml` | push to main | `ubuntu-latest` + FalkorDB container |
| 3. Release | `release.yml` | tag `v*` | self-hosted + `ubuntu-latest` |
| 4. Evaluation | `evaluation.yml` | nightly 02:00 UTC | `[self-hosted, gpu]` |
| 5. Deploy | `deploy.yml` | Stage 4 success | `[self-hosted, gpu]` |

- **Local CI**: `./scripts/ci-local.sh` runs all 5 stages on local GPU, zero cloud cost
- **Self-hosted runner**: local machine polls GitHub for jobs via `~/actions-runner/run.sh`
- **Golden dataset**: `data/golden_eval.jsonl` (50 examples) + `data/golden_baseline.json` (tracked score)
- **Regression gate**: nightly evaluation must score >= 90% of baseline or pipeline halts
- **No LLM mocking middleware**: cassettes were removed — lm15 writes through bundled raw-socket transports, so vcrpy cannot intercept it. Tests use `DummyLM` or a stub `complete(Request) -> Response` engine; `DSPYTOOLS_LIVE_LM=1` opts into the local llama.cpp lane.
