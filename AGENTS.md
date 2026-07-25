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
3. **Holdout never seen by optimizer**: `split_holdout()` before any compile call
4. **No try/except around imports**: all packages are hard dependencies
5. **Student LM (Qwen 7B local) for inference, Teacher (DeepSeek V4) for reflection only**
6. **Ruff 0 errors required before commit**

## Project Map

```
dspytools/
├── commands/  → CLI subcommands (compile, generate, skills, pipeline, export, ...)
├── core/       → Engine (hotswap, LMRegistry, registry, scheduler, MLflow, mojo_bridge, sprt_mojo_bridge, errors)
├── gfl/       → Generative Feedback Loop + paper optimizers
├── generate/  → llms.txt generation + sandbox pool
├── evolve/    → Self-evolving agent system + harness layers
├── graph/     → FalkorDB graph database + RedisVL semantic cache, cache_mojo_bridge, benchmark
├── memory/    → FalkorDB-native persistent memory layer
├── mcp/       → MCP server (65 tools, 9 resources, 3 prompts)
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
| DOX tree | 12 child AGENTS.md in `src/dspytools/*/` |

## Child DOX Index

Root governs the dspytools project — DSPy 3.3.0b1 examples running on Qwen3.5-9B via llama-cpp-server with MCP-git agentic tools. Twelve child AGENTS.md files in `src/dspytools/*/`:

| Directory | Scope |
|-----------|-------|
| `commands/` | 23 CLI commands (22 groups + 1 standalone, 110+ subcommands) |
| `core/` | Engine: hotswap, LMRegistry, registry, scheduler, output, _dspy, _io, mlflow_tracker, loaders, metrics, cost_tracker, drift_monitor, holdout, retry, mojo_bridge, errors, logging_config |
| `mcp/` | Unified MCP server, 65 tool handlers (all merged into BUILTIN_TOOLS), session pool |
| `generate/` | llms.txt generation: RepositoryAnalyzer, SandboxPool, quality scoring |
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
pytest tests/ -q              # 360 smoke tests, 5s (no LLM needed)
ruff check                     # zero errors required
dspytools --help               # explore all 24 commands (0.18s cold start)
dspytools auto-fix             # inspect and fix drift-degraded programs
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
- **vcrpy cassettes**: `tests/cassettes/` for hermetic LLM mocking (set `VCR_RECORD=1` to record)
