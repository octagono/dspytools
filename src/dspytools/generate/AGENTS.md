# DOX — generate

## Purpose

llms.txt generation pipeline — originally extracted from standalone demo scripts and consolidated into the unified dspytools CLI. Produces structured documentation (`llms.txt` files) from repository analysis using DSPy multi-stage pipelines, heuristic quality scoring, ground truth data, and optional MCP git agentic exploration.

CLI surface: `dspytools generate llms-txt | batch | explore`

## Ownership

| File | Owns | Key Exports |
|------|------|-------------|
| `__init__.py` | Public API surface | Re-exports all public symbols from submodules |
| `cache.py` | AST-based dependency caching for RepositoryAnalyzer | `AnalysisCache`, `get_analysis_cache` |
| `module.py` | DSPy signatures + RepositoryAnalyzer module + SandboxPool | `AnalyzeRepository`, `AnalyzeCodeStructure`, `GenerateLLMsTxt`, `RepositoryAnalyzer`, `SandboxPool`, `get_sandbox_pool` |
| `data.py` | Ground truth training/dev data | `build_ground_truth_examples()` |
| `explorer.py` | File tree gathering + MCP git agent | `gather_repository_info()`, `load_mcp_tools_sync()`, `GitRepoExplorer` |

Files are independent — no circular imports. `__init__.py` aggregates; all other files import only from `dspytools.core._dspy` and standard library.

## Local Contracts

### RepositoryAnalyzer pipeline (`module.py`)

- **5-stage pipeline** executed in `forward()`:
  1. `AnalyzeRepository` (ChainOfThought) → project_purpose, key_concepts, architecture_overview
  2. `AnalyzeCodeStructure` (ChainOfThought) → important_directories, entry_points, development_info
  3. `CodeAct` → usage_examples (with ChainOfThought fallback on Deno sandbox errors)
  4. `ProgramOfThought` → structured_summary from concept bullets (with ChainOfThought fallback on Deno errors)
  5. `GenerateLLMsTxt` (ChainOfThought) → final llms_txt_content
- **AST caching**: `RepositoryAnalyzer.forward()` computes a composite SHA-256 key from all inputs and checks `AnalysisCache` before running the pipeline. On cache hit, returns a `dspy.Prediction` directly. After pipeline completion, the result is cached for future runs. Cache directory: `~/.cache/dspytools/analysis/`. Cache is keyed by 24-char SHA-256 digest of `repo_url|file_tree|readme_content|package_files`.
- `AnalysisCache.warmup(paths)` — pre-computes real composite cache keys for LOCAL repository paths by scanning file_tree/readme/packages. Used by `generate warmup` CLI to reduce cold-start latency for batch processing of local repos.
- CodeAct and ProgramOfThought use `max_iters=1` and have ChainOfThought fallbacks registered via `try/except Exception` in `forward()`.
- `RepositoryAnalyzer.__init__` suppresses Deno sandbox loggers (`dspy.predict.program_of_thought`, `dspy.predict.code_act`) to `CRITICAL` level — 3B model cannot execute sandboxed code.

### Signatures (`module.py`)

- All signatures use `dspy.InputField` and `dspy.OutputField` with `desc` kwargs — no type annotations on fields by convention.
- Signature manipulation variants (`with_instructions`, `append`, `prepend`, `with_updated_fields`, `insert`, `delete`) are demonstrated on `AnalyzeRepository` and `AnalyzeCodeStructure` for downstream use.
- `AnalyzeRepositoryV2` is the variant actually used in the pipeline (via `RepositoryAnalyzer.__init__`).

### Quality scoring (delegated to `core/metrics.py` SSOT)

- `llms_txt_quality(content: str) → float` — imported from `dspytools.core.metrics`. Returns 0.0–1.0 based on:
  - JSON echo detection → instant 0.0
  - Markdown heading presence (# → +0.15, ≥1 h1 → +0.1, ≥2 h2 → +0.15)
  - Bullet lists (+0.1), bold emphasis (+0.05), code blocks (+0.1)
  - Length sweet spot 300–5000 chars (+0.25); >10000 chars penalized (−0.15)
  - Code block wrapping entire output penalized (−0.2)
  - Score clamped to [0.0, 1.0]
- `llms_txt_metric(example, prediction, trace) → float` wraps `llms_txt_quality(prediction.llms_txt_content)`. Imported from `dspytools.core.metrics`.

### Ground truth data (`data.py`)

- `build_ground_truth_examples() → tuple[list[dspy.Example], list[dspy.Example]]`
- Trainset: 5 hardcoded examples (numpy, pandas, dspy, spacy, torch)
- Devset: 1 hardcoded example (fastapi) — unseen during training
- Each example has `repo_url`, `file_tree`, `readme_content`, `package_files` inputs and `llms_txt_content` output.
- All examples call `.with_inputs("repo_url", "file_tree", "readme_content", "package_files")`.

### Repository exploration (`explorer.py`)

- `gather_repository_info(repo_dir: str | None) → tuple[str, str, str, dspy.History]`:
  - With a valid local path: scans via `Path.rglob("*")`, filters out `.`-prefixed and `__pycache__` paths
  - Without a path: returns simulated DSPy repo file tree
  - Always returns readme_content, package_files, and a `dspy.History` of steps
  - Returns `dspy.Code` and `dspy.Image` primitives on the history to demonstrate DSPy multimodal types
- `load_mcp_tools_sync() → tuple[list, list[dspy.Tool]]`:
  - Reads `.mcp.json` from `Path.cwd()`
  - Creates sessions via `MCPSessionPool` from `dspytools.mcp.loader`
  - Returns empty lists if `.mcp.json` is missing, malformed, or has no servers
  - Never raises — all exceptions caught and return `([], [])`
- `GitRepoExplorer`:
  - ReAct agent with `max_iters=12` using MCP git tools
  - Asks 5 exploration questions but only sends first 3 to the agent (practical limit)
  - 3-stage pipeline: explorer (ReAct) → summarizer (ChainOfThought) → generator (ChainOfThought)
  - `forward(repo_path: str) → dspy.Prediction` with fields: `llms_txt_content`, `purpose`, `key_concepts`, `file_tree`, `entry_points`, `exploration_summary`

### CLI integration (`commands/generate.py`)

- 4 subcommands on the `generate` Click group:
  - `llms-txt` — one-shot generation via RepositoryAnalyzer with `--local`/`--remote`, `--baml`, `--label` options
  - `batch` — evaluate RepositoryAnalyzer on devset using `dspy.Evaluate` with `llms_txt_metric`
  - `explore` — deep exploration via GitRepoExplorer + MCP tools; requires valid `.mcp.json`
  - `warmup` — pre-seed analysis cache with local repository paths (from args or `--file`)
- All commands call `setup_dspy()` first, use `from dspytools.core._dspy import dspy`, and register runs via `create_run_dir`/`save_program`/`register_run`.

## Work Guidance

- Keep signatures declarative — prefer `desc` kwargs over type annotations on DSPy fields.
- When adding a new signature variant, use the named manipulation method (`with_instructions`, `append`, etc.) — do not subclass.
- SandboxPool (`SandboxPool` class + `get_sandbox_pool()` singleton) in `module.py` maintains warm `python3` workers. Workers accept code via `sys.stdin.readline()`, exec it, and signal completion with `__SANDBOX_DONE__`. Falls back to one-shot `subprocess.run()` on pool exhaustion. `RepositoryAnalyzer.__init__` creates `self.sandbox_pool = get_sandbox_pool()` for CodeAct/PoT sandboxing.
- `SandboxPool.__init__` accepts `max_reuse: int = 50` and `max_output_size: int = 1_000_000`. Workers exceeding `max_reuse` uses are terminated and replaced with fresh processes on the next `_acquire` call, preventing memory accumulation from untrusted repo code. `max_output_size` prevents runaway output from consuming memory — exceeded output kills the worker and returns an error. `_reuse_count` tracks usage per worker index. `_total_recycled` tracks cumulative recycle events. `stats` property exposes `reuse_counts`, `max_reuse`, `recycled`, and `total_recycled` counters.
- CodeAct and ProgramOfThought stages MUST include ChainOfThought fallbacks. The 3B model (Qwen3.5-9B) cannot execute Deno sandbox code, and the fallback catches `Exception` broadly to handle any runtime sandbox error.
- Ground truth examples should be real `dspy.Example` objects with `.with_inputs()` — do not use raw dicts.
- `gather_repository_info()` is the single source of truth for file tree retrieval — CLI commands must not duplicate this logic.
- MCP tool loading (`load_mcp_tools_sync`) must be defensive (never raise) because `.mcp.json` may be absent in CI or fresh clones.
- Quality scoring heuristics are intentionally simple and rule-based — no learned scoring. Update thresholds only when empirical evaluation shows systematic bias.
- When modifying pipeline stages, ensure `RepositoryAnalyzer.forward()` output dict stays compatible with `dspy.Prediction` field names.

## Verification

- **Quality scorer unit test pattern**: Call `llms_txt_quality()` with known-good markdown and known-bad (JSON echo, empty, rambling) strings; assert return in expected ranges.
- **Ground truth integrity**: `build_ground_truth_examples()` should return exactly 5 trainset + 1 devset examples, each with all 5 fields set and `.with_inputs()` called with exactly `("repo_url", "file_tree", "readme_content", "package_files")`.
- **Pipeline smoke test**: Instantiate `RepositoryAnalyzer`, call with simulated inputs, assert `llms_txt_content` is a non-empty string.
- **MCP load defensive test**: `load_mcp_tools_sync()` with no `.mcp.json` present returns `([], [])` without raising.
- **File tree fallback test**: `gather_repository_info()` with no args returns the simulated tree (contains `"dspy/__init__.py"`).

## Child DOX Index

No subdirectories — this directory contains only flat `.py` files and this `AGENTS.md`.
