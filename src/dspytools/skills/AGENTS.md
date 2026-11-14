# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Purpose

The skills directory implements the Agent Skills system — a BM25-indexed library of reusable DSPy programs with a full lifecycle: **create → compile → optimize → deploy**. Skills follow the harness-so pattern: each skill is a directory containing `SKILL.md` (frontmatter + docs) and `program.json` (compiled DSPy program). Skills are consumed by agent pipelines via MCP tools and the `skills` CLI commands.

## Ownership

- `SkillManager` — primary lifecycle interface: create, compile, optimize, search, and list skills. Used by `commands/skills.py`, `mcp/tools.py`, and `evolve/layers/action.py`. Exported from `__init__.py`.
- `SkillLoader` — BM25-indexed skill library with directory loading from project-local `skills/`, user-wide `~/.config/dspytools/skills/`, and `~/.agents/skills/`. BM25 is the default search; hybrid semantic search (`search_embeddings`) is opt-in. `search()` delegates the inner BM25 scoring loop to `bm25_mojo_bridge.py` when available.
- `load_skill_program(skill_path)` — rebuilds a skill's saved program in its recorded shape (`metadata.json` → `signature.json` → `"predict"`). Exported from `__init__.py`.
- `bm25_mojo_bridge.py` — Mojo SIMD bridge for BM25 scoring (Phase 3). `score_documents(q_tf_matrix, idf_values, doc_lengths, avg_doc_len) → ndarray`. `HAS_MOJO` flag + pure Python fallback. Imports numpy, so `loader.py` imports it lazily inside the scoring path.
- `Skill` — dataclass representing a single skill: name, description, path, frontmatter, body, and optional program path. Exported from `__init__.py`.
- `ExternalSkill` + `search_external()` / `popular_skills()` / `list_categories()` / `try_skills_sh_api()` — ecosystem discovery over a hardcoded curated list (`_POPULAR_SKILLS`) plus an unauthenticated `https://skills.sh/api/search` fetch. Exported from `discovery.py`.

## Local Contracts

### Skill directory format
- Skills are stored as directories named by skill name. Each directory contains:
  - `SKILL.md` — YAML frontmatter (`name:`, `description:`, `signature:`, all JSON-quoted) followed by body text
  - `program.json` — compiled DSPy program (optional, present after `compile_skill`)
  - `signature.json` — `{inputs: [...], outputs: [...], module_type: ...}` (written during compile)
  - `metadata.json` — `{skill, module_type, optimizer, demos, trainset_size}` (written during compile; the load path reads `module_type`)
- Skills without `SKILL.md` are silently skipped during `load_all()`

### Primary interface contract
- `SkillManager.create_skill(name, description, signature, body="")` → `Skill`
  - Writes `SKILL.md` with frontmatter + body to `{skills_dir}/{name}/`
  - Every frontmatter scalar is JSON-encoded (`json.dumps`), a valid YAML subset — an unquoted description containing `": "` or `" #"` used to produce invalid YAML and abort GEPA's `InstructionProposer` with `ValueError("Invalid YAML skill frontmatter.")`
  - Returns the created `Skill` instance immediately (no compilation, no `program_path`)
- `SkillManager.compile_skill(name, trainset=None, optimizer="labeled_few_shot")` → `{status, skill, optimizer, demos, trainset_size, score, has_program}` or `{error: ...}`
  - Compiles a DSPy program using the skill's signature from frontmatter
  - Supports `"labeled_few_shot"` and `"bootstrap_few_shot"`; the bootstrap metric is normalized exact match (`core.metrics.answer_exact_match`)
  - Demo caps stay at DSPy's defaults — `dspy.LabeledFewShot()` (k=16) and `dspy.BootstrapFewShot`
  - Saves `program.json`, `signature.json` and `metadata.json` (`module_type="predict"`) into the skill directory
  - `optimizer` is the optimizer **actually applied**: `"none"` when no trainset was supplied (the base program is saved unchanged) and `demos` is the real demo count
  - `score` stays `None`: nothing is evaluated here, so no score is fabricated
  - The FalkorDB write is best-effort — it runs after the artifact is on disk, MERGEs a `:Skill {name}` node carrying `program_path` (the label `list_skills` and `hotswap._prefetch_dependents()` MATCH), and a graph outage only logs `skill_graph_record_failed`
- `SkillManager.auto_optimize_skill(name, trainset=None)` → `{status, skill, best, score}` or `{error: ...}`
  - Uses `GFLPipeline` from `dspytools.gfl.pipeline` for full 4-way optimizer comparison
  - Loads the existing program through `load_skill_program()` (shape-aware, version-gated), runs GFL, saves the best result back to `program.json`
  - Requires a loadable compiled program
- `SkillManager.generate_from_program(run_id, skill_name, description)` → `Skill | None`
  - Copies `program.json`, `signature.json` and (when present) `metadata.json` from the compiled run into a new skill directory
  - Returns `None` if the source `program.json` doesn't exist; otherwise the returned `Skill` carries `program_path`
- `SkillManager.get(name)` → `Skill | None` — exact-identity lookup through `SkillLoader.get()` (never a ranked match)
- `SkillManager.list_skills()` → `list[Skill]`
- `SkillManager.search(query, k=5)` → `list[Skill]` — BM25 only
- `SkillManager.search_hybrid(query, k=5)` → `list[Skill]` — embedding + BM25 fused by reciprocal rank fusion, BM25 fallback

### Program load contract
- `load_skill_program(skill_path)` → `dspy.Module | None`
  - Returns `None` when `program.json` is missing or `core.programs.is_current_dspy_version()` says it was saved by another DSPy (stale artifacts are treated as absent; the loader logs `skill_program_dspy_version_mismatch`)
  - Rebuilds the shape from `metadata.module_type` the way `core.hotswap._load_program_from_run` does: `cot`/`chain_of_thought` → `dspy.ChainOfThought`, `react` → `dspy.ReAct`, `flex` → `dspy.Flex`, default → `dspy.Predict`
  - A `CoT`/`ReAct` artifact loaded into a hardcoded `dspy.Predict` raised because the saved state has more predictors than the fresh program

### SkillLoader contracts
- `SkillLoader(skills_dir=None)` searches three paths, first-loaded wins:
  1. The provided `skills_dir` (or `Path("skills")` if None)
  2. `~/.config/dspytools/skills/` (user-wide)
  3. `~/.agents/skills/` (agent-ecosystem skills)
- `SkillLoader.load_all()` → `list[Skill]`
  - Iterates all directories under all three paths, skips hidden dirs
  - Only loads directories containing `SKILL.md`
  - Cache key is a SHA-256 of the sorted `(base, dir_name, SKILL.md mtime_ns, program.json presence/mtime)` tuples — deletions, renames, new skills and recompiles all invalidate it. There is no time-based TTL
  - Reload builds a fresh skill dict + BM25 index and swaps both; state is guarded by one `threading.RLock` (MCP issues concurrent tool calls)
- `SkillLoader.search(query, k=5)` → `list[Skill]`
  - Calls `load_all()` first, then standard BM25 scoring: k1=1.2, b=0.75
  - Tokens extracted via regex `[a-zA-Z0-9_-]+` (lowercased)
  - Returns top-k scoring skills; returns all skills (up to k) if no query tokens
- `SkillLoader.search_embeddings(query, k=5)` → `list[Skill]` — opt-in (`skills search --hybrid`)
  - Uses the shared singleton `core._embedder.get_embedder()` and passes `dimensions=settings.embedder_dimension()`
  - Skill vectors are cached by `(model, dimension, skill name)` and invalidated on reload, so a model or dimension change never reuses stale vectors
  - Fusion is reciprocal rank fusion (`1/(60 + rank)` over embedding rank and BM25 rank), not a cosine plus rank bonus
  - Any embedder failure logs `skill_embeddings_unavailable` and falls back to BM25 — the hybrid path never fails a caller over an optional backend
- `SkillLoader.get(name)` → `Skill | None` — exact key lookup (loads on demand)

### BM25 index contract
- Built on tokenized concatenation of `name + description + body`
- `_tokenize(text)`: extracts `[a-zA-Z0-9_-]+` tokens, lowercased
- `_count_docs_with_token(index, token)`: counts documents containing the token (tf > 0)
- IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1.0)`
- BM25 TF: `(tf * (k1+1)) / (tf + k1*(1-b + b*docLen/avgLen))`
- `_bm25_ranked(index, query_tokens)` owns the scoring; numpy arrives through the lazily imported Mojo bridge

### Frontmatter parsing contract
- `_parse_frontmatter(text)` → `(dict, body)`
- Uses `^---\s*\n(.*?)\n---\s*\n` regex (DOTALL mode)
- Parses `key: value` lines; JSON-quoted scalars are decoded with `json` (`_unquote_scalar`), single quotes are stripped
- Supports array values: `key: [a, b, c]` parsed as `["a", "b", "c"]`
- Returns empty dict + full text if no frontmatter delimiter found

### CLI integration
- `skills create <name> <description> [--signature/-s] [--body]` → `SkillManager.create_skill()`
- `skills compile <name> [--optimizer/-o] [--trainset/-t]` → `SkillManager.compile_skill()`; without `--trainset` it prints that the skill was saved unoptimized
- `skills auto-optimize <name> [--trainset/-t]` → `SkillManager.auto_optimize_skill()`
- `skills list` → `SkillManager.list_skills()`
- `skills search <query> [--k] [--hybrid]` → `SkillManager.search()` / `search_hybrid()`
- `skills show <name>` → `SkillManager.get()` (exact identity, not BM25 rank)
- `skills generate-from-run <run_id> <skill_name> <description>` → `SkillManager.generate_from_program()`
- `skills find` / `skills discover` / `skills categories` → `discovery.py`

### MCP integration
- `mcp/tools.py` imports `SkillManager` to expose skills as agent tools
- `evolve/layers/action.py` imports `SkillManager` to generate skills from agent execution trajectories

## Work Guidance

- Always use `SkillManager` as the public API — it wraps `SkillLoader` and handles directory creation
- `SkillLoader` is a utility class; prefer `SkillManager` unless you only need search/retrieval
- Keep heavy imports lazy: `import dspytools.skills` must load no numpy/redis/falkordb. numpy comes in through `search_embeddings()` and the BM25 bridge; `FalkorDBSkillGraph` is imported inside `compile_skill()`
- When adding a new search backend (e.g., vector DB), add it as a separate method in `SkillLoader` and expose it through a flag — do not change the BM25 default
- New optimizers for `compile_skill` should be added to the `optimizers` dict with lazy lambda wrappers
- `auto_optimize_skill` depends on `GFLPipeline` — ensure the gfl module is importable before calling
- Skills are flat directories; do not nest subdirectories inside a skill
- Skill names should be URL-safe: alphanumeric with hyphens/underscores, no spaces

## Verification

`tests/test_skills_integrity.py` is the dedicated hermetic suite (DummyLM/saved artifacts only, tmp_path skill dirs, monkeypatched graph + embedder):
```bash
uv run --no-sync pytest tests/test_skills_integrity.py -q
```

It covers: per-skill `InstructionProposer` paths plus the empty-library error, YAML round-trip with `": "` / `" #"` / quotes in the description, `skills compile --trainset` reporting the real optimizer and demo count, the `LabeledFewShot` default k, CoT program load by `module_type`, stale-version programs treated as absent, index invalidation on delete/create, hybrid opt-in with model/dimension-keyed caching, best-effort graph writes, `skills show` exact resolution, and lazy imports.

Manual smoke test:
```bash
dspytools skills create my-skill "What this skill does"
dspytools skills compile my-skill --trainset data/train.json
dspytools skills list
dspytools skills search "skill" --hybrid
dspytools skills show my-skill
```

## Child DOX Index

No subdirectories — all files are flat in `src/dspytools/skills/`. No child AGENTS.md files exist.
