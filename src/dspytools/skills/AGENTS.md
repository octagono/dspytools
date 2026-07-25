# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Purpose

The skills directory implements the Agent Skills system — a BM25-indexed library of reusable DSPy programs with a full lifecycle: **create → compile → optimize → deploy**. Skills follow the harness-so pattern: each skill is a directory containing `SKILL.md` (frontmatter + docs) and `program.json` (compiled DSPy program). Skills are consumed by agent pipelines via MCP tools and the `skills` CLI commands.

## Ownership

- `SkillManager` — primary lifecycle interface: create, compile, optimize, search, and list skills. Used by `commands/skills.py`, `mcp/tools.py`, and `evolve/layers/action.py`. Exported from `__init__.py`.
- `SkillLoader` — BM25-indexed skill library with directory loading from project-local `skills/` and user-wide `~/.config/dspytools/skills/`. Includes hybrid search: BM25 fast retrieval + BGE embedding semantic search. `search()` delegates inner BM25 scoring loop to `bm25_mojo_bridge.py` when available.
- `bm25_mojo_bridge.py` — Mojo SIMD bridge for BM25 scoring (Phase 3). `score_documents(q_tf_matrix, idf_values, doc_lengths, avg_doc_len) → ndarray`. `HAS_MOJO` flag + pure Python fallback.
- `Skill` — dataclass representing a single skill: name, description, path, frontmatter, body, and optional program path. Exported from `__init__.py`.
- `ExternalSkill` + `search_external()` / `popular_skills()` / `list_categories()` — external skill discovery via skills.sh ecosystem. Integrates with npm-based skills registry, returning skills found via CLI search. Exported from `discovery.py`.

## Local Contracts

### Skill directory format
- Skills are stored as directories named by skill name. Each directory contains:
  - `SKILL.md` — YAML-style frontmatter (`name:`, `description:`, `signature:`) followed by body text
  - `program.json` — compiled DSPy program (optional, present after `compile_skill`)
  - `signature.json` — JSON with `{inputs: [...], outputs: [...]}` (written during compile)
- Skills without `SKILL.md` are silently skipped during `load_all()`

### Primary interface contract
- `SkillManager.create_skill(name, description, signature, body="")` → `Skill`
  - Writes `SKILL.md` with frontmatter + body to `{skills_dir}/{name}/`
  - Returns the created `Skill` instance immediately (no compilation)
- `SkillManager.compile_skill(name, trainset=None, optimizer="labeled_few_shot")` → `{status, skill, optimizer, has_program}`
  - Compiles a DSPy program using the skill's signature from frontmatter
  - Supports `"labeled_few_shot"` and `"bootstrap_few_shot"` optimizers
  - Saves compiled program to `{skill_dir}/program.json` and signature metadata to `signature.json`
  - Returns `{error: ...}` dict if skill not found
  - With no trainset, saves the unoptimized base program
- `SkillManager.auto_optimize_skill(name)` → `{status, skill, best, score}` or `{error: ...}`
  - Uses `GFLPipeline` from `dspytools.gfl.pipeline` for full 4-way optimizer comparison
  - Loads existing compiled program, runs GFL pipeline, saves best result back to `program.json`
  - Requires skill to have a compiled program already
- `SkillManager.generate_from_program(run_id, skill_name, description)` → `Skill | None`
  - Copies program.json and signature.json from a compiled run directory into a new skill directory
  - Returns `None` if the source `program.json` doesn't exist
- `SkillManager.list_skills()` → `list[Skill]`
  - Delegates to `SkillLoader.load_all()`
- `SkillManager.search(query, k=5)` → `list[Skill]`
  - Calls `SkillLoader.load_all()` then `SkillLoader.search()`

### SkillLoader contracts
- `SkillLoader(skills_dir=None)` searches two paths:
  1. The provided `skills_dir` (or `Path("skills")` if None)
  2. `~/.config/dspytools/skills/` (user-wide)
- `SkillLoader.load_all()` → `list[Skill]`
  - Iterates all directories under both paths, skips hidden dirs
  - Only loads directories containing `SKILL.md`
  - Builds BM25 inverted index after loading
- `SkillLoader.search(query, k=5)` → `list[Skill]`
  - Standard BM25 scoring: k1=1.2, b=0.75
  - Tokens extracted via regex `[a-zA-Z0-9_-]+` (lowercased)
  - Returns top-k scoring skills; returns all skills (up to k) if no query tokens
- `SkillLoader.search_embeddings(query, k=5)` → `list[Skill]`
  - BGE embedding-based semantic search via `LMRegistry.get(model="openai/embeddinggemma")`
  - Falls back to BM25 search if embedder unavailable
  - Hybrid scoring: embedding cosine similarity + BM25 rank bonus (`0.3 / (rank + 1)`)
  - Gracefully handles embedder exceptions per skill

### BM25 index contract
- Built on tokenized concatenation of `name + description + body`
- `_tokenize(text)`: extracts `[a-zA-Z0-9_-]+` tokens, lowercased
- `_count_docs_with_token(index, token)`: counts documents containing the token (tf > 0)
- IDF formula: `log((N - df + 0.5) / (df + 0.5) + 1.0)`
- BM25 TF: `(tf * (k1+1)) / (tf + k1*(1-b + b*docLen/avgLen))`

### Frontmatter parsing contract
- `_parse_frontmatter(text)` → `(dict, body)`
- Uses `^---\s*\n(.*?)\n---\s*\n` regex (DOTALL mode)
- Parses `key: value` lines; strips quotes from values
- Supports array values: `key: [a, b, c]` parsed as `["a", "b", "c"]`
- Returns empty dict + full text if no frontmatter delimiter found

### CLI integration
- `skills create` → `SkillManager.create_skill()`
- `skills compile` → `SkillManager.compile_skill()`
- `skills optimize` → `SkillManager.auto_optimize_skill()`
- `skills list` → `SkillManager.list_skills()`
- `skills search` → `SkillManager.search()`

### MCP integration
- `mcp/tools.py` imports `SkillManager` to expose skills as agent tools
- `evolve/layers/action.py` imports `SkillManager` to generate skills from agent execution trajectories

## Work Guidance

- Always use `SkillManager` as the public API — it wraps `SkillLoader` and handles directory creation
- `SkillLoader` is a utility class; prefer `SkillManager` unless you only need search/retrieval
- When adding a new search backend (e.g., vector DB), add it as a fallback method in `SkillLoader` similar to `search_embeddings`
- New optimizers for `compile_skill` should be added to the `optimizers` dict with lazy lambda wrappers
- `auto_optimize_skill` depends on `GFLPipeline` — ensure the gfl module is importable before calling
- Skills are flat directories; do not nest subdirectories inside a skill
- The `search_embeddings()` method imports numpy lazily — keep imports inside the method to avoid startup cost
- Skill names should be URL-safe: alphanumeric with hyphens/underscores, no spaces

## Verification

No dedicated test suite exists yet. Verification is manual via CLI commands:
```bash
dspytools skills create --name test-skill --description "Test" --signature "question -> answer"
dspytools skills compile --name test-skill --trainset data.json
dspytools skills optimize --name test-skill
dspytools skills list
dspytools skills search --query "test"
```

Contracts to verify manually:
- `SkillManager.create_skill()` writes correct `SKILL.md` with frontmatter
- `SkillLoader.load_all()` loads skills from both project-local and user-wide paths
- BM25 search returns skills in descending score order
- `search_embeddings()` falls back to BM25 gracefully when embedder is unavailable
- `compile_skill()` with no trainset saves base program without error
- `auto_optimize_skill()` returns error dict for skills without compiled programs
- Frontmatter parsing handles missing, partial, and malformed frontmatter

## Child DOX Index

No subdirectories — all files are flat in `src/dspytools/skills/`. No child AGENTS.md files exist.
