# DOX — graph

## Purpose

FalkorDB + Redis graph database integration for dspytools. Provides O(1) graph traversal for skill dependencies, program lineage tracking, semantic caching, and general-purpose Redis-backed response caching.

## Ownership

- `GraphClient` — singleton FalkorDB + Redis client with connection pooling. **SSOT** for all graph/cache operations.
- `FalkorDBSkillGraph` — skill dependency graph. **SSOT** — `SelfEvolveEngine.SkillGraph` delegates here, JSON is fallback only.
- `SemanticCache` — embedding-based semantic cache for LLM responses (FalkorDB native vector index)
- `RedisCache` — general-purpose Redis cache with namespace isolation and TTL
- `cache_mojo_bridge.py` — Mojo SIMD bridge for vector blob serialization (Phase 1). `HAS_MOJO` flag + NumPy fallback. Standalone benchmark target — no longer consumed by `SemanticCache` (FalkorDB native vector search replaced blob storage).
- `benchmark.py` — Correctness fuzz + throughput benchmark for Phase 1 vector serialization. `python -m dspytools.graph.benchmark --all`.

## Local Contracts

### `client.py` — GraphClient singleton

- `GraphClient()` — singleton via `__new__`, lazy FalkorDB + Redis initialization (no `@lru_cache` wrapper needed since `__new__` already ensures singleton).
- `graph(name)` — returns FalkorDB graph for Cypher queries
- `redis()` — returns Redis client for vector search and caching
- `ping()` — checks connection health
- `ensure_indexes()` — creates FalkorDB native vector indexes for semantic search:
  - Vector index on `Skill.embedding` (default graph)
  - Vector index on `CacheVec.embedding` (`llm_cache` graph, for `SemanticCache`)
  - Vector index on `Memory.embedding` (`memories` graph, for `MemoryManager`)
  - All use `ensure_vector_index()` — idempotent (catches "already indexed" error)
  - Dimension from `embedder_dimension()` (env `DSPYTOOLS_EMBEDDING_DIM`, default 768 for embeddinggemma)
- `ensure_vector_index(graph_name, label, property, dim)` — creates a FalkorDB vector index idempotently. FalkorDB lacks `CREATE VECTOR INDEX IF NOT EXISTS`, so catches "already indexed" error and ignores it. All other errors propagate.
- `flush_all()` — flushes all data (use with caution)
- `_load_config()` — reads FalkorDB connection from ConfigCache (`falkordb.host`, `falkordb.port`, `falkordb.password`)

### `skill_graph.py` — FalkorDBSkillGraph

- `add_dependency(skill, depends_on)` — records dependency edge
- `get_dependencies(skill)` — returns dependencies for a skill
- `get_dependents(skill)` — returns skills that depend on this skill
- `transitive_dependents(skill)` — BFS traversal for all indirect dependents
- `record_task_profile(profile, pattern_type, success)` — tracks task execution
- `record_program(run_id, optimizer, score, parent_id, dataset_hash)` — tracks program lineage
- `program_lineage(run_id)` — returns full ancestry chain
- `list_skills()` — returns all skills in graph
- `skill_stats(name)` — returns per-skill statistics
- `_rows_to_dicts(result)` — converts FalkorDB result_set (list-of-lists) to list of dicts
- `_col_values(result, col_idx)` — extracts single column from result_set

### `cache.py` — SemanticCache

- Two-tier: SHA-256 exact-match (O(1), Redis GET/SETEX) + FalkorDB native vector index (O(log N), `db.idx.vector.queryNodes`)
- Tier 1 stores at `{name}:exact:{hash}` as JSON with optional `_vector` for cache-thaw
- Tier 2 stores `:CacheVec` graph nodes with `embedding` (vecf32), `response`, `prompt`, `metadata` properties
- `check(prompt)` — returns cached response if found. On Tier 2 hit via `db.idx.vector.queryNodes`, returns distance and cached vector (`_vector` key) so `store()` can avoid recomputing the embedding on miss. Score = distance (lower = more similar).
- `store(prompt, response, metadata, _vector=None)` — accepts optional pre-computed vector from `check()` miss to avoid duplicate embedding call. Creates `:CacheVec` node with `vecf32($vec)` embedding.
- `clear()` — clears Tier 1 via Redis SCAN + Tier 2 via `MATCH (c:CacheVec) DETACH DELETE c`
- `stats()` — returns cache statistics (exact_entries, semantic_entries, total_entries, hit_count, miss_count, hit_rate_pct)
- `__init__()` accepts `model_name` param — loads per-model distance threshold from config via `cache_threshold_path()`, falling back to `"default"` key. Creates vector index via `ensure_vector_index()`.
- `adjust_threshold(drift_delta: float) → None` — dynamically tightens/loosens the similarity threshold based on quality drift. When `drift_delta > 0.05` (quality dropping), tightens by 10% per call (min 0.01). When `drift_delta < 0` (stable/improving), loosens by 10% per call (max 0.30). Wired into `HotSwapManager.infer()` — triggers on critical drift alerts.
- Uses shared embedder singleton via `dspytools.core._embedder.get_embedder()`
- No RediSearch/FT.SEARCH dependency — FalkorDB container lacks RediSearch module

### `redis_cache.py` — General-purpose Redis cache

- `RedisCache(namespace, default_ttl, max_entries)` — namespaced cache with TTL
- `get(key)`, `set(key, value, ttl)`, `delete(key)` — basic operations
- `exists(key)`, `ttl(key)`, `expire(key, ttl)` — key management
- `keys(pattern)`, `flush()`, `count()`, `memory_usage()` — bulk operations (all use SCAN cursor, not blocking KEYS)
- `stats()` — returns entries, memory, avg TTL
- `_evict_if_needed()` — approximate eviction: samples excess + 2 keys via random.sample, evicts oldest-TTL within that sample (avoids sorting all keys)
- `get_mcp_cache()` — singleton for MCP tool responses (5s TTL, 256 entries)
- `get_compile_cache()` — singleton for compile results (1hr TTL, 64 entries)

### `migrate.py` — JSON → FalkorDB migration

- `migrate_all()`, `migrate_skill_graph()`, `migrate_morphology()`, `migrate_program_registry()`
- Uses `config_dir()` / `compiled_dir()` for path resolution

## Work Guidance

- Always use `get_graph_client()` for FalkorDB/Redis access
- FalkorDB handles graph queries (Cypher), Redis handles vector search and caching
- FalkorDB `result.result_set` is list-of-lists `[[val1, val2]]`, NOT list-of-dicts
- All code must use `row[index]` not `row["key"]` or `row.values()`
- New dependencies should be added via `add_dependency()` for tracking
- Semantic cache uses cosine similarity with configurable distance threshold
- `RedisCache` namespaces prevent key collisions between MCP, compile, and graph data
- MCP tool cache (5s TTL) is Redis-backed — persists across MCP server restarts
- CircuitBreaker wraps FalkorDB/Redis connection calls — service down doesn't crash the process

## Verification

- `dspytools graph status` — checks FalkorDB connection
- `dspytools graph skill-tree` — visualizes skill dependencies
- `dspytools graph query "MATCH (n) RETURN count(n)"` — runs Cypher query
- `dspytools graph redis status` — Redis connection + version
- `dspytools graph redis stats` — MCP + compile cache statistics
- `dspytools graph redis keys` — list cached keys

## Child DOX Index

No subdirectories — all files are flat in `src/dspytools/graph/`. No child AGENTS.md files exist.
