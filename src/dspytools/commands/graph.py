"""dspytools graph — Manage FalkorDB graph database.

Provides commands for querying, visualizing, and migrating graph data.
"""

from __future__ import annotations

import json
import time as _time
from collections import deque

from dspytools.cli.output import error, header, info, ok, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.graph.client import get_graph_client
from dspytools.graph.migrate import (
    migrate_all,
    migrate_morphology,
    migrate_program_registry,
    migrate_skill_graph,
)
from dspytools.graph.skill_graph import FalkorDBSkillGraph


@click.group(name="graph", cls=LLMGroup)
def graph_cmd():
    """Manage FalkorDB graph database."""


@graph_cmd.command(name="status", cls=LLMCommand)
def graph_status():
    """Show FalkorDB connection status."""
    client = get_graph_client()
    if client.ping() and client._falkordb is not None:
        ok("FalkorDB connection: OK")
        graphs = client._falkordb.list_graphs()
        if graphs:
            info(f"Graphs: {', '.join(graphs)}")
        else:
            info("No graphs found")
    else:
        error("FalkorDB connection: FAILED")


@graph_cmd.command(name="query", cls=LLMCommand)
@click.argument("cypher_query")
@click.option("--params", "-p", help="JSON parameters for query")
def graph_query(cypher_query: str, params: str | None):
    """Execute a Cypher query on FalkorDB."""
    client = get_graph_client()
    graph = client.graph()

    parsed_params = json.loads(params) if params else {}
    result = graph.query(cypher_query, parsed_params)

    if result.result_set:
        rows = [[str(v) for v in row] for row in result.result_set]
        # Extract column names from header if available
        headers = []
        if hasattr(result, "header") and result.header:
            headers = [h[1] for h in result.header]
        else:
            headers = [f"col{i}" for i in range(len(rows[0]))] if rows else []
        table("Query Result", headers, rows)
    else:
        info("No results")


@graph_cmd.command(name="skill-tree", cls=LLMCommand)
@click.option("--skill", "-s", help="Show dependencies for specific skill")
def graph_skill_tree(skill: str | None):
    """Visualize skill dependency tree."""
    graph = FalkorDBSkillGraph()

    if skill:
        deps = graph.get_dependencies(skill)
        dependents = graph.get_dependents(skill)
        transitive = graph.transitive_dependents(skill)
        stats = graph.skill_stats(skill)

        header(f"Skill: {skill}")
        info(f"Dependencies: {', '.join(deps) if deps else 'none'}")
        info(f"Direct dependents: {', '.join(dependents) if dependents else 'none'}")
        info(
            f"Transitive dependents: {', '.join(transitive) if transitive else 'none'}"
        )
        info(f"Stats: {json.dumps(stats, indent=2)}")
    else:
        skills = graph.list_skills()
        if skills:
            rows = [[s["name"], (s.get("description") or "")[:60]] for s in skills]
            table("Skills", ["Name", "Description"], rows)
        else:
            info("No skills in graph")


@graph_cmd.command(name="program-lineage", cls=LLMCommand)
@click.argument("run_id")
def graph_program_lineage(run_id: str):
    """Show program ancestry chain."""
    graph = FalkorDBSkillGraph()
    lineage = graph.program_lineage(run_id)

    if lineage:
        rows = [
            [r["id"], r.get("optimizer", ""), str(r.get("score", ""))] for r in lineage
        ]
        table("Program Lineage", ["Run ID", "Optimizer", "Score"], rows)
    else:
        info(f"No lineage found for {run_id}")


@graph_cmd.command(name="migrate", cls=LLMCommand)
@click.option(
    "--target", type=click.Choice(["all", "skills", "morphology", "programs"])
)
def graph_migrate(target: str):
    """Migrate existing JSON data to FalkorDB."""
    header(f"Migrating {target}...")

    if target == "all":
        results = migrate_all()
    elif target == "skills":
        results = migrate_skill_graph()
    elif target == "morphology":
        results = migrate_morphology()
    elif target == "programs":
        results = migrate_program_registry()
    else:
        results = {"error": "Unknown target"}

    ok(f"Migration complete: {json.dumps(results, indent=2)}")


@graph_cmd.command(name="add-dependency", cls=LLMCommand)
@click.argument("skill")
@click.argument("depends_on")
def graph_add_dependency(skill: str, depends_on: str):
    """Add a dependency edge between two skills."""
    graph = FalkorDBSkillGraph()
    graph.add_dependency(skill, depends_on)
    ok(f"Dependency added: {skill} → {depends_on}")


@graph_cmd.command(name="dependents", cls=LLMCommand)
@click.argument("skill")
@click.option("--transitive", "-t", is_flag=True, help="Include transitive dependents")
def graph_dependents(skill: str, transitive: bool):
    """Show skills that depend on a given skill."""
    graph = FalkorDBSkillGraph()

    if transitive:
        deps = graph.transitive_dependents(skill)
        label = "Transitive dependents"
    else:
        deps = graph.get_dependents(skill)
        label = "Direct dependents"

    if deps:
        rows = [[d] for d in deps]
        table(label, ["Skill"], rows)
    else:
        info(f"No dependents for '{skill}'")


@graph_cmd.command(name="record-program", cls=LLMCommand)
@click.argument("run_id")
@click.argument("optimizer")
@click.option("--score", "-s", type=float, default=0.0, help="Quality score")
@click.option("--dataset-hash", "-d", default=None, help="Dataset hash")
@click.option("--parent-id", "-p", default=None, help="Parent run ID")
def graph_record_program(
    run_id: str,
    optimizer: str,
    score: float,
    dataset_hash: str | None,
    parent_id: str | None,
):
    """Record a compiled program in the graph."""
    graph = FalkorDBSkillGraph()
    graph.record_program(
        run_id=run_id,
        optimizer=optimizer,
        score=score,
        dataset_hash=dataset_hash,
        parent_id=parent_id,
    )
    ok(f"Program recorded: {run_id} (optimizer={optimizer}, score={score})")


@graph_cmd.command(name="search", cls=LLMCommand)
@click.argument("query")
@click.option("--limit", "-l", default=10, type=int, help="Max results")
def graph_search(query: str, limit: int):
    """Search skills by name or description."""
    graph = FalkorDBSkillGraph()
    skills = graph.list_skills()

    # Simple substring search
    results = []
    for s in skills:
        name = s.get("name", "")
        desc = s.get("description") or ""
        if query.lower() in name.lower() or query.lower() in desc.lower():
            results.append(s)

    if results:
        rows = [[s["name"], (s.get("description") or "")[:60]] for s in results[:limit]]
        table("Search Results", ["Name", "Description"], rows)
    else:
        info(f"No skills matching '{query}'")


@graph_cmd.command(name="stats", cls=LLMCommand)
def graph_stats_cmd():
    """Show detailed graph statistics."""
    client = get_graph_client()
    g = client.graph()

    # Count all node labels
    node_result = g.query("MATCH (n) RETURN labels(n) AS labels, count(n) AS cnt")
    edge_result = g.query("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt")

    header("Graph Statistics")
    info("  Node labels:")
    if node_result.result_set:
        for row in node_result.result_set:
            info(f"    {row[0]}: {row[1]}")
    else:
        info("    (none)")

    info("  Edge types:")
    if edge_result.result_set:
        for row in edge_result.result_set:
            info(f"    {row[0]}: {row[1]}")
    else:
        info("    (none)")


@graph_cmd.command(name="flush", cls=LLMCommand)
@click.confirmation_option(prompt="This will delete ALL graph data. Continue?")
def graph_flush():
    """Flush all graph data (dangerous!)."""
    client = get_graph_client()
    client.flush_all()
    ok("All graph data flushed")


@graph_cmd.command(name="benchmark", cls=LLMCommand)
@click.option(
    "--queries", "-q", default=10, type=int, help="Number of benchmark queries to run"
)
@click.option(
    "--warmup", "-w", default=3, type=int, help="Warmup iterations before measurement"
)
def graph_benchmark(queries: int, warmup: int):
    """Benchmark FalkorDB query latency (p50/p95/p99).

    Runs a standard set of Cypher queries against the graph to measure
    response time percentiles. Useful for performance validation.
    """

    client = get_graph_client()
    g = client.graph()

    benchmarks = [
        ("MATCH (n) RETURN count(n)", "node_count"),
        ("MATCH ()-[r]->() RETURN type(r), count(r)", "edge_types"),
        ("MATCH (n) RETURN labels(n), count(n)", "node_labels"),
    ]

    header("FalkorDB Benchmark")
    info(f"  Warmup: {warmup} iterations")
    info(f"  Measured: {queries} iterations per query")

    for cypher, label in benchmarks:
        # Warmup
        for _ in range(warmup):
            g.query(cypher)

        # Measured runs
        latencies = []
        for _ in range(queries):
            start = _time.perf_counter()
            g.query(cypher)
            elapsed = (_time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        if not latencies:
            error(f"  {label}: all {queries} queries failed — skipping")
            continue

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95_idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        p99_idx = min(int(len(latencies) * 0.99), len(latencies) - 1)
        p95 = latencies[p95_idx]
        p99 = latencies[p99_idx]

        info(
            f"  {label}: p50={p50:.2f}ms  p95={p95:.2f}ms  p99={p99:.2f}ms  ({len(latencies)} samples)"
        )

    ok("Benchmark complete")


@graph_cmd.command(name="cascade", cls=LLMCommand)
@click.argument("skill")
@click.option(
    "--depth", "-d", default=0, type=int, help="Max cascade depth (0 = unlimited)"
)
@click.option(
    "--dry-run", is_flag=True, default=True, help="Show cascade without executing"
)
def graph_cascade(skill: str, depth: int, dry_run: bool):
    """Trace downstream dependents that need re-optimization when a skill improves.

    If --depth is specified, limits cascade to N levels of transitive dependents.
    Use --no-dry-run to trigger recompile queue for all affected skills.
    """
    from dspytools.evolve.self_evolve import SkillGraph

    sg = SkillGraph()
    all_deps = sg.transitive_dependents(skill)

    if depth > 0:
        # BFS with depth limit
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(skill, 0)])
        limited: list[str] = []
        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for dep in sg.get_dependents(current):
                if dep not in visited:
                    visited.add(dep)
                    limited.append(dep)
                    queue.append((dep, d + 1))
        all_deps = limited

    if not all_deps:
        info(f"No downstream dependents for '{skill}'")
        return

    info(f"Cascade from '{skill}' affects {len(all_deps)} skill(s):")
    for dep in all_deps:
        info(f"  - {dep}")

    if dry_run:
        info("")
        info("Use --no-dry-run to queue recompilation for affected skills")
        return

    # Queue recompilation
    from dspytools.core.drift_monitor import get_drift_monitor

    monitor = get_drift_monitor()
    for dep in all_deps:
        monitor.request_recompile(dep)
        ok(f"  Queued recompile: {dep}")

    info(
        f"{len(all_deps)} recompile(s) queued — run `dspytools self auto-fix --no-dry-run` to process"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Redis Cache Commands


@graph_cmd.group(name="redis")
def redis_cmd():
    """Manage Redis cache (MCP response cache, compile cache)."""


@redis_cmd.command(name="status", cls=LLMCommand)
def redis_status():
    """Show Redis connection status and cache statistics."""
    client = get_graph_client()
    if client.ping():
        ok("Redis connection: OK")
        info_obj = client.redis().info("server")
        info(f"Redis version: {info_obj.get('redis_version', '?')}")
        info(f"Used memory: {info_obj.get('used_memory_human', '?')}")
    else:
        error("Redis connection: FAILED")


@redis_cmd.command(name="stats", cls=LLMCommand)
def redis_stats():
    """Show cache statistics for all namespaces."""
    from dspytools.graph.redis_cache import get_compile_cache, get_mcp_cache

    mcp = get_mcp_cache().stats()
    compile_c = get_compile_cache().stats()

    header("MCP Response Cache")
    info(f"  Entries: {mcp['entries']}/{mcp['max_entries']}")
    info(f"  Memory: {mcp['memory_human']}")
    info(f"  Default TTL: {mcp['default_ttl']}s")
    info(f"  Avg TTL remaining: {mcp['avg_ttl_remaining']}s")

    header("Compile Cache")
    info(f"  Entries: {compile_c['entries']}/{compile_c['max_entries']}")
    info(f"  Memory: {compile_c['memory_human']}")
    info(f"  Default TTL: {compile_c['default_ttl']}s")


@redis_cmd.command(name="flush", cls=LLMCommand)
@click.option(
    "--namespace", "-n", type=click.Choice(["mcp", "compile", "all"]), default="all"
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def redis_flush(namespace: str, yes: bool):
    """Flush Redis cache entries."""
    from dspytools.graph.redis_cache import get_compile_cache, get_mcp_cache

    if not yes:
        click.confirmation = click.confirm(f"Flush {namespace} cache?")
        if not click.confirmation:
            info("Aborted")
            return

    flushed = 0
    if namespace in ("mcp", "all"):
        flushed += get_mcp_cache().flush()
    if namespace in ("compile", "all"):
        flushed += get_compile_cache().flush()
    ok(f"Flushed {flushed} entries from {namespace} cache")


@redis_cmd.command(name="get", cls=LLMCommand)
@click.argument("key")
@click.option("--namespace", "-n", type=click.Choice(["mcp", "compile"]), default="mcp")
def redis_get(key: str, namespace: str):
    """Get a value from Redis cache."""
    from dspytools.graph.redis_cache import RedisCache

    cache = RedisCache(namespace=namespace)
    val = cache.get(key)
    if val is None:
        info(f"Cache miss: {key}")
    else:
        info(f"Cache hit: {key}")
        info(f"Value: {json.dumps(val, default=str, indent=2)[:500]}")


@redis_cmd.command(name="set", cls=LLMCommand)
@click.argument("key")
@click.argument("value")
@click.option("--namespace", "-n", type=click.Choice(["mcp", "compile"]), default="mcp")
@click.option("--ttl", "-t", type=int, default=300, help="TTL in seconds")
def redis_set(key: str, value: str, namespace: str, ttl: int):
    """Set a value in Redis cache."""
    from dspytools.graph.redis_cache import RedisCache

    cache = RedisCache(namespace=namespace)
    cache.set(key, value, ttl=ttl)
    ok(f"Set {namespace}:{key} (TTL={ttl}s)")


@redis_cmd.command(name="keys", cls=LLMCommand)
@click.option(
    "--namespace", "-n", type=click.Choice(["mcp", "compile", "all"]), default="mcp"
)
@click.option("--pattern", "-p", default="*", help="Key pattern filter")
def redis_keys(namespace: str, pattern: str):
    """List Redis cache keys."""
    from dspytools.graph.redis_cache import RedisCache

    namespaces = ["mcp", "compile"] if namespace == "all" else [namespace]
    all_keys = []
    for ns in namespaces:
        cache = RedisCache(namespace=ns)
        for k in cache.keys(pattern):
            all_keys.append(f"{ns}:{k}")

    if all_keys:
        rows = [[k] for k in all_keys[:50]]
        table("Cache Keys", ["Key"], rows)
    else:
        info("No cache keys found")
