"""dspytools memory — Manage FalkorDB-backed persistent memory.

Provides CLI commands for adding, searching, listing, updating, and
deleting memories stored in the FalkorDB graph database with semantic
search via embeddings.
"""

from __future__ import annotations

import datetime
import json

from dspytools.cli.output import error, header, info, ok, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.memory.manager import get_memory_manager


@click.group(name="memory", cls=LLMGroup)
def memory_cmd():
    """Manage FalkorDB persistent memory."""


@memory_cmd.command(name="add", cls=LLMCommand)
@click.argument("content")
@click.option("--user-id", "-u", default="dspytools", help="User ID")
@click.option("--agent-id", "-a", default=None, help="Agent ID")
@click.option("--run-id", "-r", default=None, help="Run ID")
def memory_add(content: str, user_id: str, agent_id: str | None, run_id: str | None):
    """Add a memory to the graph."""

    manager = get_memory_manager()
    result = manager.add(
        content,
        user_id=user_id,
        agent_id=agent_id,
        run_id=run_id,
    )
    ok(f"Memory added: {result.get('id', 'unknown')}")
    if result.get("deduplicated"):
        warn("  (deduplicated — identical content already exists)")
    if result.get("entities"):
        info(f"  Entities: {', '.join(result['entities'])}")
    if result.get("tags"):
        info(f"  Tags: {', '.join(result['tags'])}")


@memory_cmd.command(name="search", cls=LLMCommand)
@click.argument("query")
@click.option("--user-id", "-u", default="dspytools", help="User ID")
@click.option("--agent-id", "-a", default=None, help="Agent ID")
@click.option("--limit", "-l", default=5, type=int, help="Max results")
def memory_search(query: str, user_id: str, agent_id: str | None, limit: int):
    """Search memories by semantic similarity."""

    manager = get_memory_manager()
    results = manager.search(query, user_id=user_id, agent_id=agent_id, limit=limit)

    if not results:
        info("No matching memories found")
        return

    header(f"Memories matching '{query}'")
    rows = []
    for m in results:
        score = m.get("score", 0.0)
        content = m.get("memory", m.get("content", ""))[:80]
        mid = m.get("id", "")[:12]
        tags = ", ".join(m.get("tags", []))
        rows.append([mid, f"{score:.3f}", content, tags])

    table("Results", ["ID", "Score", "Content", "Tags"], rows)


@memory_cmd.command(name="list", cls=LLMCommand)
@click.argument("user_id", required=False, default="dspytools")
@click.option("--agent-id", "-a", default=None, help="Agent ID")
def memory_list(user_id: str, agent_id: str | None):
    """List all memories for a user (default: dspytools).

    Examples:

        dspytools memory list

        dspytools memory list myuser
    """

    manager = get_memory_manager()
    results = manager.get_all(user_id=user_id, agent_id=agent_id)

    if not results:
        info("No memories found")
        return

    header(f"Memories for user '{user_id}'")
    rows = []
    for m in results:
        mid = m.get("id", "")[:12]
        content = m.get("memory", m.get("content", ""))[:80]
        tags = ", ".join(m.get("tags", []))
        created = m.get("created_at", "")
        if isinstance(created, float):
            created = datetime.datetime.fromtimestamp(created).strftime(
                "%Y-%m-%d %H:%M"
            )
        rows.append([mid, content, tags, str(created)[:19]])

    table("Memories", ["ID", "Content", "Tags", "Created"], rows)


@memory_cmd.command(name="get", cls=LLMCommand)
@click.argument("memory_id")
def memory_get(memory_id: str):
    """Get a specific memory by ID."""

    manager = get_memory_manager()
    result = manager.get(memory_id)

    if not result:
        error(f"Memory '{memory_id}' not found")
        return

    header(f"Memory: {result.get('id', memory_id)}")
    info(f"Content: {result.get('memory', result.get('content', ''))}")
    info(f"User: {result.get('user_id', '')}")
    if result.get("agent_id"):
        info(f"Agent: {result['agent_id']}")
    if result.get("run_id"):
        info(f"Run: {result['run_id']}")
    if result.get("tags"):
        info(f"Tags: {', '.join(result['tags'])}")
    if result.get("entities"):
        info(f"Entities: {', '.join(result['entities'])}")
    if result.get("metadata"):
        info(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    info(f"Created: {result.get('created_at', '')}")


@memory_cmd.command(name="update", cls=LLMCommand)
@click.argument("memory_id")
@click.argument("content")
def memory_update(memory_id: str, content: str):
    """Update a memory's content."""

    manager = get_memory_manager()
    result = manager.update(memory_id, content)

    if result.get("error"):
        error(result["error"])
    else:
        ok(f"Memory updated: {memory_id}")


@memory_cmd.command(name="delete", cls=LLMCommand)
@click.argument("memory_id")
@click.confirmation_option(prompt="Delete this memory?")
def memory_delete(memory_id: str):
    """Delete a memory by ID."""

    manager = get_memory_manager()
    result = manager.delete(memory_id)

    if result.get("error"):
        error(result["error"])
    else:
        ok(f"Memory deleted: {memory_id}")


@memory_cmd.command(name="stats", cls=LLMCommand)
@click.option("--user-id", "-u", default="dspytools", help="User ID")
def memory_stats(user_id: str):
    """Show memory statistics."""

    manager = get_memory_manager()
    stats = manager.stats(user_id=user_id)

    header("Memory Statistics")
    info(f"User: {stats.get('user_id', user_id)}")
    info(f"Total memories: {stats.get('total_memories', 0)}")
    info(f"Total entities: {stats.get('total_entities', 0)}")
    info(f"Total tags: {stats.get('total_tags', 0)}")


@memory_cmd.command(name="reset", cls=LLMCommand)
@click.confirmation_option(prompt="This will delete ALL memories. Continue?")
def memory_reset():
    """Reset all memories (dangerous!)."""

    manager = get_memory_manager()
    manager.reset()
    ok("All memories cleared")


@memory_cmd.command(name="history", cls=LLMCommand)
@click.argument("memory_id")
def memory_history(memory_id: str):
    """Show memory history."""

    manager = get_memory_manager()
    results = manager.history(memory_id)

    if not results:
        info(f"No history found for {memory_id}")
        return

    header(f"History for {memory_id}")
    for entry in results:
        info(f"  {entry.get('created_at', '')}: {entry.get('content', '')[:80]}")
