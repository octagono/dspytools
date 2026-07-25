"""dspytools agent — Create, list, run ReAct agents with MCP tools.

Tool Loading Strategy
─────────────────────
MCP servers (git-mcp, mlflow, etc.) expose 50+ tools. Qwen 3B has only 8K
context and tool descriptions consume ~5.5K tokens alone. To stay within
budget, `agent run` uses BM25 relevance ranking to select only the top-k
most relevant tools for each question, instead of loading everything.

Architecture
────────────
1. `agent new` — saves agent config with all tool metadata (not just names)
2. `agent run` — loads full tool pool, ranks by BM25 against the question,
   passes only top-k to ReAct. The agent never sees irrelevant tools.
3. BM25 ranker is a lightweight `_rank_tools()` function in this module
   (no numpy dependency). It tokenizes tool names + descriptions and scores
   against the question tokens using standard BM25 with k1=1.2, b=0.75.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import agents_dir
from dspytools.core._io import read_json, write_json
from dspytools.core.setup import setup_dspy
from dspytools.mcp.loader import load_mcp_tools_sync

# ── BM25 Tool Ranker ──────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_-]+")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased alphanumeric tokens."""
    return _TOKEN_RE.findall(text.lower())


def _build_tool_index(tools: list[Any]) -> dict[str, dict[str, Any]]:
    """Build an index of tool name → {tokens, tf} for BM25 scoring."""
    index: dict[str, dict[str, Any]] = {}
    for t in tools:
        text = f"{t.name} {t.desc}"
        tokens = _tokenize(text)
        index[t.name] = {
            "tokens": tokens,
            "tf": Counter(tokens),
            "tool": t,
        }
    return index


def _rank_tools(
    query: str,
    tools: list[Any],
    k: int = 5,
    k1: float = 1.2,
    b: float = 0.75,
) -> list[tuple[Any, float]]:
    """Rank tools by BM25 relevance to the query string.

    Returns top-k (tool, score) tuples, best-first.
    """
    if not query or not tools:
        return [(t, 0.0) for t in tools[:k]]

    index = _build_tool_index(tools)
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [(t, 0.0) for t in tools[:k]]

    N = len(index)
    avg_len = sum(v["tf"].total() for v in index.values()) / max(N, 1)

    scores: list[tuple[Any, float]] = []
    for name, entry in index.items():
        dl = len(entry["tokens"])
        tf = entry["tf"]
        q_set = set(query_tokens)

        doc_score = 0.0
        for qt in q_set:
            tft = tf.get(qt, 0)
            if tft == 0:
                continue
            df = sum(1 for v in index.values() if qt in v["tokens"])
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            doc_score += (
                idf * (tft * (k1 + 1)) / (tft + k1 * (1 - b + b * dl / max(avg_len, 1)))
            )

        scores.append((entry["tool"], doc_score))

    scores.sort(key=lambda x: -x[1])
    return scores[:k]


# ── CLI Commands ──────────────────────────────────────────────────────────


@click.group(name="agent", cls=LLMGroup)
def agent_cmd():
    """Manage ReAct agents with MCP tools."""


@agent_cmd.command(name="new", cls=LLMCommand)
@click.argument("name")
@click.option(
    "--signature",
    "-s",
    default="question -> answer",
    help="Agent signature (default: 'question -> answer')",
)
@click.option("--mcp-config", "-c", default=".mcp.json", help="MCP config file path")
@click.option("--max-iters", default=10, type=int, help="Max agent iterations")
def agent_new(name: str, signature: str, mcp_config: str, max_iters: int):
    """Create a new ReAct agent with MCP tools.

    NAME: Agent name (e.g., git-analyzer)

    Scans MCP servers (git-mcp, mlflow, etc.) and saves the full tool
    inventory so `agent run` can select the right tools on-the-fly.
    """
    click.echo(f"  Loading MCP tools from {mcp_config}...")
    sessions, tools = load_mcp_tools_sync(mcp_config)

    if not tools:
        click.echo("  No MCP tools loaded. Agent will have no tools.")
    else:
        click.echo(f"  Indexed {len(tools)} MCP tools")

    # Save agent config
    _tools = tools or []
    agent_config = {
        "name": name,
        "signature": signature,
        "max_iters": max_iters,
        "mcp_config": mcp_config,
        "tool_count": len(_tools),
        "tool_names": [t.name for t in _tools],
    }

    agent_path = agents_dir() / f"{name}.json"
    write_json(agent_path, agent_config)
    click.echo(f"  Agent '{name}' saved to {agent_path}")

    if tools:
        assert tools is not None  # narrow for pyright
        click.echo(f"\n  Tools available ({len(tools)} total):")
        # Group by server prefix (first segment before underscore)
        for t in tools:
            desc_short = (t.desc or "")[:80].replace("\n", " ").strip()
            click.echo(f"    • {t.name}: {desc_short}...")
        click.echo("")
        click.echo("  Run `dspytools agent run <name> <question>` to use the agent.")
        click.echo(
            "  Only the top-k most relevant tools are loaded per-question (BM25)."
        )


@agent_cmd.command(name="list", cls=LLMCommand)
def agent_list():
    """List saved agents."""
    from dspytools.core.registry import list_agents

    agents = list_agents()
    if agents:
        for a in agents:
            click.echo(f"  {a['name']}  ({a['size']}B)")
    else:
        click.echo("  No agents configured")


@agent_cmd.command(name="run", cls=LLMCommand)
@click.argument("name")
@click.argument("question")
@click.option("--model", help="LM to use for the agent")
@click.option("--max-iters", default=10, type=int, help="Max agent iterations")
@click.option(
    "--top-k",
    default=5,
    type=int,
    help="Number of top-ranked tools to load per question (default: 5)",
)
@click.option(
    "--show-tools/--hide-tools",
    default=False,
    help="Show which tools were selected for this question",
)
def agent_run(
    name: str,
    question: str,
    model: str | None,
    max_iters: int,
    top_k: int,
    show_tools: bool,
):
    """Run an agent with a question.

    Only the top-k most relevant tools (by BM25 score against the question)
    are loaded into the ReAct agent, keeping the prompt within the LM's
    context window.

    Examples:

    \b
        dspytools agent run my-agent "list git repositories"
        dspytools agent run my-agent "search mlflow traces" --top-k 3
        dspytools agent run my-agent "compare two programs" --show-tools
    """
    setup_dspy(model=model)

    agent_path = agents_dir() / f"{name}.json"
    if not agent_path.exists():
        click.echo(f"  Agent '{name}' not found", err=True)
        raise click.Abort()

    config = read_json(agent_path)

    # Load full MCP tool pool
    click.echo(f"  Loading MCP tools from {config.get('mcp_config', '.mcp.json')}...")
    sessions, tools = load_mcp_tools_sync(config.get("mcp_config", ".mcp.json"))

    if not tools:
        click.echo("  No MCP tools available. Running without tools.")
    else:
        click.echo(f"  Pool: {len(tools)} tools indexed")

    # Rank and select top-k tools for this question
    if tools:
        ranked = _rank_tools(question, tools, k=top_k)
        selected = [t for t, _ in ranked]
        click.echo(
            f"  Selected {len(selected)}/{len(tools)} tools "
            f"(BM25 rank, top-{top_k} for question)"
        )
        for t, score in ranked:
            desc_short = t.desc[:60].replace("\n", " ")
            if show_tools:
                click.echo(f"    {score:.3f}  {t.name}: {desc_short}...")
            else:
                click.echo(f"    • {t.name}")
        tools = selected
    else:
        selected = tools

    from dspytools.core._dspy import dspy

    # Use ReAct v1 — text-based parsing (Thought/Action/Observation) that
    # works reliably with small models (Qwen 7B). ReActV2 requires native
    # function calling (BAML/JSON schema) which Qwen 7B can't produce.
    agent = dspy.ReAct(
        "question -> answer",
        tools=selected,
        max_iters=min(max_iters, config.get("max_iters", 10)),
    )

    click.echo("  Running agent...")
    result = agent(question=question)
    answer = getattr(result, "answer", str(result))
    click.echo(f"\n  Answer: {answer}")


@agent_cmd.command(name="delete", cls=LLMCommand)
@click.argument("name")
@click.confirmation_option(prompt="Delete this agent?")
def agent_delete(name: str):
    """Delete a saved agent."""
    from dspytools.core.registry import delete_agent

    if delete_agent(name):
        click.echo(f"  Deleted agent '{name}'")
    else:
        click.echo(f"  Agent '{name}' not found", err=True)
        raise click.Abort()
