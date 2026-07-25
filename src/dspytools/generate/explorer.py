"""Repository exploration — file tree gathering + MCP Git ReAct agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dspytools.core.logging_config import get_logger

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.core._io import read_json
from dspytools.mcp.loader import MCPSessionPool


def gather_repository_info(
    repo_dir: str | None = None,
) -> tuple[str, str, str, dspy.History]:
    """Gather local repository info: file_tree, readme, package_files, history.

    Falls back to simulated DSPy repo analysis when no path given.
    """

    messages: list[dict] = []

    if repo_dir and Path(repo_dir).exists():
        root = Path(repo_dir)
        files = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file()
            and not any(part.startswith((".", "__pycache__")) for part in p.parts)
        )
        file_tree = "\n".join(files)
        messages.append(
            {
                "step": "gather",
                "content": f"Local scan: {len(files)} files in {repo_dir}",
            }
        )
    else:
        file_tree = """dspy/__init__.py
dspy/adapters/__init__.py
dspy/adapters/chat_adapter.py
dspy/adapters/json_adapter.py
dspy/adapters/xml_adapter.py
dspy/adapters/baml_adapter.py
dspy/clients/__init__.py
dspy/clients/base_lm.py
dspy/clients/lm.py
dspy/predict/__init__.py
dspy/predict/predict.py
dspy/predict/chain_of_thought.py
dspy/predict/react.py
dspy/predict/react_v2.py
dspy/predict/code_act.py
dspy/predict/program_of_thought.py
dspy/predict/rlm.py
dspy/predict/best_of_n.py
dspy/predict/refine.py
dspy/predict/parallel.py
dspy/teleprompt/__init__.py
dspy/teleprompt/mipro_v2.py
dspy/teleprompt/gepa.py
dspy/teleprompt/bootstrap.py
dspy/teleprompt/better_together.py
dspy/teleprompt/ensemble.py
dspy/teleprompt/knn_fewshot.py
dspy/primitives/example.py
dspy/primitives/prediction.py
dspy/primitives/module.py
dspy/primitives/history.py
dspy/evaluate/__init__.py
dspy/evaluate/evaluate.py
dspy/datasets/__init__.py
dspy/streaming/__init__.py
dspy/streaming/stream_listener.py
dspy/tools/__init__.py
pyproject.toml
README.md"""
        messages.append(
            {"step": "gather", "content": "Simulated DSPy file tree (no GitHub API)"}
        )

    readme_content = """# DSPy: Programming—not Prompting—Foundation Models

DSPy is a framework for algorithmically optimizing LM prompts
and weights, especially when LMs are used multiple times in a pipeline.

## Key Features
- **Declarative Signatures**: Define I/O contracts
- **Composable Modules**: Stack and chain LM calls
- **Automatic Optimization**: MIPROv2, GEPA, BootstrapFewShot
- **Tool Use**: ReAct, CodeAct, ProgramOfThought
- **Streaming & Async**: Full async/streaming support"""
    messages.append(
        {"step": "readme", "content": f"README: {len(readme_content)} chars"}
    )

    package_files = """=== pyproject.toml ===
[project]
name = "dspy-ai"
version = "3.2.1"
description = "Programming—not prompting—Foundation Models"
requires-python = ">=3.10"
dependencies = ["litellm", "pydantic", "requests"]"""
    messages.append({"step": "packages", "content": "pyproject.toml parsed"})

    # Code primitive
    setup_code = dspy.Code(
        code="\n".join(
            line for line in file_tree.split("\n") if line.startswith("dspy/")
        ),
    )
    messages.append(
        {
            "step": "code_primitive",
            "content": f"Code primitive: {len(setup_code.code)} chars",
        }
    )

    # Image primitive
    architecture_image = dspy.Image(url="https://dspy.ai/static/dspy-logo.svg")
    messages.append(
        {
            "step": "image_primitive",
            "content": f"Image primitive: {architecture_image.url}",
        }
    )

    history = dspy.History(messages=messages)
    return file_tree, readme_content, package_files, history


# ── MCP Git ReAct Explorer ─────────────────────────────────────────────────


def load_mcp_tools_sync() -> tuple[list[Any], list[dspy.Tool]]:
    """Load MCP git tools from .mcp.json config.

    Returns (sessions, tools) — tools are dspy.Tool instances.
    """
    sessions: list[Any] = []
    tools: list[dspy.Tool] = []

    # Try to load MCP config
    mcp_config_path = Path.cwd() / ".mcp.json"
    if not mcp_config_path.exists():
        return sessions, tools

    try:
        config = read_json(mcp_config_path)
        mcp_servers = config.get("mcpServers", {})
    except (json.JSONDecodeError, OSError) as e:
        get_logger(__name__).warning("Failed to load .mcp.json: %s", e)
        return sessions, tools

    if not mcp_servers:
        return sessions, tools

    pool = MCPSessionPool()
    for server_name, server_config in mcp_servers.items():
        session = pool.create_session(server_name, server_config)
        sessions.append(session)
        for tool in session.tools:
            tools.append(tool)

    return sessions, tools


class GitRepoExplorer(dspy.Module):
    """ReAct agent with git MCP tools for deep repo analysis.

    Uses git-mcp tools to:
      - List files/directories
      - Show git history
      - Read file contents
      - Identify project structure

    Then generates a comprehensive llms.txt.
    """

    def __init__(self, mcp_tools: list[dspy.Tool], teacher: dspy.LM | None = None):
        super().__init__()
        self.mcp_tools = mcp_tools

        # ReAct v1 — text-based parsing for small model compatibility (Qwen 7B)
        self.explorer = dspy.ReAct(
            "repo_path, question -> answer",
            tools=mcp_tools,
            max_iters=12,
        )

        self.summarizer = dspy.ChainOfThought(
            "repo_path, exploration_summary -> purpose, key_concepts, file_tree, entry_points"
        )

        self.generator = dspy.ChainOfThought(
            "repo_path, purpose, key_concepts, file_tree, entry_points -> llms_txt_content"
        )

    def forward(self, repo_path: str) -> dspy.Prediction:
        """Analyze a local git repo and produce llms.txt using MCP tools."""
        exploration_questions = [
            "List all files and directories in this repository",
            "Show the git log for the last 10 commits",
            "Read the README file and describe what this project does",
            "Identify the main entry points and key files",
            "Read pyproject.toml, setup.py, or package config files",
        ]

        exploration_results = []
        for question in exploration_questions[:3]:
            result = self.explorer(repo_path=str(repo_path), question=question)
            exploration_results.append(f"Q: {question}\nA: {result.answer}")

        exploration_summary = "\n\n".join(exploration_results)

        summary = self.summarizer(
            repo_path=str(repo_path),
            exploration_summary=exploration_summary,
        )

        llms_txt = self.generator(
            repo_path=str(repo_path),
            purpose=summary.purpose,
            key_concepts=summary.key_concepts,
            file_tree=summary.file_tree,
            entry_points=summary.entry_points,
        )

        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            purpose=summary.purpose,
            key_concepts=summary.key_concepts,
            file_tree=summary.file_tree,
            entry_points=summary.entry_points,
            exploration_summary=exploration_summary,
        )
