"""Rich-click configuration for dspytools CLI.

Optimal UX/UI: branded theme, categorized panels, consistent styling.
All command files should import click from this module:
    from dspytools.cli.rich_config import click
"""

from __future__ import annotations

import rich_click as click
from rich_click.rich_help_configuration import RichHelpConfiguration

# Re-export LLM-powered classes for command files
from dspytools.cli.llm_help import (  # noqa: F401
    LLMCommand,
    LLMGroup,
    llm_command,
    llm_group,
)

# ── RichHelpConfiguration (dataclass-based, rich-click 1.9+) ────────────────────────────
click.rich_click.CONFIG = RichHelpConfiguration(
    # ── Theme ─────────────────────────────────────────────
    theme="nord-modern",
    # ── Text styling ──────────────────────────────────────
    text_markup="rich",
    text_emojis=False,
    # ── Layout ────────────────────────────────────────────
    max_width=100,
    commands_before_options=True,  # Commands above options for better UX
    default_panels_first=False,
    show_arguments=True,
    group_arguments_options=True,
    # ── Panel styling ─────────────────────────────────────
    style_commands_panel_border="dim",
    style_options_panel_border="dim",
    style_errors_panel_border="red",
    style_errors_suggestion="dim italic",
    style_errors_suggestion_command="bold cyan",
    # ── Panel titles ─────────────────────────────────────
    options_panel_title="Options",
    commands_panel_title="Commands",
    arguments_panel_title="Arguments",
    errors_panel_title="Error",
    # ── Header/footer ─────────────────────────────────────
    header_text="dspytools — DSPy program lifecycle CLI",
    footer_text="Documentation: https://docs.example.com",
    # ── Error messages ────────────────────────────────────
    errors_suggestion="Try 'dspytools COMMAND --help' for more information",
    errors_epilogue=None,
    # ── Help sections ─────────────────────────────────────
    use_click_short_help=False,
    options_table_help_sections=[
        "help",
        "deprecated",
        "envvar",
        "default",
        "required",
        "metavar",
    ],
    # ── Table column types ────────────────────────────────
    options_table_column_types=[
        "required",
        "opt_short",
        "opt_long",
        "metavar",
        "help",
    ],
    commands_table_column_types=[
        "name",
        "help",
    ],
    # ── Table expand ───────────────────────────────────────
    style_options_table_expand=False,
    style_commands_table_expand=False,
    style_options_table_border_style="dim",
    style_commands_table_border_style="dim",
)

# ── Command Groups (root CLI) ──────────────────────────────────────────
click.rich_click.COMMAND_GROUPS = {
    "dspytools": [
        {
            "name": "Programs & Inference",
            "commands": ["run", "compile", "evaluate", "compare", "inspect", "export"],
        },
        {
            "name": "LM & Configuration",
            "commands": ["configure", "lora", "doctor"],
        },
        {
            "name": "Agent & Skills",
            "commands": ["agent", "skills", "tool", "mcp"],
        },
        {
            "name": "Pipeline & Optimization",
            "commands": ["gfl", "self", "pipeline", "distill", "data"],
        },
        {
            "name": "Generation",
            "commands": ["generate", "module", "signature"],
        },
        {
            "name": "Graph & Memory",
            "commands": ["graph", "memory"],
        },
    ],
    "dspytools compile": [
        {
            "name": "Optimizers",
            "commands": [
                "knn",
                "mipro",
                "gepa",
                "copro",
                "simba",
                "bootstrap-few-shot",
                "labeled-few-shot",
                "better-together",
                "ensemble",
                "finetune",
                "grpo",
            ],
        },
        {
            "name": "Async & Management",
            "commands": ["submit", "status", "list", "cancel", "cost", "gfl"],
        },
    ],
    "dspytools gfl": [
        {
            "name": "Comparison",
            "commands": ["spin", "lse", "gepa", "meta-optimize"],
        },
        {
            "name": "Data & Analysis",
            "commands": ["synthesize", "consolidate", "decompose", "ab-test"],
        },
    ],
    "dspytools skills": [
        {
            "name": "Lifecycle",
            "commands": ["create", "compile", "optimize", "list", "search"],
        },
        {
            "name": "Ecosystem",
            "commands": ["find", "discover", "categories"],
        },
    ],
    "dspytools graph": [
        {
            "name": "Graph Operations",
            "commands": ["status", "query", "skill-tree", "program-lineage", "stats"],
        },
        {
            "name": "Management",
            "commands": ["add-dependency", "dependents", "record-program", "search"],
        },
        {
            "name": "Redis Cache",
            "commands": ["redis"],
        },
    ],
    "dspytools self": [
        {
            "name": "Evolution",
            "commands": ["optimize", "ask", "status"],
        },
    ],
    "dspytools configure": [
        {
            "name": "Shell Completion",
            "commands": ["completion"],
        },
    ],
    "dspytools configure completion": [
        {
            "name": "Completion",
            "commands": ["install", "uninstall", "status", "show"],
        },
    ],
    "dspytools lora": [
        {
            "name": "Adapter Management",
            "commands": ["load", "unload", "list", "chat", "test"],
        },
        {
            "name": "Discovery & Health",
            "commands": ["discover", "health", "extract", "evaluate", "train"],
        },
    ],
    "dspytools agent": [
        {
            "name": "Agent Management",
            "commands": ["create", "list", "run"],
        },
    ],
    "dspytools memory": [
        {
            "name": "Memory Operations",
            "commands": ["add", "search", "list", "get", "update", "delete"],
        },
        {
            "name": "Management",
            "commands": ["stats", "reset", "history"],
        },
    ],
    "dspytools generate": [
        {
            "name": "Generation",
            "commands": ["llms-txt", "batch", "explore"],
        },
    ],
    "dspytools distill": [
        {
            "name": "LoRA Distillation",
            "commands": ["run", "list-frameworks", "stats", "prepare-colab"],
        },
    ],
}

# ── Option Groups (per subcommand) ─────────────────────────────────────
click.rich_click.OPTION_GROUPS = {
    "dspytools compile knn": [
        {"name": "Required", "options": ["--program", "--trainset"]},
        {"name": "Optional", "options": ["--label", "--train-field", "--val-field"]},
    ],
    "dspytools compile mipro": [
        {"name": "Required", "options": ["--program", "--trainset"]},
        {"name": "Optional", "options": ["--label", "--train-field", "--val-field"]},
    ],
    "dspytools compile gfl": [
        {"name": "Required", "options": ["--program", "--trainset"]},
        {"name": "Pipeline", "options": ["--single", "--halving", "--draft"]},
        {"name": "Optional", "options": ["--label", "--train-field", "--val-field"]},
    ],
    "dspytools run hot": [
        {"name": "Required", "options": ["--program"]},
        {"name": "Optional", "options": ["--input"]},
    ],
    "dspytools skills compile": [
        {"name": "Required", "options": ["--name"]},
        {"name": "Optional", "options": ["--trainset", "--optimizer"]},
    ],
    "dspytools skills optimize": [
        {"name": "Required", "options": ["--name"]},
    ],
    "dspytools graph query": [
        {"name": "Required", "options": ["--query"]},
    ],
    "dspytools graph redis set": [
        {"name": "Required", "options": ["--key", "--value"]},
        {"name": "Optional", "options": ["--ttl"]},
    ],
    "dspytools memory add": [
        {"name": "Required", "options": ["--content"]},
        {"name": "Optional", "options": ["--user-id", "--agent-id", "--run-id"]},
    ],
    "dspytools memory search": [
        {"name": "Required", "options": ["--query"]},
        {"name": "Optional", "options": ["--user-id", "--limit"]},
    ],
}
