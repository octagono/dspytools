"""CLI introspection — builds trainset from the dspytools command tree."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy


def get_all_commands(cli: Any) -> dict[str, dict]:
    """Introspect the click CLI group and return all commands with metadata."""
    commands = {}

    for cmd_name in sorted(cli.list_commands(None)):
        if cmd_name == "self":
            continue
        cmd = cli.get_command(None, cmd_name)
        if cmd is None:
            continue

        info = {
            "name": cmd_name,
            "description": cmd.help or "",
            "subcommands": [],
            "options": [],
            "examples": [],
            "help_text": _build_help_text(cmd_name, cmd),
        }

        # Get subcommands — handle both normal and lazy groups
        sub_commands: dict[str, Any] = {}
        if hasattr(cmd, "list_commands"):
            for sub_name in cmd.list_commands(None):
                sub_cmd = cmd.get_command(None, sub_name)
                if sub_cmd:
                    sub_commands[sub_name] = sub_cmd
        elif hasattr(cmd, "commands"):
            sub_commands = cmd.commands or {}

        for sub_name, sub_cmd in sub_commands.items():
            sub_info = {
                "name": sub_name,
                "description": sub_cmd.help or "",
                "options": [],
            }
            for param in sub_cmd.params or []:
                if hasattr(param, "opts") and param.opts and hasattr(param, "help"):
                    sub_info["options"].append(
                        {
                            "name": param.opts[0],
                            "help": getattr(param, "help", "") or "",
                            "type": str(getattr(param, "type", "str")),
                        }
                    )
            info["subcommands"].append(sub_info)

        # Get top-level options
        for param in cmd.params or []:
            if hasattr(param, "opts") and param.opts and hasattr(param, "help"):
                info["options"].append(
                    {
                        "name": param.opts[0],
                        "help": getattr(param, "help", "") or "",
                    }
                )

        info["examples"] = _build_examples(cmd_name, info["subcommands"])
        commands[cmd_name] = info

    return commands


def build_trainset_from_cli(cli: Any) -> list[dspy.Example]:
    """Build a DSPy trainset by introspecting the CLI tree."""
    commands = get_all_commands(cli)
    trainset = []

    for cmd_name, info in commands.items():
        subcommands_str = _format_subcommands(info["subcommands"])
        examples_str = "\n".join(info["examples"])
        answer = info["help_text"]

        trainset.append(
            dspy.Example(
                command=f"dspytools {cmd_name}",
                subcommands=subcommands_str,
                examples=examples_str,
                answer=answer,
            ).with_inputs("command", "subcommands", "examples")
        )

        for sub in info["subcommands"]:
            sub_examples = _build_sub_examples(cmd_name, sub["name"])
            trainset.append(
                dspy.Example(
                    command=f"dspytools {cmd_name} {sub['name']}",
                    subcommands=_format_subcommands(info["subcommands"]),
                    examples="\n".join(sub_examples),
                    answer=f"Command: dspytools {cmd_name} {sub['name']}\n\n"
                    f"{sub['description']}\n\n"
                    f"Options: {', '.join(o['name'] for o in sub['options'])}",
                ).with_inputs("command", "subcommands", "examples")
            )

    return trainset


def _format_subcommands(subcommands: list[dict]) -> str:
    lines = []
    for sub in subcommands:
        opts = ", ".join(o["name"] for o in sub.get("options", []))
        lines.append(f"  {sub['name']}: {sub['description']}")
        if opts:
            lines.append(f"    Options: {opts}")
    return "\n".join(lines)


def _build_help_text(cmd_name: str, cmd: Any) -> str:
    sub_list = []
    if hasattr(cmd, "commands"):
        for sub_name, sub_cmd in cmd.commands.items():
            sub_list.append(f"  {sub_name}: {sub_cmd.help or ''}")

    lines = [f"Command: dspytools {cmd_name}"]
    lines.append(f"\n{cmd.help or ''}")
    if sub_list:
        lines.append("\nSubcommands:")
        lines.extend(sub_list)
    lines.append(f"\nFor detailed help, use: dspytools {cmd_name} <subcommand> --help")
    return "\n".join(lines)


def _build_examples(cmd_name: str, subcommands: list[dict]) -> list[str]:
    examples = []
    if subcommands:
        examples.append(f"dspytools {cmd_name} {subcommands[0]['name']} --help")
        if len(subcommands) > 1:
            examples.append(f"dspytools {cmd_name} {subcommands[-1]['name']} --help")
    examples.append(f"dspytools {cmd_name} --help")
    return examples


def _build_sub_examples(cmd_name: str, sub_name: str) -> list[str]:
    examples = [f"dspytools {cmd_name} {sub_name} --help"]
    task_examples = [
        ("configure", "lm", "dspytools configure lm list"),
        ("compile", "knn", "dspytools compile knn ModuleName trainset.json --k 2"),
        ("run", "predict", 'dspytools run predict "topic->tweet" -i topic=AI'),
        ("tool", "list", "dspytools tool list"),
        ("agent", "new", "dspytools agent new git-helper --max-iters 8"),
    ]
    for t_cmd, t_sub, ex in task_examples:
        if t_cmd == cmd_name and t_sub == sub_name:
            examples.append(ex)
    return examples
