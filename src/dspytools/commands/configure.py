"""dspytools configure — Manage LM configuration and API keys."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.env import get_key, list_keys, set_key
from dspytools.config.settings import load_config, save_user_config


@click.group(name="configure", cls=LLMGroup)
def configure_cmd():
    """Manage API keys and LM configuration."""


@configure_cmd.group(name="key")
def key_cmd():
    """Manage API keys."""


@key_cmd.command(name="set", cls=LLMCommand)
@click.argument("provider")
@click.argument("key", required=False)
@click.option("--stdin", is_flag=True, help="Read key from stdin (pipe-safe)")
def key_set(provider: str, key: str | None = None, stdin: bool = False):
    """Set an API key for a provider (e.g., openai, deepseek, anthropic).

    If KEY is not provided, prompts interactively with hidden input.
    Use --stdin to pipe the key: echo $KEY | configure key set deepseek --stdin
    """
    if stdin:
        key = sys.stdin.read().strip()
    elif key is None:
        key = click.prompt(f"{provider.upper()}_API_KEY", hide_input=True)
    if not key:
        raise click.ClickException("API key cannot be empty")
    set_key(provider, key)
    click.echo(f"  Set {provider.upper()}_API_KEY")


@key_cmd.command(name="list", cls=LLMCommand)
def key_list():
    """List all configured API keys (values masked)."""
    keys = list_keys()
    if keys:
        for provider, masked in keys.items():
            click.echo(f"  {provider}: {masked}")
    else:
        click.echo("  No API keys configured. Use `dspytools configure key set`")


@key_cmd.command(name="get", cls=LLMCommand)
@click.argument("provider")
def key_get(provider: str):
    """Show an API key (masked)."""
    key = get_key(provider)
    if key:
        masked = key[:8] + "..." if len(key) > 8 else "***"
        click.echo(f"  {provider}: {masked}")
    else:
        click.echo(f"  No key configured for {provider}")


@configure_cmd.group(name="lm")
def lm_cmd():
    """Manage language models."""


@lm_cmd.command(name="set", cls=LLMCommand)
@click.argument("model")
@click.option("--api-base", help="API base URL (e.g., http://localhost:8000/v1)")
@click.option("--provider", help="Provider prefix (e.g., openai, deepseek)")
@click.option(
    "--role",
    type=click.Choice(["student", "teacher", "default"]),
    default="default",
    help="LM role",
)
def lm_set(model: str, api_base: str | None, provider: str | None, role: str):
    """Set a language model for a specific role.

    Roles:
      student  - Model used for inference (e.g., Qwen local LLM)
      teacher  - Model used for optimization/reflection (e.g., DeepSeek)
      default  - Default fallback model
    """
    cfg = load_config()
    cfg.setdefault("lm", {})
    cfg["lm"].setdefault("registry", {})

    entry = {"model": model}
    if api_base:
        entry["api_base"] = api_base
    if provider:
        entry["provider"] = provider
    elif model and "/" in model:
        entry["provider"] = model.split("/")[0]

    if role == "default":
        cfg["lm"]["default"] = model
        cfg["lm"]["registry"][model] = entry
    elif role == "student":
        cfg["lm"]["student"] = entry
    elif role == "teacher":
        cfg["lm"]["teacher"] = entry

    save_user_config(cfg)
    click.echo(f"  {role.capitalize()} LM set to: {model}")


@lm_cmd.command(name="list", cls=LLMCommand)
def lm_list():
    """List configured language models.

    Resolution order for inference: student → default → openai/gpt-4o.
    Teacher is used only for optimization/reflection (GEPA, distill).
    """
    cfg = load_config()
    student = cfg.get("lm", {}).get("student")
    teacher = cfg.get("lm", {}).get("teacher")

    if student:
        click.echo(f"  Student: {student.get('model', '?')}")
        for k in ("api_base", "provider", "temperature", "max_tokens"):
            v = student.get(k)
            if v:
                click.echo(f"    {k}: {v}")
    else:
        click.echo(
            "  Student: not set (set via `dspytools configure lm set --role student`)"
        )

    if teacher:
        click.echo(f"  Teacher: {teacher.get('model', '?')}")
        for k in ("api_base", "provider", "temperature", "max_tokens"):
            v = teacher.get(k)
            if v:
                click.echo(f"    {k}: {v}")
        if student and teacher.get("model") == student.get("model"):
            click.echo("    ⚠ same as student — optimization needs a stronger teacher")
    else:
        click.echo(
            "  Teacher: not set (set via `dspytools configure lm set --role teacher`)"
        )


@lm_cmd.command(name="get", cls=LLMCommand)
@click.option(
    "--role",
    type=click.Choice(["student", "teacher", "default"]),
    default="student",
    help="Which model to show",
)
def lm_get(role: str):
    """Show current model for a specific role."""
    cfg = load_config()
    role_key = role if role in ("student", "teacher") else "default"
    entry = cfg.get("lm", {}).get(role_key)

    if role == "default":
        entry = cfg.get("lm", {}).get("default")
        click.echo(f"  Default: {entry or 'not set'}")
    elif entry:
        click.echo(f"  {role.capitalize()}: {entry.get('model', '?')}")
        if entry.get("api_base"):
            click.echo(f"    api_base: {entry['api_base']}")
    else:
        click.echo(f"  No {role} LM configured")


@configure_cmd.group(name="adapter")
def adapter_cmd():
    """Manage DSPy adapters."""


@adapter_cmd.command(name="set", cls=LLMCommand)
@click.argument("type", type=click.Choice(["chat", "json", "xml", "baml", "twostep"]))
def adapter_set(type: str):
    """Set the default DSPy adapter type."""
    cfg = load_config()
    cfg.setdefault("dspy", {})
    cfg["dspy"]["adapter"] = type
    save_user_config(cfg)
    click.echo(f"  Adapter set to: {type}")


@adapter_cmd.command(name="list", cls=LLMCommand)
def adapter_list():
    """List available adapter types."""
    for name, desc in {
        "chat": "ChatAdapter — default, [[ ## field ## ]] delimiters",
        "json": "JSONAdapter — JSON input/output",
        "xml": "XMLAdapter — XML-tagged prompts",
        "baml": "BAMLAdapter — compact nested Pydantic schema",
        "twostep": "TwoStepAdapter — plan then generate",
    }.items():
        click.echo(f"  {name}: {desc}")


@configure_cmd.group(name="cache")
def cache_cmd():
    """Manage DSPy cache."""


@cache_cmd.command(name="enable", cls=LLMCommand)
def cache_enable():
    """Enable DSPy cache with secure pickle."""
    from dspytools.core._dspy import dspy

    dspy.configure_cache(restrict_pickle=True)
    click.echo("  Cache enabled (restricted pickle)")


@cache_cmd.command(name="disable", cls=LLMCommand)
def cache_disable():
    """Disable DSPy cache."""
    from dspytools.core._dspy import dspy

    dspy.configure_cache(restrict_pickle=False)
    click.echo("  Cache disabled")


@cache_cmd.command(name="clear", cls=LLMCommand)
def cache_clear():
    """Clear the DSPy cache directory."""
    from dspytools.config.settings import cache_dir

    cdir = cache_dir()
    if cdir.exists():
        shutil.rmtree(cdir)
        click.echo(f"  Cache cleared: {cdir}")
    else:
        click.echo("  No cache found")


@configure_cmd.group(name="dspy")
def dspy_cmd():
    """Configure DSPy runtime settings."""


@dspy_cmd.command(name="set", cls=LLMCommand)
@click.option("--track-usage/--no-track-usage", default=None, help="Track token usage")
@click.option("--async-workers", type=int, help="Max async workers (default: 8)")
@click.option("--num-threads", type=int, help="Thread count for Parallel (default: 8)")
@click.option(
    "--max-errors", type=int, help="Stop parallel after N errors (default: 10)"
)
@click.option(
    "--disable-history/--enable-history", default=None, help="Disable LM call history"
)
@click.option(
    "--max-history-size", type=int, help="Max history entries (default: 10000)"
)
@click.option(
    "--allow-async-sync/--no-async-sync",
    default=None,
    help="Allow async→sync tool conversion",
)
@click.option(
    "--provide-traceback/--no-traceback",
    default=None,
    help="Include tracebacks in errors",
)
@click.option("--warn-type/--no-warn-type", default=None, help="Warn on type mismatch")
@click.option("--apply", is_flag=True, help="Apply configuration immediately")
def dspy_set(
    track_usage: bool | None,
    async_workers: int | None,
    num_threads: int | None,
    max_errors: int | None,
    disable_history: bool | None,
    max_history_size: int | None,
    allow_async_sync: bool | None,
    provide_traceback: bool | None,
    warn_type: bool | None,
    apply: bool,
):
    """Set DSPy runtime configuration settings."""
    cfg = load_config()
    cfg.setdefault("dspy", {})

    settings = {}
    if track_usage is not None:
        settings["track_usage"] = track_usage
    if async_workers is not None:
        settings["async_max_workers"] = async_workers
    if num_threads is not None:
        settings["num_threads"] = num_threads
    if max_errors is not None:
        settings["max_errors"] = max_errors
    if disable_history is not None:
        settings["disable_history"] = disable_history
    if max_history_size is not None:
        settings["max_history_size"] = max_history_size
    if allow_async_sync is not None:
        settings["allow_tool_async_sync_conversion"] = allow_async_sync
    if provide_traceback is not None:
        settings["provide_traceback"] = provide_traceback
    if warn_type is not None:
        settings["warn_on_type_mismatch"] = warn_type

    cfg["dspy"].update(settings)
    save_user_config(cfg)

    for k, v in settings.items():
        click.echo(f"  {k}: {v}")

    if apply:
        from dspytools.core._dspy import dspy

        dspy.configure(**settings)
        click.echo("  Applied to runtime")
    else:
        click.echo("  Saved to config (use --apply to activate now)")


@dspy_cmd.command(name="show", cls=LLMCommand)
def dspy_show():
    """Show current DSPy runtime configuration."""
    cfg = load_config()
    dspy_cfg = cfg.get("dspy", {})

    defaults = {
        "track_usage": False,
        "async_max_workers": 8,
        "num_threads": 8,
        "max_errors": 10,
        "disable_history": False,
        "max_history_size": 10000,
        "allow_tool_async_sync_conversion": False,
        "provide_traceback": False,
        "warn_on_type_mismatch": False,
    }

    for k, default in defaults.items():
        current = dspy_cfg.get(k, default)
        marker = " (custom)" if k in dspy_cfg else ""
        click.echo(f"  {k}: {current}{marker}")


@dspy_cmd.command(name="optimize", cls=LLMCommand)
def dspy_optimize():
    """Apply optimal DSPy configuration for this environment."""
    from dspytools.core._dspy import dspy

    env = {
        "production": {
            "track_usage": True,
            "async_max_workers": 4,
            "max_errors": 5,
            "disable_history": True,
            "warn_on_type_mismatch": False,
            "provide_traceback": False,
        },
        "development": {
            "track_usage": True,
            "async_max_workers": 8,
            "max_errors": 10,
            "disable_history": False,
            "warn_on_type_mismatch": True,
            "provide_traceback": True,
        },
        "optimization": {
            "track_usage": False,
            "async_max_workers": 16,
            "num_threads": 16,
            "max_errors": 20,
            "disable_history": True,
            "warn_on_type_mismatch": False,
            "provide_traceback": False,
        },
    }

    # Auto-detect: if teacher LM configured, use "optimization"
    cfg = load_config()
    teacher = cfg.get("lm", {}).get("teacher")
    if teacher:
        profile = "optimization"
    else:
        profile = "development"

    settings = env[profile]
    dspy.configure(**settings)

    cfg = load_config()
    cfg.setdefault("dspy", {})
    cfg["dspy"].update(settings)
    cfg["dspy"]["profile"] = profile
    save_user_config(cfg)

    click.echo(f"  Profile: {profile}")
    for k, v in settings.items():
        click.echo(f"  {k}: {v}")
    click.echo("  Applied to runtime")


# ── Shell Completion ─────────────────────────────────────────────────────


_COMPLETION_MARKER = "# >>> dspytools completion >>>"
_COMPLETION_MARKER_END = "# <<< dspytools completion <<<"


def _detect_shell() -> str:
    """Auto-detect shell from $SHELL or $0."""

    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    if "fish" in shell:
        return "fish"
    return "bash"


def _generate_completion_script(shell: str) -> str:
    """Generate the completion script for the given shell.

    For zsh, generates a custom script that uses compadd -d instead of
    _describe to avoid the grouped preview box. Shows commands with their
    help text as inline descriptions only.
    """

    env = os.environ.copy()
    env["_DSPYTOOLS_COMPLETE"] = f"{shell}_source"

    # Use the dspytools console script if available, fall back to python -c
    result = subprocess.run(
        ["dspytools"],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Fallback: invoke via python -c
        result = subprocess.run(
            [sys.executable, "-c", "from dspytools.main import cli; cli()"],
            env=env,
            capture_output=True,
            text=True,
        )

    script = result.stdout.strip()

    # For zsh: replace Click's _describe with compadd -d to avoid preview box
    if shell == "zsh":
        script = _zsh_compact_script()

    return script


def _zsh_compact_script() -> str:
    """Generate a compact zsh completion script without _describe preview box.

    Uses compadd -d to show commands with their help text as inline
    descriptions, without the grouped _describe formatting that creates
    a preview box in some zsh/fzf-tab configurations.
    """
    return """#compdef dspytools

_dspytools_completion() {
    (( ! $+commands[dspytools] )) && return 1

    local -a cmds desc
    local type key descr

    while IFS= read -r type; do
        IFS= read -r key
        IFS= read -r descr
        if [[ "$type" == "plain" ]]; then
            cmds+=("$key")
            [[ "$descr" == "_" ]] && descr=""
            desc+=("$descr")
        elif [[ "$type" == "dir" ]]; then
            _path_files -/
            return
        elif [[ "$type" == "file" ]]; then
            _path_files -f
            return
        fi
    done < <(env COMP_WORDS="${words[*]}" COMP_CWORD=$((CURRENT-1)) _DSPYTOOLS_COMPLETE=zsh_complete dspytools)

    [[ ${#cmds[@]} -gt 0 ]] && compadd -d desc -- "${cmds[@]}"
}

if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
    _dspytools_completion "$@"
else
    compdef _dspytools_completion dspytools
fi"""


def _get_shell_config_path(shell: str):
    """Return the config file path for the given shell."""

    home = Path.home()

    if shell == "bash":
        return home / ".bashrc"
    elif shell == "zsh":
        return home / ".zshrc"
    elif shell == "fish":
        d = home / ".config" / "fish" / "completions"
        d.mkdir(parents=True, exist_ok=True)
        return d / "dspytools.fish"
    else:
        raise ValueError(f"unsupported shell: {shell}")


def _wrap_script(shell: str, script: str) -> str:
    """Wrap the completion script with marker comments for easy removal."""
    return f"{_COMPLETION_MARKER}\n{script}\n{_COMPLETION_MARKER_END}"


@configure_cmd.group(name="completion")
def completion_cmd():
    """Shell completion installation and management."""


@completion_cmd.command(name="install", cls=LLMCommand)
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=None,
    help="Shell type (auto-detected if not specified)",
)
def completion_install(shell: str | None):
    """Install shell completion for dspytools.

    Auto-detects your shell from $SHELL. Writes the completion script
    to the appropriate config file (~/.bashrc, ~/.zshrc, or
    ~/.config/fish/completions/dspytools.fish).

    Restart your shell or source the config file after installation.
    """
    detected = shell or _detect_shell()
    click.echo(f"  Shell: {detected}")

    script = _generate_completion_script(detected)
    if not script:
        click.echo("  ✗ Failed to generate completion script")
        raise SystemExit(1)

    config_path = _get_shell_config_path(detected)

    if detected == "fish":
        # Fish uses a standalone file, not sourced from config
        config_path.write_text(script + "\n")
        click.echo(f"  ✓ Written to {config_path}")
    else:
        # Bash/zsh: append marked block to config file
        wrapped = _wrap_script(detected, script)

        existing = ""
        if config_path.exists():
            existing = config_path.read_text()

        # Remove old block if present
        if _COMPLETION_MARKER in existing:
            pattern = (
                re.escape(_COMPLETION_MARKER)
                + r".*?"
                + re.escape(_COMPLETION_MARKER_END)
                + r"\n?"
            )
            existing = re.sub(pattern, "", existing, flags=re.DOTALL)

        config_path.write_text(existing.rstrip("\n") + "\n\n" + wrapped + "\n")
        click.echo(f"  ✓ Added to {config_path}")

    click.echo(f"\n  Restart your shell or run: source {config_path}")


@completion_cmd.command(name="uninstall", cls=LLMCommand)
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=None,
    help="Shell type (auto-detected if not specified)",
)
def completion_uninstall(shell: str | None):
    """Remove shell completion for dspytools."""
    detected = shell or _detect_shell()
    config_path = _get_shell_config_path(detected)

    if detected == "fish":
        if config_path.exists():
            config_path.unlink()
            click.echo(f"  ✓ Removed {config_path}")
        else:
            click.echo("  Completion not installed (file not found)")
        return

    # Bash/zsh: remove marked block
    if not config_path.exists():
        click.echo("  Completion not installed (config file not found)")
        return

    existing = config_path.read_text()
    if _COMPLETION_MARKER not in existing:
        click.echo("  Completion not installed (marker not found)")
        return

    pattern = (
        re.escape(_COMPLETION_MARKER)
        + r".*?"
        + re.escape(_COMPLETION_MARKER_END)
        + r"\n?"
    )
    cleaned = re.sub(pattern, "", existing, flags=re.DOTALL)
    config_path.write_text(cleaned.rstrip("\n") + "\n")
    click.echo(f"  ✓ Removed from {config_path}")


@completion_cmd.command(name="status", cls=LLMCommand)
def completion_status():
    """Show shell completion installation status."""
    detected = _detect_shell()
    config_path = _get_shell_config_path(detected)

    click.echo(f"  Detected shell: {detected}")
    click.echo(f"  Config file: {config_path}")

    if detected == "fish":
        installed = config_path.exists()
    else:
        installed = (
            config_path.exists() and _COMPLETION_MARKER in config_path.read_text()
        )

    if installed:
        click.echo("  Status: ✓ installed")
    else:
        click.echo("  Status: ✗ not installed")
        click.echo("\n  Run: dspytools configure completion install")


@completion_cmd.command(name="show", cls=LLMCommand)
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"]),
    default=None,
    help="Shell type (auto-detected if not specified)",
)
def completion_show(shell: str | None):
    """Print the completion script for manual sourcing.

    Usage:
        eval "$(dspytools configure completion show)"
    """
    detected = shell or _detect_shell()
    script = _generate_completion_script(detected)
    click.echo(script)
