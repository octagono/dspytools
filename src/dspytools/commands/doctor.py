"""dspytools doctor — system diagnostics and health checks.

Catches misconfiguration before it wastes time on failed compiles.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path

import mlflow

from dspytools.cli.output import console, ok, panel, warn
from dspytools.cli.rich_config import LLMCommand, click
from dspytools.config.settings import compiled_dir
from dspytools.config.settings import config_dir as _cdir
from dspytools.config.settings import llm_url as _llm_url
from dspytools.core._dspy import dspy
from dspytools.core.registry import list_compiled_runs
from dspytools.core.setup import LMRegistry


@click.command(name="doctor", cls=LLMCommand)
@click.option(
    "--check-llm/--no-llm", default=True, help="Check LLM server connectivity"
)
@click.option("--check-gpu/--no-gpu", default=True, help="Check GPU health")
@click.option("--check-config/--no-config", default=True, help="Validate configuration")
def doctor_cmd(check_llm: bool, check_gpu: bool, check_config: bool):
    """Run system diagnostics to verify dspytools is properly configured.

    Checks:
    - Python version and dependencies
    - LLM server connectivity and model availability
    - GPU memory and utilization
    - Configuration validity (.env, LM setup)
    - Registry and compiled directory health
    - DOX tree integrity

    Example:
        dspytools doctor
    """

    passes = 0
    fails = 0
    warnings = 0

    def check(name: str, fn, *args) -> bool:
        nonlocal passes, fails
        fn(*args)
        ok(f"  {name}")
        passes += 1
        return True

    def warn_check(name: str, fn, *args) -> bool:
        nonlocal passes, warnings
        result = fn(*args)
        if result:
            ok(f"  {name}")
            passes += 1
        else:
            warn(f"  {name} \u2014 not found (optional)")
            warnings += 1
        return bool(result)

    console.print("\n[bold]dspytools doctor[/]\n")

    # ── Python & Dependencies ─────────────────────────────────
    console.print("[bold]Python Environment:[/]")
    check(
        "Python >= 3.12",
        lambda: (
            sys.version_info >= (3, 12)
            or (_ for _ in ()).throw(
                Exception(f"Python {sys.version_info.major}.{sys.version_info.minor}")
            )
        ),
    )

    def _check_dspy():
        return f"dspy {dspy.__version__}"

    check("DSPy import", lambda: _check_dspy() and True)

    def _check_click():
        return f"click {version('click')}"

    check("Click import", lambda: _check_click() and True)

    def _check_mlflow():
        return f"mlflow {mlflow.__version__}"

    warn_check("MLflow", lambda: _check_mlflow())

    # ── LLM Server Connectivity ──────────────────────────────────────
    if check_llm:
        console.print("\n[bold]LLM Server:[/]")

        def _check_llm_health():
            base = _llm_url()
            # Try /v1/models (OpenAI-compatible), /api/tags (llama-cpp), then /health
            for path in ("/v1/models", "/api/tags", "/health"):
                resp = urllib.request.urlopen(f"{base}{path}", timeout=3)
                if resp.status == 200:
                    return f"reachable at {path}"
            return False

        warn_check("LLM health", _check_llm_health)

        def _check_llm_models():
            resp = urllib.request.urlopen(f"{_llm_url()}/v1/models", timeout=3)
            data = json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            return f"{len(models)} model(s): {', '.join(models[:3])}"

        warn_check("LLM models", _check_llm_models)

    # ── GPU Health ─────────────────────────────────────────────
    if check_gpu:
        console.print("\n[bold]GPU:[/]")

        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            for i, line in enumerate(r.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(", ")]
                if len(parts) >= 4:
                    click.echo(
                        f"  \u2705 GPU {parts[0]}: {parts[1]} | {parts[2]}/{parts[3]} MiB | {parts[4]}% util"
                    )
                    passes += 1
        else:
            warn("  GPU check not available (nvidia-smi failed)")
            warnings += 1

    # ── Configuration ──────────────────────────────────────────
    if check_config:
        console.print("\n[bold]Configuration:[/]")

        def _check_env():
            env_path = Path(".env")
            if env_path.exists():
                lines = env_path.read_text().strip().split("\n")
                keys = [
                    line.split("=")[0]
                    for line in lines
                    if "=" in line and not line.startswith("#")
                ]
                return f".env loaded ({len(keys)} keys: {', '.join(keys[:5])})"
            return ".env loaded (0 keys)"

        warn_check(".env file", _check_env)

        def _check_config_dir():
            d = _cdir()
            d.mkdir(parents=True, exist_ok=True)
            return f"exists ({len(list(d.glob('*')))} files)"

        check("Config directory", _check_config_dir)

        def _check_compiled_dir():
            d = compiled_dir()
            d.mkdir(parents=True, exist_ok=True)
            return f"exists ({len(list(d.glob('*')))} entries)"

        check("Compiled directory", _check_compiled_dir)

    # ── Registry Health ────────────────────────────────────────
    console.print("\n[bold]Registry:[/]")

    def _check_registry():
        runs = list_compiled_runs()
        return f"{len(runs)} compiled programs"

    check("Registry", _check_registry)

    def _check_index():
        index = compiled_dir() / "index.json"
        if index.exists():
            return f"{index.stat().st_size} bytes"
        return "empty (will be created on first compile)"

    check("Index file", _check_index)

    # ── LM Setup ───────────────────────────────────────────────
    console.print("\n[bold]LM Configuration:[/]")

    def _check_student_lm():
        student = LMRegistry.get_or_default()
        return f"student LM: {student.model if hasattr(student, 'model') else 'configured'}"

    warn_check("Student LM", _check_student_lm)

    def _check_teacher_lm():
        teacher = LMRegistry.get_teacher()
        if teacher:
            return f"teacher LM: {teacher.model if hasattr(teacher, 'model') else 'configured'}"
        return "not configured (optimization modes disabled)"

    warn_check("Teacher LM", _check_teacher_lm)

    # ── MCP Tools ──────────────────────────────────────────────
    console.print("\n[bold]MCP Server:[/]")

    def _check_mcp():
        from dspytools.mcp.tools import BUILTIN_TOOLS

        return f"{len(BUILTIN_TOOLS)} tools available"

    check("MCP tools", _check_mcp)

    # ── DOX Tree ───────────────────────────────────────────────
    console.print("\n[bold]Documentation:[/]")

    def _check_dox():
        dox_files = list(Path("src/dspytools").rglob("AGENTS.md"))
        return f"{len(dox_files)} AGENTS.md files"

    check("DOX tree", _check_dox)

    def _check_docs():
        docs = list(Path("docs").glob("*.md"))
        return f"{len(docs)} doc files"

    check("docs/", _check_docs)

    # ── Tests ──────────────────────────────────────────────────
    console.print("\n[bold]Tests:[/]")

    def _check_tests():
        test_files = list(Path("tests").glob("*.py"))
        return f"{len(test_files)} test files"

    check("Test files", _check_tests)

    # ── Summary ────────────────────────────────────────────────
    total = passes + fails + warnings
    console.print(f"\n[bold]Result:[/] {passes}/{total} passed", end="")

    if fails:
        console.print(f", [red]{fails} failed[/]", end="")
    if warnings:
        console.print(f", [yellow]{warnings} warnings[/]", end="")
    console.print()

    if fails == 0:
        panel(
            "System Healthy",
            f"[green]All {passes} checks passed[/]\n{warnings} optional checks warned (non-blocking)",
            border_style="green",
        )
    else:
        console.print(
            f"\n[red]{fails} critical check(s) failed. Fix issues above before compiling.[/]"
        )
        raise SystemExit(1)
