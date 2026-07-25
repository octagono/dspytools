"""dspytools export — package compiled programs for sharing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dspytools.cli.output import console, header, info, ok, panel, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click


@click.group(name="export", cls=LLMGroup)
def export_cmd():
    """Export compiled programs in portable formats."""


@export_cmd.command(name="package", cls=LLMCommand)
@click.argument("run_id")
@click.option(
    "--format", "-f", "fmt", default="zip", type=click.Choice(["zip", "tar", "dir"])
)
@click.option("--output", "-o", default=None, help="Output path")
def export_package(run_id: str, fmt: str, output: str | None):
    """Package a compiled program for sharing.

    Creates a portable package containing:
    - program.json (compiled state)
    - metadata.json (lineage, cost, score)
    - SKILL.md template (Agent Skills format)

    Example:
        dspytools export package abc123 --format zip
    """
    from dspytools.config.settings import compiled_dir
    from dspytools.core.registry import get_lineage, get_run

    meta = get_run(run_id)
    if not meta:
        click.echo(f"Run '{run_id}' not found", err=True)
        raise click.Abort()

    src_dir = compiled_dir() / run_id
    if not src_dir.exists():
        click.echo(f"No compiled program found for '{run_id}'", err=True)
        raise click.Abort()

    output_dir = Path(output) if output else Path(f"export_{run_id}")

    # Copy program files
    if fmt == "dir":
        output_dir.mkdir(parents=True, exist_ok=True)
        for f in src_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
    elif fmt == "zip":
        output_path = Path(str(output_dir) + ".zip")
        shutil.make_archive(str(output_dir), "zip", src_dir)
        output_dir = output_path
    elif fmt == "tar":
        output_path = Path(str(output_dir) + ".tar.gz")
        shutil.make_archive(str(output_dir.with_suffix("")), "gztar", src_dir)
        output_dir = output_path

    # Write metadata
    meta_path = output_dir / "metadata.json" if fmt == "dir" else None
    if meta_path:
        meta_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "metadata": meta,
                    "lineage": [
                        entry.get("lineage", {}) for entry in get_lineage(run_id)
                    ],
                },
                indent=2,
                default=str,
            )
        )

    panel(
        "Program Exported",
        f"[bold]Run ID:[/] {run_id}\n"
        f"[bold]Output:[/] {output_dir}\n"
        f"[bold]Format:[/] {fmt}",
        border_style="green",
    )
    ok(f"Exported \u2192 {output_dir}")


@export_cmd.command(name="info", cls=LLMCommand)
@click.argument("run_id")
def export_info(run_id: str):
    """Show export-ready metadata for a compiled program."""
    from dspytools.config.settings import compiled_dir
    from dspytools.core.registry import get_lineage, get_run

    meta = get_run(run_id)
    if not meta:
        click.echo(f"Run '{run_id}' not found", err=True)
        return

    lineage = get_lineage(run_id)
    src_dir = compiled_dir() / run_id

    console.print(f"\n[bold]Program:[/] {run_id}")
    console.print(f"[bold]Optimizer:[/] {meta.get('optimizer', '?')}")
    console.print(f"[bold]Module:[/] {meta.get('module', '?')}")
    console.print(f"[bold]Score:[/] {meta.get('score', '?')}")
    console.print(f"[bold]Lineage Depth:[/] {len(lineage)}")
    console.print(
        f"[bold]Files:[/] {list(src_dir.glob('*')) if src_dir.exists() else 'N/A'}"
    )

    if lineage:
        console.print("\n[bold]Lineage Chain:[/]")
        for entry in lineage:
            opt = entry.get("lineage", {}).get("optimizer", "?")
            score = entry.get("score", "?")
            console.print(f"  {opt} \u2192 score {score}")


@export_cmd.command(name="list", cls=LLMCommand)
def export_list():
    """List compiled programs available for export."""
    from dspytools.core.registry import list_compiled_runs

    runs = list_compiled_runs()
    if not runs:
        info("No compiled programs found")
        return

    rows = []
    for r in runs:
        run_id = r.get("id", "?")[:20]
        optimizer = r.get("optimizer", "?")
        module = r.get("module", "?")
        score = r.get("score", "N/A")
        if score is not None and score != "N/A":
            score = f"{float(score):.3f}"
        created = r.get("created", "")[:16] if r.get("created") else ""
        rows.append([run_id, optimizer, module, str(score)[:8], created])

    header("Compiled Programs Available for Export")
    table(
        "Exportable Programs",
        ["Run ID", "Optimizer", "Module", "Score", "Created"],
        rows,
    )
