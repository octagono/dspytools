"""dspytools evaluate — Evaluate DSPy programs."""

from __future__ import annotations

import importlib.util as util
import sys as _sys
from pathlib import Path

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from dspytools.cli.output import console, error, panel, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core.loaders import load_trainset
from dspytools.core.logging_config import get_logger
from dspytools.core.setup import setup_dspy

_log = get_logger(__name__)


@click.group(name="evaluate", cls=LLMGroup)
def evaluate_cmd():
    """Evaluate DSPy programs on datasets."""


@evaluate_cmd.command(name="run", cls=LLMCommand)
@click.argument("module")
@click.argument("devset")
@click.option(
    "--metric",
    default="exact_match",
    type=click.Choice(["exact_match", "passage_match", "semantic_f1"]),
)
@click.option("--num-threads", default=1, type=int)
@click.option("--display-table", is_flag=True, help="Show results table")
@click.option("--lm", help="LM model to use")
def evaluate_run(
    module: str,
    devset: str,
    metric: str,
    num_threads: int,
    display_table: bool,
    lm: str | None,
):
    """Evaluate a module on a devset.

    MODULE: Name of a generated module (e.g., mymodule) or path to .json program
    DEVSET: Path to JSON file with evaluation examples
    """
    from dspytools.core._dspy import dspy

    setup_dspy(model=lm)

    # Load the module
    program = _load_program(module)

    # Load devset
    examples = load_trainset(devset)

    # Build metric
    metric_fn = _get_metric(metric)

    # Run evaluation

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as pbar:
        pbar.add_task("Evaluating...", total=None)

        evaluator = dspy.Evaluate(
            devset=examples,
            metric=metric_fn,
            num_threads=num_threads,
            display_progress=False,
            display_table=display_table,
        )
        result = evaluator(program)

    _log.info(
        "evaluate_run",
        module=module,
        score=result.score,
        examples=len(examples),
        metric=metric,
    )
    panel(
        "Evaluation Results",
        f"[bold]Score:[/] {result.score:.2f}%\n"
        f"[bold]Devset:[/] {len(examples)} examples\n"
        f"[bold]Metric:[/] {metric}\n"
        f"[bold]Program:[/] {module}",
        border_style="green",
    )


@evaluate_cmd.command(name="list-metrics", cls=LLMCommand)
def evaluate_metrics():
    """List available built-in metrics."""
    table(
        "Built-in Metrics",
        ["Name", "Description"],
        [
            ["exact_match", "Exact string match between prediction and label"],
            ["passage_match", "Check if prediction contains passage text"],
            ["semantic_f1", "Semantic F1 score (requires numpy)"],
        ],
    )


def _load_program(module: str):
    """Load a module by name or path."""
    from dspytools.config.settings import modules_dir
    from dspytools.core._dspy import dspy

    # Try as a generated module first
    mod_path = modules_dir() / f"{module.lower()}.py"
    if mod_path.exists():
        spec = util.spec_from_file_location(module, str(mod_path))
        if spec and spec.loader:
            mod = util.module_from_spec(spec)
            _sys.modules[module] = mod
            spec.loader.exec_module(mod)
            # Find any dspy.Module subclass in the module
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, dspy.Module)
                    and attr is not dspy.Module
                ):
                    return attr()
        error(f"Could not load module '{module}'")
        raise click.Abort()

    # Try as a compiled program file
    prog_path = Path(module)
    if prog_path.exists():
        program = dspy.Predict("input -> output")
        program.load(str(prog_path))
        return program

    error(f"Module '{module}' not found")
    raise click.Abort()


def _get_metric(name: str):
    """Get a built-in metric function."""
    from dspytools.core._dspy import dspy

    metrics = {
        "exact_match": lambda ex, pred, trace=None: (
            1.0 if getattr(pred, "answer", "") == getattr(ex, "answer", "") else 0.0
        ),
        "passage_match": lambda ex, pred, trace=None: (
            1.0 if getattr(ex, "answer", "") in getattr(pred, "answer", "") else 0.0
        ),
        "semantic_f1": dspy.evaluate.SemanticF1(),
    }
    return metrics.get(name, metrics["exact_match"])
