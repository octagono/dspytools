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
@click.option(
    "--answer-field",
    help="Prediction/example field to score (default: the program's output field)",
)
def evaluate_run(
    module: str,
    devset: str,
    metric: str,
    num_threads: int,
    display_table: bool,
    lm: str | None,
    answer_field: str | None,
):
    """Evaluate a module on a devset.

    MODULE: Name of a generated module (e.g., mymodule) or path to a compiled
    run directory / program.json
    DEVSET: Path to JSON file with evaluation examples
    """
    from dspytools.core._dspy import dspy

    setup_dspy(model=lm)

    # Load the module
    program = _load_program(module)

    # Load devset
    examples = load_trainset(devset)

    # Build metric — resolve the scored field from the program's own schema so
    # `input -> output` datasets are not graded against a nonexistent `answer`
    # field (guaranteed 0.0) or a SemanticF1 that reads question/response.
    val_field = _resolve_answer_field(program, answer_field)
    metric_fn = _get_metric(metric, val_field)

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
        field=val_field,
    )
    panel(
        "Evaluation Results",
        f"[bold]Score:[/] {result.score:.2f}%\n"
        f"[bold]Devset:[/] {len(examples)} examples\n"
        f"[bold]Metric:[/] {metric} (field: {val_field})\n"
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
            ["semantic_f1", "Normalized token F1 between prediction and label"],
        ],
    )


def _load_program(module: str):
    """Load a module by name or by compiled-run path.

    Paths are reconstructed through ``core.hotswap._load_program_from_run``
    (``metadata.json`` ``module_type`` + ``signature.json``) so the program is
    evaluated in the shape it was saved in. ``dspy.Predict(...).load(path)``
    cannot do that: DSPy requires the program to be recreated in the same
    shape, and silently drops demos when it is not.
    """
    from dspytools.config.settings import modules_dir
    from dspytools.core._dspy import dspy
    from dspytools.core.hotswap import _load_program_from_run

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

    # Try as a compiled run directory or a program.json inside one
    prog_path = Path(module)
    if prog_path.is_file() and prog_path.name == "program.json":
        run_path = prog_path.parent
    elif prog_path.is_dir() and (prog_path / "program.json").exists():
        run_path = prog_path
    elif prog_path.exists():
        error(
            f"'{module}' is not a compiled run: expected a run directory or a "
            "program.json with metadata.json + signature.json alongside it"
        )
        raise click.Abort()
    else:
        error(f"Module '{module}' not found")
        raise click.Abort()

    program = _load_program_from_run(run_path.name, run_path=run_path)
    if program is None:
        error(
            f"Could not reconstruct the program in '{run_path}' — a compiled run "
            "needs program.json, metadata.json (module_type) and signature.json"
        )
        raise click.Abort()
    return program


def _resolve_answer_field(program, explicit: str | None = None) -> str:
    """Field on prediction/example to score.

    Explicit ``--answer-field`` wins. Otherwise use the first non-reasoning
    output field of the program's own signature (``answer``, ``output``,
    ``code``, …), defaulting to ``"output"``.
    """
    if explicit:
        return explicit
    fields: list[str] = []
    predictors = getattr(program, "predictors", None)
    if callable(predictors):
        for predictor in predictors():
            signature = getattr(predictor, "signature", None)
            fields.extend(getattr(signature, "output_fields", {}) or {})
    for name in fields:
        if name not in ("reasoning", "rationale"):
            return name
    return "output"


def _field_text(obj, field: str) -> str:
    """String value of ``field`` on an example or prediction ("" when absent)."""
    value = getattr(obj, field, "")
    return "" if value is None else str(value).strip()


def _metric_field(ex, pred, val_field: str | None) -> str:
    """Field a metric should score.

    Explicit field wins. Otherwise resolve per example/prediction from the
    conventional names, so ``input -> answer`` and ``input -> output`` schemas
    both work for callers that do not pass a field (e.g. the MCP ``evaluate``
    tool re-uses these factories).
    """
    if val_field:
        return val_field
    for name in ("answer", "output"):
        if getattr(pred, name, None) not in (None, "") or getattr(
            ex, name, None
        ) not in (
            None,
            "",
        ):
            return name
    return "output"


def _token_f1(prediction: str, gold: str) -> float:
    """Normalized token F1 — the offline replacement for ``SemanticF1``.

    ``dspy.evaluate.SemanticF1`` reads ``example.question``/``example.response``
    and ``prediction.response``, which do not exist in this project's
    ``input``/``output`` schemas (it raised ``AttributeError``). This scores the
    resolved field pair instead, with no model call.
    """
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    remaining = list(gold_tokens)
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _get_metric(name: str, val_field: str | None = None):
    """Get a built-in metric function scoring ``val_field``.

    ``val_field=None`` resolves the field per example/prediction
    (``answer`` then ``output``) — callers that know the program's schema should
    pass it explicitly (``evaluate run`` does). ``exact_match`` delegates to
    ``core.metrics.answer_exact_match`` so the CLI shares one normalization rule
    with the compile/GFL path.
    """
    from dspytools.core.metrics import answer_exact_match

    def _exact_match(ex, pred, trace=None) -> float:
        field = _metric_field(ex, pred, val_field)
        gold = _field_text(ex, field)
        got = _field_text(pred, field)
        return 1.0 if answer_exact_match(gold, got) else 0.0

    def _passage_match(ex, pred, trace=None) -> float:
        field = _metric_field(ex, pred, val_field)
        gold = _field_text(ex, field)
        return 1.0 if gold and gold in _field_text(pred, field) else 0.0

    def _semantic_f1(ex, pred, trace=None) -> float:
        field = _metric_field(ex, pred, val_field)
        return _token_f1(_field_text(pred, field), _field_text(ex, field))

    metrics = {
        "exact_match": _exact_match,
        "passage_match": _passage_match,
        "semantic_f1": _semantic_f1,
    }
    return metrics.get(name, _exact_match)
