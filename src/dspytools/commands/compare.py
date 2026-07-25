"""dspytools compare — side-by-side evaluation of compiled programs.

Compares two compiled programs on the same devset with statistical confidence.
"""

from __future__ import annotations

import random

from dspytools.cli.output import console, info, label_option, ok, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click


@click.group(name="compare", cls=LLMGroup)
def compare_cmd():
    """Compare compiled programs side-by-side."""


@compare_cmd.command(name="programs", cls=LLMCommand)
@click.argument("program_a")
@click.argument("program_b")
@click.argument("devset_path")
@click.option("--metric", "-m", default="exact_match", help="Metric function name")
@label_option
def compare_programs(
    program_a: str, program_b: str, devset_path: str, metric: str, label: str | None
):
    """Compare two compiled programs on a devset.

    Runs both programs on the same examples and reports:
    - Score comparison
    - Statistical significance (bootstrap p-value)
    - Per-example differences
    - Winner with confidence

    Example:
        dspytools compare programs abc123 def456 devset.json
    """
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.loaders import load_trainset
    from dspytools.core.registry import get_run
    from dspytools.core.setup import setup_dspy

    setup_dspy()

    # Load programs
    hotswap = HotSwapManager()
    hotswap.load_all()

    prog_a = hotswap.get(program_a)
    prog_b = hotswap.get(program_b)

    if not prog_a or not prog_b:
        missing = []
        if not prog_a:
            missing.append(program_a)
        if not prog_b:
            missing.append(program_b)
        click.echo(f"Program(s) not found: {', '.join(missing)}", err=True)
        click.echo("Use dspytools compile first or check list_compiled_runs")
        raise click.Abort()

    # Load devset
    devset = load_trainset(devset_path)
    info(f"Devset: {len(devset)} examples")

    meta_a = get_run(program_a)
    if meta_a is None:
        raise click.ClickException(f"Run '{program_a}' not found")
    meta_b = get_run(program_b)
    if meta_b is None:
        raise click.ClickException(f"Run '{program_b}' not found")

    console.print(f"\n[bold]Program A:[/] {program_a} ({meta_a.get('optimizer', '?')})")
    console.print(f"[bold]Program B:[/] {program_b} ({meta_b.get('optimizer', '?')})")
    console.print()

    # Evaluate both
    from dspytools.core.metrics import exact_match_metric

    metric_fn = exact_match_metric("output")
    scores_a = []
    scores_b = []
    ties = 0
    a_wins = 0
    b_wins = 0

    def _score(prog, example):
        kwargs = (
            example.inputs()
            if hasattr(example, "inputs")
            else {"input": getattr(example, "input", "")}
        )
        pred = prog(**kwargs)
        return metric_fn(example, pred)

    for ex in devset[: min(50, len(devset))]:
        sa = _score(prog_a, ex)
        sb = _score(prog_b, ex)
        scores_a.append(sa)
        scores_b.append(sb)

        if sa > sb:
            a_wins += 1
        elif sb > sa:
            b_wins += 1
        else:
            ties += 1

    score_a = sum(scores_a) / len(scores_a) if scores_a else 0
    score_b = sum(scores_b) / len(scores_b) if scores_b else 0

    # Bootstrap p-value

    n_bootstrap = 500
    diff = score_a - score_b
    diffs = []
    for _ in range(n_bootstrap):
        sample = random.choices(list(zip(scores_a, scores_b)), k=len(scores_a))
        sa = sum(s[0] for s in sample) / len(sample)
        sb = sum(s[1] for s in sample) / len(sample)
        diffs.append(sa - sb)

    p_value = (
        sum(1 for d in diffs if d <= 0) / n_bootstrap
        if diff > 0
        else sum(1 for d in diffs if d >= 0) / n_bootstrap
    )

    # Winner
    if diff > 0 and p_value < 0.05:
        winner = "A"
        confidence = "significant"
    elif diff < 0 and p_value < 0.05:
        winner = "B"
        confidence = "significant"
    else:
        winner = "tie"
        confidence = f"not significant (p={p_value:.3f})"

    rows = [
        ["Score", f"{score_a:.4f}", f"{score_b:.4f}", f"{diff:+.4f}"],
        ["Wins", str(a_wins), str(b_wins), str(ties) + " ties"],
        ["p-value", "", "", f"{p_value:.4f}"],
        ["Winner", winner if winner != "tie" else "-", "", confidence],
    ]
    table(f"Compare: {program_a} vs {program_b}", ["Metric", "A", "B", "\u0394"], rows)

    if winner != "tie":
        ok(f"Program {winner} is better ({confidence})")
    else:
        warn("No significant difference between programs")
