"""dspytools pipeline — compose and run multi-module DSPy pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.cli.output import console, info, label_option, ok, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click


@click.group(name="pipeline", cls=LLMGroup)
def pipeline_cmd():
    """Compose and manage multi-module DSPy pipelines."""


@pipeline_cmd.command(name="compose", cls=LLMCommand)
@click.argument("modules", nargs=-1)
@click.option("--name", "-n", default="pipeline", help="Pipeline name")
@label_option
def pipeline_compose(modules: tuple[str, ...], name: str, label: str | None):
    """Chain multiple DSPy modules into a pipeline.

    Example:
        dspytools pipeline compose analyze gen sum --name repo2llms
    """
    from dspytools.core._dspy import dspy
    from dspytools.core.output import create_run_dir, save_program
    from dspytools.core.registry import register_run
    from dspytools.core.setup import setup_dspy

    setup_dspy()

    if len(modules) < 2:
        click.echo("Need at least 2 modules to compose a pipeline.", err=True)
        raise click.Abort()

    info(
        f"Composing pipeline '{name}' from {len(modules)} modules: {', '.join(modules)}"
    )

    # Create a chained module
    class Pipeline(dspy.Module):
        def __init__(self):
            super().__init__()
            self.stages = []
            for _mod_name in modules:
                # Try to predict each module
                stage = dspy.Predict("input -> output")
                self.stages.append(stage)

        def forward(self, input: str) -> dspy.Prediction:
            x = input
            for _i, stage in enumerate(self.stages):
                result = stage(input=x)
                x = getattr(result, "output", str(result))
            return dspy.Prediction(output=x)

    pipeline = Pipeline()

    run_id, run_path = create_run_dir(f"pipeline_{name}", label)
    save_program(
        run_path,
        pipeline,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="pipeline",
    )
    register_run(run_id, {"name": name, "modules": list(modules), "type": "pipeline"})

    ok(f"Pipeline '{name}' created with {len(modules)} stages \u2192 {run_id}")
    info(f"Run: dspytools run {run_id} --input 'your text'")


@pipeline_cmd.command(name="list", cls=LLMCommand)
def pipeline_list():
    """List saved pipelines."""
    from dspytools.core.registry import list_compiled_runs

    runs = list_compiled_runs()
    pipelines = [r for r in runs if r.get("type") == "pipeline"]

    if pipelines:
        rows = [
            [
                r["id"],
                r.get("name", "?"),
                str(r.get("modules", [])),
            ]
            for r in pipelines
        ]
        table("Pipelines", ["ID", "Name", "Modules"], rows)
    else:
        info("No pipelines found. Create one with: dspytools pipeline compose")


@pipeline_cmd.command(name="run", cls=LLMCommand)
@click.argument("pipeline_id")
@click.option(
    "--input", "-i", "input_text", required=True, help="Input text for first stage"
)
def pipeline_run(pipeline_id: str, input_text: str):
    """Run a composed pipeline on input text."""
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.setup import setup_dspy

    setup_dspy()

    hotswap = HotSwapManager()
    try:
        hotswap.swap(pipeline_id)
    except KeyError:
        console.print(f"[red]Pipeline '{pipeline_id}' not found.[/red]")
        raise click.Abort()
    result = hotswap.infer(input=input_text)

    console.print(f"\n[bold]Input:[/] {input_text}")
    console.print(f"[bold]Output:[/] {result.get('output', result)}")
