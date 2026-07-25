"""dspytools module — Generate, list, show, call DSPy modules.

The module generator is a compilable DSPy module living in
dspytools.generators — uses ChainOfThought with a descriptive signature to
produce optimized Python code via the LLM. Run `dspytools compile` against
dspytools.generators to boost quality.
"""

import importlib.util
import json
import sys

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core.registry import delete_module, list_modules
from dspytools.core.setup import setup_dspy
from dspytools.generators import ModuleGeneratorDSPy

# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@click.group(name="module", cls=LLMGroup)
def module_cmd():
    """Manage DSPy modules."""


@module_cmd.command(name="new", cls=LLMCommand)
@click.argument("name")
@click.option(
    "--signature", "-s", "sig_name", help="Signature name (from signature new)"
)
@click.option("--model", help="LM to use for generation")
@click.option(
    "--type",
    "-t",
    "mod_type",
    default="Predict",
    type=click.Choice(
        [
            "Predict",
            "ChainOfThought",
            "ReActV2",
            "MultiChainComparison",
            "Tool",
            "Generate",
        ]
    ),
    help="Module type (default: Predict)",
)
@click.option("--instructions", "-i", help="Class docstring or system prompt")
@click.option(
    "--from-prompt", help="Inline prompt: 'inputs -> outputs' (creates sig on-the-fly)"
)
def module_new(
    name: str,
    sig_name: str | None,
    model: str | None,
    mod_type: str,
    instructions: str | None,
    from_prompt: str | None,
):
    """Generate a DSPy module with a typed forward() method.

    Uses a DSPy ChainOfThought module as the generator. The LLM produces
    optimized Python code with a proper typed forward().

    The generator itself is a compilable DSPy module.

    Examples:

      dspytools module new AnswerQuestion --signature QASignature

      dspytools module new QA -t ChainOfThought -i "QA helper" \
          --from-prompt "question -> answer"

      dspytools module new Planner
    """
    if model is None:
        # Code generation is a teacher-level task — use teacher LM by default
        from dspytools.config.settings import load_config

        cfg = load_config()
        teacher = cfg.get("lm", {}).get("teacher", {})
        model = teacher.get("model")
    setup_dspy(model=model)

    from dspytools.core._dspy import dspy

    # Determine signature string
    if from_prompt:
        sig_str = from_prompt
        if "->" not in sig_str:
            click.echo("  --from-prompt must contain '->' separator", err=True)
            raise click.Abort()
    elif sig_name:
        from dspytools.config.settings import signatures_dir

        sig_path = signatures_dir() / f"{sig_name}.py"
        if not sig_path.exists():
            click.echo(
                f"  Signature '{sig_name}' not found. Create it first with "
                "'dspytools signature new'",
                err=True,
            )
            raise click.Abort()
        spec = importlib.util.spec_from_file_location(sig_name, sig_path)
        if not spec or not spec.loader:
            click.echo(f"  Failed to load signature '{sig_name}'", err=True)
            raise click.Abort()
        sig_mod = importlib.util.module_from_spec(spec)
        sys.modules[sig_name] = sig_mod
        spec.loader.exec_module(sig_mod)
        sig_cls = getattr(sig_mod, sig_name, None)
        if sig_cls is None:
            for attr_name in dir(sig_mod):
                attr = getattr(sig_mod, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, dspy.Signature)
                    and attr is not dspy.Signature
                ):
                    sig_cls = attr
                    break
        if sig_cls is None:
            click.echo(f"  No Signature class found in '{sig_path}'", err=True)
            raise click.Abort()
        sig_str = str(sig_cls)
    else:
        click.echo("  Either --signature or --from-prompt is required", err=True)
        raise click.Abort()

    if not instructions:
        instructions = f"A {mod_type.lower()} module for {name}"

    # Run the DSPy generator module
    gen = ModuleGeneratorDSPy()
    code, has_tools, out_fields = gen.generate(
        name,
        sig_str,
        mod_type,
        instructions,
    )

    if not code or len(code) < 80:
        click.echo("  DSPy generator produced incomplete output", err=True)
        raise click.Abort()

    # Save to modules directory
    from dspytools.config.settings import modules_dir

    mod_path = modules_dir() / f"{name.lower()}.py"
    mod_path.write_text(code.strip() + "\n")

    n_out = len(out_fields)
    click.echo(
        f"  Module '{name}' saved ({mod_type}, DSPy-generated"
        f"{', has tools' if has_tools else ''}"
        f"{f', {n_out} output fields' if n_out else ''})"
    )
    click.echo(f"  Signature: {sig_str[:60]}{'...' if len(sig_str) > 60 else ''}")


@module_cmd.command(name="list", cls=LLMCommand)
def module_list():
    """List generated modules."""
    mods = list_modules()
    if mods:
        for mod in mods:
            click.echo(f"  {mod['name']}  ({mod['size']}B)")
    else:
        click.echo("  No modules generated yet")


@module_cmd.command(name="show", cls=LLMCommand)
@click.argument("name")
def module_show(name: str):
    """Show the source code of a generated module."""
    from dspytools.config.settings import modules_dir

    mod_path = modules_dir() / f"{name.lower()}.py"
    if not mod_path.exists():
        click.echo(f"  Module '{name}' not found", err=True)
        raise click.ClickException(f"Module '{name}' not found")

    code = mod_path.read_text()
    click.echo(code)


@module_cmd.command(name="call", cls=LLMCommand)
@click.argument("name")
@click.option(
    "--inputs",
    "-i",
    multiple=True,
    help="Inputs as KEY=VALUE (can be specified multiple times)",
)
def module_call(name: str, inputs: tuple[str, ...]):
    """Call a generated module with inputs."""
    from dspytools.config.settings import modules_dir

    class_name = name.replace("-", "_")
    mod_path = modules_dir() / f"{class_name.lower()}.py"
    if not mod_path.exists():
        click.echo(f"  Module '{name}' not found", err=True)
        raise click.Abort()

    spec = importlib.util.spec_from_file_location(class_name, str(mod_path))
    if not spec or not spec.loader:
        click.echo(f"  Failed to load module '{name}'", err=True)
        raise click.Abort()

    mod = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = mod
    spec.loader.exec_module(mod)

    # Parse inputs (support JSON format or KEY=VALUE)
    kwargs = {}
    for inp in inputs:
        if inp.startswith("{") and inp.endswith("}"):
            try:
                parsed = json.loads(inp)
                if isinstance(parsed, dict):
                    kwargs.update({k.strip(): v for k, v in parsed.items()})
                    continue
            except json.JSONDecodeError:
                pass
        if "=" in inp:
            k, v = inp.split("=", 1)
            kwargs[k.strip()] = v.strip()

    setup_dspy()
    func_name = f"{class_name.lower()}_call"
    if hasattr(mod, func_name):
        result = getattr(mod, func_name)(**kwargs)
        click.echo(f"  Result: {result}")
    else:
        # Scan for a dspy.Module subclass or callable class (case-insensitive)
        from dspytools.core._dspy import dspy

        found_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name, None)
            if attr is None or not isinstance(attr, type):
                continue
            if issubclass(attr, dspy.Module) and attr is not dspy.Module:
                found_cls = attr
                break
        if found_cls is None:
            # Fallback: try PascalCase from class_name
            pascal = class_name.replace("_", " ").title().replace(" ", "")
            if hasattr(mod, pascal):
                found_cls = getattr(mod, pascal)
        if found_cls is not None:
            instance = found_cls()
            result = instance(**kwargs)
            click.echo(f"  Result: {result}")
        else:
            click.echo(f"  No callable entry point in module '{name}'", err=True)
            raise click.Abort()


@module_cmd.command(name="delete", cls=LLMCommand)
@click.argument("name")
@click.confirmation_option(prompt="Delete this module?")
def module_delete(name: str):
    """Delete a generated module."""
    if delete_module(name):
        click.echo(f"  Deleted module '{name}'")
    else:
        click.echo(f"  Module '{name}' not found", err=True)
        raise click.Abort()
