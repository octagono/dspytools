"""dspytools signature — Generate, list, show, delete DSPy signatures.

The signature generator is a compilable DSPy module living in
dspytools.generators — uses ChainOfThought with a descriptive signature to
produce optimized Python code via the LLM. Run `dspytools compile` against
dspytools.generators to boost quality.
"""

import re

from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core.registry import delete_signature, list_signatures
from dspytools.core.setup import setup_dspy
from dspytools.generators import SignatureGeneratorDSPy

# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@click.group(name="signature", cls=LLMGroup)
def signature_cmd():
    """Manage DSPy signatures."""


@signature_cmd.command(name="new", cls=LLMCommand)
@click.argument("prompt")
@click.option("--name", "-n", help="Output name (default: from prompt)")
@click.option("--model", help="LM to use for generation")
@click.option(
    "--instructions",
    "-i",
    help="Class docstring / system instructions for the signature",
)
def signature_new(
    prompt: str, name: str | None, model: str | None, instructions: str | None
):
    """Generate a DSPy signature using the LLM.

    Uses a DSPy ChainOfThought module as the generator. The LLM produces
    optimized Python code with proper types, descriptions, and instructions.

    Prompt format: "inputs -> outputs" with optional field descriptions.
    Use = after the type to add descriptions (e.g. "query: str = The search query").

    Examples:

      dspytools signature new "question: str = The question, context: str = Background info -> answer: str = The final answer"

      dspytools signature new "module_name: str, score: float -> should_recompile: bool, optimizer: str" -n RecompileSig

      dspytools signature new "code: str, lang: str -> issues: list[str], score: float"
    """
    setup_dspy(model=model)

    if "->" not in prompt:
        raise click.ClickException("prompt must contain '->' separator")

    # Infer class name from prompt if not provided
    if not name:
        out_part = prompt.rsplit("->", 1)[1].strip()
        first_out = (
            out_part.split(",")[0].strip().split(":")[0].strip().split("=")[0].strip()
        )
        name = f"Generated{''.join(w.capitalize() for w in first_out.split('_'))}"
        if "_" in name or "-" in name:
            name = "".join(w.capitalize() for w in name.replace("-", "_").split("_"))
        elif not name[0].isupper():
            name = name[0].upper() + name[1:]

    if not instructions:
        inames = prompt.rsplit("->", 1)[0].strip()
        inames_short = ", ".join(
            f.split(":")[0].strip().split("=")[0].strip() for f in inames.split(",")[:3]
        )
        instructions = (
            f"Generate a Python class named {name} that inherits from "
            f"dspy.Signature. Use dspy.InputField() for inputs and "
            f"dspy.OutputField() for outputs with descriptive 'desc' args. "
            f"Inputs: {inames_short}. Use from __future__ import annotations "
            f"and from dspytools.core._dspy import dspy."
        )

    # Run the DSPy generator module
    gen = SignatureGeneratorDSPy()
    code, field_count, warnings = gen.generate(prompt, name, instructions)

    if not code or len(code) < 80:
        raise click.ClickException("DSPy generator produced incomplete output")

    # Extract signature string from generated header comment
    first_line = code.strip().split("\n")[0].strip().strip('"').strip("'")
    if "Generated signature:" in first_line:
        sig_str = first_line.replace("Generated signature:", "").strip()
    else:
        sig_str = prompt

    # Save to signatures directory
    from dspytools.config.settings import signatures_dir

    sig_path = signatures_dir() / f"{name}.py"
    sig_path.write_text(code.strip() + "\n")

    n_in = (
        sig_str.split("->")[0].count(",") + 1
        if "," in sig_str.split("->")[0].strip()
        else 1
    )
    n_out = (
        sig_str.split("->")[1].count(",") + 1
        if "," in sig_str.split("->")[1].strip()
        else 1
    )

    click.echo(
        f"  Signature '{name}' saved ({n_in} inputs, {n_out} outputs, DSPy-generated)"
    )
    click.echo(f"  Signature: {sig_str}")
    for w in warnings or []:
        click.echo(f"  Note: {w}")


@signature_cmd.command(name="list", cls=LLMCommand)
def signature_list():
    """List generated signatures."""
    sigs = list_signatures()
    if sigs:
        for sig in sigs:
            click.echo(f"  {sig['name']}  ({sig['size']}B)")
    else:
        click.echo("  No signatures generated yet")


@signature_cmd.command(name="show", cls=LLMCommand)
@click.argument("name")
def signature_show(name: str):
    """Show a generated signature's contents."""
    from dspytools.config.settings import signatures_dir

    path = signatures_dir() / f"{name}.py"
    if path.exists():
        click.echo(path.read_text())
    else:
        raise click.ClickException(f"signature '{name}' not found")


@signature_cmd.command(name="delete", cls=LLMCommand)
@click.argument("name")
@click.confirmation_option(prompt="Delete this signature?")
def signature_delete(name: str):
    """Delete a generated signature."""
    if delete_signature(name):
        click.echo(f"  Deleted signature '{name}'")
    else:
        raise click.ClickException(f"signature '{name}' not found")


@signature_cmd.command(name="manipulate", cls=LLMCommand)
@click.argument("name")
@click.option("--append-field", help="Append output field (name:type:desc)")
@click.option("--prepend-field", help="Prepend input field (name:type:desc)")
@click.option("--delete-field", help="Delete field by name")
@click.option("--instructions", help="Set new instructions text")
def signature_manipulate(
    name: str,
    append_field: str | None,
    prepend_field: str | None,
    delete_field: str | None,
    instructions: str | None,
):
    """Manipulate a DSPy signature file."""
    from dspytools.config.settings import signatures_dir

    path = signatures_dir() / f"{name}.py"
    if not path.exists():
        raise click.ClickException(f"signature '{name}' not found")

    content = path.read_text()
    changes = []

    if instructions:
        # Replace or add instructions in the signature
        content = re.sub(
            r'(class \w+\([^)]*\):[^"]*"""[^"]*?)(?:\n[^"]*?)*(""")',
            lambda m: f"{m.group(1)}{instructions}\n{m.group(2)}",
            content,
            count=1,
        )
        changes.append("instructions updated")

    if delete_field:
        # Remove field references
        content = content.replace(f'"{delete_field}"', '"removed"')
        changes.append(f"deleted field '{delete_field}'")

    if append_field or prepend_field:
        changes.append("field manipulation requires regeneration (use 'signature new')")

    if changes:
        path.write_text(content)
        click.echo(f"  Signature '{name}' updated: {', '.join(changes)}")
    else:
        click.echo(
            "  No manipulation specified. Use --append-field, "
            "--prepend-field, --delete-field, or --instructions"
        )
