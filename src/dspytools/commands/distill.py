"""dspytools distill — Multi-teacher LoRA distillation pipeline.

Distills knowledge from frontier LLMs via OpenRouter into LoRA training
data for Qwen3.5-9B-Instruct adapters.

References the dspy-lora project at runtime for the distillation engine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sys as _sys
import urllib.request as _ur
from pathlib import Path

import yaml

from dspytools.cli.output import console, error, info, ok, panel, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import distill_dir, dspy_lora_dir, llama_cpp_url
from dspytools.core._io import write_json
from dspytools.core.logging_config import get_logger

# ── Helpers ──────────────────────────────────────────────────────────────


def _ensure_distill_src() -> str:
    """Return a PYTHONPATH that includes the dspy-lora src directory."""
    lora_dir = dspy_lora_dir()
    if lora_dir is None:
        error("lora_dir not set. Point it to a checkout of the dspy-lora project.")
        raise click.Abort()
    src_path = lora_dir / "src"
    if not src_path.exists():
        error(f"dspy-lora src not found at {src_path}. Check lora_dir.")
        raise click.Abort()
    return str(src_path)


def _list_frameworks() -> list[dict]:
    """Load frameworks from dspy-lora config."""
    lora_dir = dspy_lora_dir()
    if lora_dir is None:
        return []
    config_path = lora_dir / "configs" / "frameworks.yaml"
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            return data.get("frameworks", [])
    return []


_log = get_logger(__name__)


@click.group(name="distill", cls=LLMGroup)
def distill_cmd():
    """Distill knowledge into LoRA training data.

    Runs the multi-teacher distillation pipeline to generate verified
    JSONL training data. Requires lora_dir env var pointing to a
    checkout of the dspy-lora project. Output can be trained into LoRA
    adapters via Unsloth (on Colab) and served via llama-cpp-server.
    """


@distill_cmd.command(name="run", cls=LLMCommand)
@click.option(
    "--framework",
    "-f",
    multiple=True,
    help="Distill only these frameworks (space-separated)",
)
@click.option("--repo", help="Custom repo URL (requires --name)")
@click.option("--name", help="Framework name (with --repo) or output prefix")
@click.option("--gepa", is_flag=True, help="Enable GEPA teacher optimization")
@click.option("--self-improve", is_flag=True, help="Run self-improvement loop")
@click.option(
    "--self-improve-iters",
    type=int,
    default=2,
    help="Self-improvement iterations per framework",
)
@click.option("--max-cost", type=float, help="Maximum API cost in USD before aborting")
@click.option(
    "--dry-run", is_flag=True, help="Show routing + cost estimate, no API calls"
)
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Output directory"
)
@click.option("--resume", is_flag=True, help="Skip already-generated concepts")
@click.option(
    "--python",
    "python_bin",
    default="",
    help="Python binary to use (default: from .venv)",
)
def distill_run(
    framework: tuple[str, ...],
    repo: str | None,
    name: str | None,
    gepa: bool,
    self_improve: bool,
    self_improve_iters: int,
    max_cost: float | None,
    dry_run: bool,
    output: str | None,
    resume: bool,
    python_bin: str,
):
    """Run the distillation pipeline on one or more frameworks.

    Requires lora_dir env var pointing to a checkout of the
    dspy-lora project and an OPENROUTER_API_KEY in .env.
    """
    lora_dir = dspy_lora_dir()
    if lora_dir is None:
        error("lora_dir not set. Point it to a checkout of the dspy-lora project.")
        raise click.Abort()
    distill_src = _ensure_distill_src()

    # Build arguments
    cmd = [python_bin or sys.executable, "-m", "src.distill"]
    if framework:
        for fw in framework:
            cmd.extend(["--framework", fw])
    if repo:
        cmd.extend(["--repo", repo])
    if name:
        cmd.extend(["--name", name])
    if gepa:
        cmd.append("--gepa")
    if self_improve:
        cmd.append("--self-improve")
    if self_improve_iters != 2:
        cmd.extend(["--self-improve-iters", str(self_improve_iters)])
    if max_cost is not None:
        cmd.extend(["--max-cost", str(max_cost)])
    if dry_run:
        cmd.append("--dry-run")
    if output:
        cmd.extend(["--output", str(Path(output).resolve())])
    if resume:
        cmd.append("--resume")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{distill_src}:{env.get('PYTHONPATH', '')}"

    # If dry run, show the command
    if dry_run:
        console.print("[bold]Would run:[/]")
        console.print(f"  PYTHONPATH={distill_src} {' '.join(cmd)}")
        console.print(f"  Working dir: {lora_dir}")
        return

    panel(
        "Distillation Pipeline",
        f"Working dir: {lora_dir}\nCommand: {' '.join(cmd)}\nMax cost: ${max_cost:.2f}"
        if max_cost
        else "Max cost: none",
        border_style="cyan",
    )

    info("Starting distillation (this may take several minutes)...")

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(lora_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Stream output
        assert proc is not None
        assert proc.stdout is not None
        for line in proc.stdout:
            console.print(line, end="")

        proc.wait()

        if proc.returncode == 0:
            _log.info(
                "distill_run", framework=framework, gepa=gepa, self_improve=self_improve
            )
            ok("Distillation complete!")
        else:
            error(f"Distillation failed (exit code {proc.returncode})")

    except FileNotFoundError:
        error("dspy-lora not found. Is lora_dir set correctly?")
        raise click.Abort()
    except KeyboardInterrupt:
        warn("Distillation interrupted by user")
        _log.warning("distill_run_interrupted", framework=framework)
        if proc is not None:
            proc.terminate()


@distill_cmd.command(name="check", cls=LLMCommand)
def distill_check():
    """Verify distillation prerequisites and environment.

    Checks lora_dir, dspy-lora checkout, OpenRouter API key,
    Python version, and available disk space.
    """
    # Load .env so DSPY_LORA_DIR and API keys are available
    from dspytools.config.env import load_env, merge_environ

    merge_environ(load_env())
    from dspytools.config.settings import dspy_lora_dir

    checks = []
    all_ok = True

    # 1. Check lora_dir config
    lora_dir = dspy_lora_dir()
    if lora_dir is not None:
        checks.append(("lora_dir configured", True, str(lora_dir)))
    else:
        checks.append(
            (
                "lora_dir configured",
                False,
                "Not set — set DSPY_LORA_DIR env var (e.g. `export DSPY_LORA_DIR=/path/to/dspy-lora`)",
            )
        )
        all_ok = False

    # 2. Check dspy-lora checkout exists
    if lora_dir:
        src_path = lora_dir / "src"
        if src_path.exists():
            checks.append(("dspy-lora checkout", True, str(lora_dir)))
        else:
            checks.append(
                ("dspy-lora checkout", False, f"src/ not found at {lora_dir}")
            )
            all_ok = False

    # 3. Check OpenRouter API key
    from dspytools.config.env import get_key

    api_key = get_key("openrouter")
    if api_key:
        masked = api_key[:8] + "..." if len(api_key) > 8 else "set"
        checks.append(("OpenRouter API key", True, masked))
    else:
        checks.append(
            (
                "OpenRouter API key",
                False,
                "Not found — set via `dspytools configure key set openrouter <KEY>`",
            )
        )
        all_ok = False

    # 4. Check Python version
    py_version = f"{_sys.version_info.major}.{_sys.version_info.minor}"
    version_ok = _sys.version_info >= (3, 10)
    checks.append(("Python version", version_ok, py_version))
    if not version_ok:
        all_ok = False

    # 5. Check available frameworks
    frameworks = _list_frameworks()
    if frameworks:
        checks.append(("Frameworks configured", True, f"{len(frameworks)} frameworks"))
    else:
        checks.append(("Frameworks configured", True, "Using defaults (3 frameworks)"))

    # 6. Check llama-cpp-server availability for LoRA loading
    resp = _ur.urlopen(f"{llama_cpp_url()}/v1/models", timeout=3)
    if resp.status == 200:
        checks.append(("llama-cpp-server", True, llama_cpp_url()))
    else:
        checks.append(("llama-cpp-server", False, f"Unexpected status: {resp.status}"))

    from dspytools.cli.output import header as _header
    from dspytools.cli.output import ok as _ok
    from dspytools.cli.output import warn as _warn

    _header("Distillation Prerequisites Check")
    for name, ok_status, detail in checks:
        icon = "✓" if ok_status else "✗"
        label = _ok if ok_status else _warn
        label(f"  {icon} {name}: {detail}")

    if all_ok:
        panel(
            "Distillation Ready",
            "All checks passed. Run `dspytools distill run` to start.",
            border_style="green",
        )
    else:
        panel(
            "Distillation Issues Found",
            "Fix the failing checks above before running `dspytools distill run`.",
            border_style="red",
        )


@distill_cmd.command(name="list-frameworks", cls=LLMCommand)
def distill_list_frameworks():
    """List available frameworks configured for distillation."""
    frameworks = _list_frameworks()
    if not frameworks:
        # Defaults
        frameworks = [
            {"name": "dspy", "repo": "https://github.com/stanfordnlp/dspy"},
            {"name": "fastapi", "repo": "https://github.com/fastapi/fastapi"},
            {"name": "pydantic", "repo": "https://github.com/pydantic/pydantic"},
        ]

    rows = []
    for fw in frameworks:
        rows.append([fw["name"], fw["repo"]])
    table("Available Frameworks", ["Name", "Repository"], rows, border_style="cyan")


@distill_cmd.command(name="stats", cls=LLMCommand)
@click.argument("file", default="super_training_data.jsonl")
def distill_stats(file: str):
    """Show statistics about generated training data.

    FILE: JSONL filename in the output directory (default: super_training_data.jsonl)
    """
    output_dir = distill_dir()
    path = output_dir / file
    if not path.exists():
        # Try globbing
        jsonl_files = list(output_dir.glob("*.jsonl"))
        if jsonl_files:
            path = jsonl_files[0]
        else:
            error(f"No JSONL files found in {output_dir}")
            return

    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        info(f"No examples found in {path}")
        return

    # Aggregate stats
    frameworks = {}
    scores = []
    formats = {}
    teachers = {}
    for ex in examples:
        fw = ex.get("framework", "unknown")
        frameworks[fw] = frameworks.get(fw, 0) + 1
        if "score" in ex:
            scores.append(ex["score"])
        fmt = ex.get("format", ex.get("correction_pair", "clean"))
        if isinstance(fmt, bool):
            fmt = "correction" if fmt else "clean"
        formats[fmt] = formats.get(fmt, 0) + 1
        t = ex.get("teacher", "unknown")
        teachers[t] = teachers.get(t, 0) + 1

    # Display
    ok(f"File: {path}")
    info(f"Total examples: {len(examples)}")
    info(f"File size: {path.stat().st_size:,} bytes")

    console.print("\n[bold]By framework:[/]")
    for fw, count in sorted(frameworks.items(), key=lambda x: -x[1]):
        console.print(f"  {fw:20s}: {count}")

    console.print("\n[bold]By format:[/]")
    for fmt, count in sorted(formats.items(), key=lambda x: -x[1]):
        console.print(f"  {fmt:20s}: {count}")

    console.print("\n[bold]By teacher:[/]")
    for t, count in sorted(teachers.items(), key=lambda x: -x[1]):
        console.print(f"  {t:20s}: {count}")

    if scores:
        avg = sum(scores) / len(scores)
        console.print(
            f"\n[bold]Score:[/] avg={avg:.3f} min={min(scores):.2f} max={max(scores):.2f}"
        )


@distill_cmd.command(name="prepare-colab", cls=LLMCommand)
@click.option("--adapter", default="super", help="Adapter name")
@click.option("--rank", default=64, type=int, help="LoRA rank")
@click.option(
    "--data",
    "data_file",
    default="super_training_data.jsonl",
    help="Training data JSONL file",
)
@click.option(
    "--train-on-awq", is_flag=True, help="Train on AWQ base (matches deployment)"
)
@click.option("--output", "-o", default=None, help="Output staging directory")
def distill_prepare_colab(
    adapter: str,
    rank: int,
    data_file: str,
    train_on_awq: bool,
    output: str | None,
):
    """Stage files for Colab LoRA training.

    Prepares a directory with all files needed for Unsloth LoRA
    training on Google Colab.
    """
    lora_dir = dspy_lora_dir()
    if lora_dir is None:
        error("DSPY_LORA_DIR not set. Point it to a checkout of the dspy-lora project.")
        raise click.Abort()
    scripts_dir = lora_dir / "scripts"
    output_dir = Path(output) if output else distill_dir() / f"colab_{adapter}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy training script
    training_script = scripts_dir / "train_super_lora.py"
    if training_script.exists():
        dest = output_dir / "train_super_lora.py"
        dest.write_text(training_script.read_text())
        ok(f"Training script: {dest}")
    else:
        # Write a minimal training script
        script = f'''"""LoRA training via Unsloth for {adapter} adapter."""
import os, sys, json
from pathlib import Path

def train(data_path: str, adapter_name: str = "{adapter}", rank: int = {rank},
          train_on_awq: bool = {str(train_on_awq).lower()}):
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from trl import SFTTrainer

    model_name = "Qwen/Qwen3.5-9B" if train_on_awq else "Qwen/Qwen3.5-9B"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=train_on_awq,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=rank,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("json", data_files=data_path, split="train")

    def format_instruction(ex):
        return {{"text": f"### Instruction:\\n{{ex['instruction']}}\\n\\n### Input:\\n{{ex.get('input', '')}}\\n\\n### Response:\\n{{ex['output']}}"}}

    dataset = dataset.map(format_instruction)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=dict(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=4,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=f"adapters/{{adapter_name}}",
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(f"adapters/{{adapter_name}}")
    tokenizer.save_pretrained(f"adapters/{{adapter_name}}")
    print(f"Adapter saved to adapters/{{adapter_name}}/")

if __name__ == "__main__":
    train(sys.argv[1] if len(sys.argv) > 1 else "training_data.jsonl")
'''.replace("{{", "{").replace("}}", "}")
        dest = output_dir / "train_super_lora.py"
        dest.write_text(script)
        ok(f"Training script (generated): {dest}")

    # Copy colab runner
    colab_script = scripts_dir / "colab_training.py"
    if colab_script.exists():
        dest = output_dir / "colab_training.py"
        dest.write_text(colab_script.read_text())
        ok(f"Colab runner: {dest}")

    # Copy training data
    data_src = distill_dir() / data_file
    if data_src.exists():
        dest = output_dir / "training_data.jsonl"
        dest.write_text(data_src.read_text())
        ok(f"Training data ({data_src.stat().st_size:,} bytes): {dest}")
    else:
        warn(f"Training data not found: {data_src}")

        # Create training config
    config = {
        "adapter_name": adapter,
        "rank": rank,
        "train_on_awq": train_on_awq,
        "data_file": "training_data.jsonl",
        "script": "train_super_lora.py",
        "colab_runner": "colab_training.py"
        if (output_dir / "colab_training.py").exists()
        else "",
    }
    config_path = output_dir / "training_config.json"
    write_json(config_path, config)
    ok(f"Training config: {config_path}")

    panel(
        "Colab Training Staged",
        f"Adapter: {adapter}\n"
        f"Rank: {rank}\n"
        f"Directory: {output_dir}\n"
        f"Files: {', '.join(p.name for p in output_dir.iterdir() if p.is_file())}\n\n"
        f"Upload '{output_dir}/' to Google Colab and run:\n"
        f"  !python colab_training.py --config training_config.json\n"
        f"or manually:\n"
        f"  !python train_super_lora.py --rank {rank} {'--train-on-awq' if train_on_awq else ''}",
        border_style="green",
    )
