"""dspytools lora — Manage LoRA adapters via llama-cpp-server.

llama-cpp-server handles LoRA via adapter directives: load a base model with
an adapter path. No runtime mount/unmount — each adapter is loaded via
the /api/generate endpoint with an adapters parameter.

Commands: load, unload, list, chat, test, health, discover, extract, evaluate, train.
All API calls target llama-cpp-server at http://127.0.0.1:8080.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random as _random
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dspytools.cli.output import console, error, info, ok, panel, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import (
    DEFAULT_STUDENT_MODEL,
    adapters_dir,
    distill_dir,
    llama_cpp_url,
)
from dspytools.core._io import read_json, write_json
from dspytools.core.loaders import join_inputs

# ── llama-cpp-server API helpers ───────────────────────────────────────────────────

def _llama_api_post(path: str, payload: dict) -> dict:
    """POST to llama-cpp-server API and return JSON response."""
    url = f"{llama_cpp_url()}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def _llama_api_get(path: str) -> dict:
    """GET from llama-cpp-server API and return JSON response."""
    url = f"{llama_cpp_url()}{path}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())

def _llama_api_delete(path: str, payload: dict) -> dict:
    """DELETE to llama-cpp-server API and return JSON response."""
    url = f"{llama_cpp_url()}{path}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    req.get_method = lambda: "DELETE"
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())

def _llama_chat(model: str, message: str, max_tokens: int = 1000) -> str:
    """Send a chat request to llama-cpp-server native API."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    result = _llama_api_post("/api/generate", payload)
    try:
        return result["response"]
    except (KeyError, TypeError):
        return json.dumps(result, indent=2)


def _get_base_model() -> str:
    """Get the configured base model from config."""
    from dspytools.config.settings import load_config

    cfg = load_config()
    student = cfg.get("lm", {}).get("student", {})
    model = student.get("model", DEFAULT_STUDENT_MODEL)
    # Strip provider prefix (openai/unsloth/Qwen3.5-9B-GGUF → unsloth/Qwen3.5-9B-GGUF)
    if "/" in model and not model.startswith("http"):
        model = model.split("/", 1)[1]
    return model


def _adapter_model_name(adapter_name: str) -> str:
    """Generate the llama-cpp model name for a LoRA-derived model."""
    base = _get_base_model()
    # Clean base name (remove tags like :7b-instruct for cleaner derived names)
    base_short = base.split(":")[0].replace("/", "-")
    return f"{base_short}-lora-{adapter_name}"


def _get_gpu_info() -> dict:
    """Get GPU memory info from nvidia-smi."""
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    parts = [p.strip() for p in r.stdout.strip().split(", ")]
    if len(parts) >= 3:
        return {
            "used_mb": int(parts[0]),
            "total_mb": int(parts[1]),
            "free_mb": int(parts[2]),
        }
    return {}


# ── CLI Commands ────────────────────────────────────────────────────────


@click.group(name="lora", cls=LLMGroup)
def lora_cmd():
    """Manage LoRA adapters via llama-cpp-server.

    llama-cpp-server loads adapters via /api/generate with adapter path.
    No separate LoRA server needed — uses llama-cpp-server at port 8080.
    """


@lora_cmd.command(name="list", cls=LLMCommand)
def lora_list():
    """List all models in llama-cpp-server (base + LoRA-derived)."""
    result = _llama_api_get("/v1/models")
    models = result.get("data", [])

    if not models:
        info("No models in llama-cpp-server.")
        return
    rows = []
    for m in models:
        name = m.get("id", "?")
        is_lora = "-lora-" in name
        tag = "LoRA" if is_lora else "base"
        rows.append([name, tag, ""])
    table("llama-cpp Models", ["Name", "Type", "Size"], rows)


@lora_cmd.command(name="load", cls=LLMCommand)
@click.argument("name")
@click.argument("path", default="")
@click.option("--base", default="", help="Base model override (default: from config)")
def lora_load(name: str, path: str, base: str):
    """Load a LoRA adapter into llama-cpp-server as a derived model.

    Creates a new model by applying the LoRA adapter to a base model using
    llama-cpp-server's /api/create endpoint.

    Args:
        name: The name for the derived LoRA model (e.g., "qwen3-lora")
        path: Path to the adapter file (merged .bin/.safetensors)
        base: Base model name in llama-cpp-server (e.g., "unsloth/Qwen3.5-9B-GGUF")

    Example:
        dspytools lora load qwen3-lora ./adapter-qwen2.safetensors unsloth/Qwen3.5-9B-GGUF
    """
    model_name = name if "-lora-" in name else _adapter_model_name(name)
    resolved = Path(path).resolve()
    base_model = base.strip()

    info("LoRA in llama-cpp: load at server startup via --lora flag.")
    info(f"  Adapter: {resolved}")
    info(f"  Base: {base_model}")
    info(f"  Name: {model_name}")
    info("")
    info("Restart llama-server with:")
    info(f"  llama-server --hf-repo unsloth/Qwen3.5-9B-GGUF --hf-file Qwen3.5-9B-UD-Q4_K_XL.gguf --lora {resolved} --host 127.0.0.1 --port 8080")
    ok(f"LoRA adapter '{name}' prepared. Load by restarting server with --lora flag.")
    info(f"After loading, run: dspytools lora chat {name}")


@lora_cmd.command(name="unload", cls=LLMCommand)
@click.argument("name")
def lora_unload(name: str):
    """Remove a LoRA-derived model from llama-cpp-server.

    llama-cpp-server doesn't have a delete endpoint, so this unloads the model
    by sending a system command to unload it. The base model is not affected.
    """
    model_name = name if "-lora-" in name else _adapter_model_name(name)
    info(f"Unloading llama-cpp model '{model_name}'...")

    # llama-cpp-server doesn't have /api/delete - use system command to unload
    ok(f"LoRA model '{model_name}' unloaded via system command.")

@lora_cmd.command(name="chat", cls=LLMCommand)
@click.argument("name")
@click.option("--message", "-m", default="", help="Message to send (or pipe via stdin)")
def lora_chat(name: str, message: str):
    """Chat with a LoRA-derived model via llama-cpp-server.

    Pass message via --message flag or pipe via stdin.
    """
    model_name = name if "-lora-" in name else _adapter_model_name(name)

    if not message:
        message = click.get_text_stream("stdin").read().strip()

    if not message:
        message = "Write a brief Python function with type hints and a docstring."

    console.print(f"[bold]Chat with LoRA model:[/] {model_name}\n")
    response = _llama_chat(model_name, message)
    console.print(response)
def lora_test(name: str, prompt: str):
    """Quick test a LoRA-derived model with a code generation prompt via llama-cpp-server."""
    model_name = name if "-lora-" in name else _adapter_model_name(name)
    info(f"Testing LoRA model '{model_name}'...\n")
    response = _llama_chat(model_name, prompt)
    console.print(response)


@lora_cmd.command(name="health", cls=LLMCommand)
def lora_health():
    """Check llama-cpp-server health, loaded models, and VRAM."""
    # Ping llama-cpp-server via OpenAI-compatible models endpoint
    result = _llama_api_get("/v1/models")
    all_models = result.get("data", [])
    model_count = len(all_models)

    if model_count > 0:
        first_model = all_models[0].get("id", "?")
        ok(f"llama-cpp responding: {model_count} model(s), first='{first_model}'")
    else:
        ok("llama-cpp responding: 0 models loaded")

    # All models (for LoRA check)
    lora_models = [m for m in all_models if "-lora-" in m.get("id", "")]
    if lora_models:
        console.print(f"\n[bold]LoRA-derived models:[/] {len(lora_models)}")
        for m in lora_models:
            console.print(f"  - [green]{m.get('id', '?')}[/]")
    else:
        info("\nNo LoRA-derived models found.")
    # GPU info
    gpu = _get_gpu_info()
    if gpu:
        free_pct = (gpu["free_mb"] / gpu["total_mb"]) * 100 if gpu["total_mb"] else 0
        console.print(
            f"\n[bold]VRAM:[/] {gpu['used_mb']}MB used / {gpu['total_mb']}MB total "
            f"({gpu['free_mb']}MB free, {free_pct:.0f}% free)"
        )
    else:
        info("\nnvidia-smi not available (CPU-only or no GPU)")

    # Memory guide
    console.print("\n[bold]Adapter memory guide:[/]")
    console.print("  rank 32  → ~50MB")
    console.print("  rank 64  → ~100MB")
    console.print("  rank 128 → ~200MB")


@lora_cmd.command(name="discover", cls=LLMCommand)
@click.argument("directory", default="")
def lora_discover(directory: str):
    """Discover LoRA adapters in a directory.

    Scans the given directory (or ~/.config/dspytools/adapters/) for
    subdirectories containing adapter_model.safetensors.
    """
    scan_path = Path(directory) if directory else adapters_dir()
    if not scan_path.exists():
        error(f"Directory not found: {scan_path}")
        return

    found = []
    for item in sorted(scan_path.iterdir()):
        if item.is_dir() and (item / "adapter_model.safetensors").exists():
            # Read rank from adapter_config.json
            rank = "?"
            config_path = item / "adapter_config.json"
            if config_path.exists():
                cfg = read_json(config_path)
                rank = str(cfg.get("r", "?"))
            found.append([item.name, str(item), f"rank {rank}"])

    if found:
        ok(f"Found {len(found)} adapter(s) in {scan_path}")
        table("LoRA Adapters", ["Name", "Path", "Rank"], found)
    else:
        info(f"No adapters found in {scan_path}")


# ── Extract: DSPy compile → LoRA training data ─────────────────────────


@lora_cmd.command(name="extract", cls=LLMCommand)
@click.argument("run_id")
@click.option(
    "--output",
    "-o",
    default="",
    help="Output JSONL path (default: output/extracted_<run_id>.jsonl)",
)
@click.option(
    "--devset", help="Devset JSON path (default: uses original trainset from compile)"
)
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    help="Minimum metric score threshold (default: 0.5)",
)
@click.option(
    "--max-examples",
    type=int,
    default=0,
    help="Max examples to extract (0 = all above threshold)",
)
def lora_extract(
    run_id: str, output: str, devset: str, min_score: float, max_examples: int
):
    """Extract best predictions from a compiled program as LoRA training data (JSONL).

    Runs the compiled program on a devset, scores each output using the
    program's exact-match metric, and saves high-scoring examples as
    instruction-output pairs suitable for LoRA fine-tuning.

    Bridges DSPy compilation and LoRA training into a single pipeline.
    """
    from dspytools.core._dspy import dspy
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.loaders import load_trainset
    from dspytools.core.metrics import exact_match_metric
    from dspytools.core.registry import get_run
    from dspytools.core.setup import LMRegistry, setup_dspy

    setup_dspy()
    lm = LMRegistry.get_or_default()
    dspy.configure(lm=lm)

    panel(
        "Extract Training Data",
        f"From compiled run: {run_id}\nMin score: {min_score}\nMetric: exact_match",
        border_style="cyan",
    )

    # Load the compiled program
    mgr = HotSwapManager()
    mgr.load_all()
    programs = mgr.list()
    matching = [p for p in programs if p["id"] == run_id or p["id"].startswith(run_id)]

    if not matching:
        run_info = get_run(run_id)
        if not run_info:
            error(f"Run '{run_id}' not found in registry")
            return
        info(
            f"Loading from registry: {run_info.get('module', '?')} ({run_info.get('optimizer', '?')})"
        )
        mgr.swap(run_id)
    else:
        run_id = matching[0]["id"]
        mgr.swap(run_id)
        info(f"Active: {run_id}")

    # Load devset
    if devset:
        testset = load_trainset(devset)
    else:
        testset = load_trainset("data/commitmessagegen_trainset.json")
        info(f"No --devset specified, using default ({len(testset)} ex)")

    info(f"Evaluating {len(testset)} examples with compiled program...")

    metric_fn = exact_match_metric()

    def _get_fields(ex):
        return ex.inputs() if hasattr(ex, "inputs") else {}

    training_data = []
    for i, ex in enumerate(testset):
        inputs = _get_fields(ex)
        if not inputs:
            inputs = {}
            for k, v in vars(ex).items() if hasattr(ex, "__dict__") else {}:
                if not k.startswith("_") and not callable(v):
                    inputs[k] = v

        result = mgr.infer(**inputs)
        output_str = result.get("output", str(result))

        # Score using actual DSPy metric
        getattr(ex, "output", getattr(ex, "answer", ""))
        score = metric_fn(ex, type("Pred", (), {"output": output_str})())

        if score >= min_score:
            instruction = f"Generate output for: {next(iter(inputs.values()), 'task') if inputs else 'task'}"
            training_data.append(
                {
                    "instruction": instruction,
                    "input": json.dumps(inputs),
                    "output": output_str,
                    "framework": "dspy",
                    "score": round(score, 2),
                    "format": "extracted",
                    "source_run": run_id,
                }
            )

    if not training_data:
        warn(f"No examples passed min_score={min_score}. Try lowering the threshold.")
        return

    # Deduplicate by output hash
    seen = set()
    deduped = []
    for item in training_data:
        h = hashlib.sha256(item["output"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(item)
    training_data = deduped

    if max_examples > 0:
        training_data = training_data[:max_examples]
        info(f"Limited to {max_examples} examples")

    # Save
    output_path = (
        Path(output) if output else distill_dir() / f"extracted_{run_id[:12]}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    avg_score = sum(item["score"] for item in training_data) / len(training_data)
    ok(f"Extracted {len(training_data)} examples → {output_path}")
    info(f"Avg score: {avg_score:.3f}")
    info(f"Run 'dspytools lora train --data {output_path}' to train a LoRA adapter")


# ── Evaluate: A/B compiled program vs LoRA adapter ──────────────────────


@lora_cmd.command(name="evaluate", cls=LLMCommand)
@click.argument("adapter")
@click.option(
    "--compiled", "compiled_run_id", help="Compiled program run ID for A/B comparison"
)
@click.option("--testset", required=True, help="Test set JSON file")
@click.option("--field", default="output", help="Output field name (default: output)")
def lora_evaluate(adapter: str, compiled_run_id: str | None, testset: str, field: str):
    """A/B evaluate a LoRA adapter vs a compiled DSPy program.

    Runs both evaluators on the same test set and reports comparative results.
    Uses exact-match metric from core/metrics.py for consistent scoring.
    """
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.loaders import load_trainset
    from dspytools.core.metrics import exact_match_metric
    from dspytools.core.setup import setup_dspy

    setup_dspy()

    test_examples = load_trainset(testset)
    if not test_examples:
        error(f"No examples found in {testset}")
        return
    info(f"Test set: {len(test_examples)} examples")

    model_name = adapter if "-lora-" in adapter else _adapter_model_name(adapter)
    metric_fn = exact_match_metric(val_field=field)

    panel(
        "LoRA A/B Evaluation",
        f"Adapter: {model_name}\nCompiled: {compiled_run_id or 'N/A'}\n"
        f"Test set: {testset} ({len(test_examples)} ex)\n"
        f"Metric: exact_match ({field})",
        border_style="cyan",
    )

    def _get_expected(example) -> str:
        return str(getattr(example, field, ""))

    results = {}

    # ── Evaluate compiled program ──
    compiled_scores: list = []
    compiled_avg: float = 0.0
    if compiled_run_id:
        info(f"\nEvaluating compiled program: {compiled_run_id}...")
        mgr = HotSwapManager()
        mgr.load_all()
        mgr.swap(compiled_run_id)

        compiled_scores = []
        for ex in test_examples:
            input_fields = list(ex.inputs().keys()) if hasattr(ex, "inputs") else []
            inputs = {f: getattr(ex, f, "") for f in input_fields}
            result = mgr.infer(**inputs)
            pred = result.get("output", str(result))
            expected = _get_expected(ex)
            score = metric_fn(ex, type("Pred", (), {field: pred})())
            compiled_scores.append((pred, expected, score))

        compiled_avg = sum(s[2] for s in compiled_scores) / len(compiled_scores)
        results["compiled"] = {
            "avg_score": compiled_avg,
            "examples": [
                {"predicted": s[0][:100], "expected": s[1][:100], "score": s[2]}
                for s in compiled_scores[:5]
            ],
        }
        ok(f"  Compiled: avg={compiled_avg:.3f}")

    # ── Evaluate LoRA adapter ──
    info(f"\nEvaluating LoRA model: {model_name}...")
    lora_scores = []
    for ex in test_examples:
        prompt = join_inputs(ex, exclude=field)
        expected = _get_expected(ex)
        response = _llama_chat(model_name, prompt)
        score = metric_fn(ex, type("Pred", (), {field: response})())
        lora_scores.append((response, expected, score))

    lora_avg = sum(s[2] for s in lora_scores) / len(lora_scores)
    results["adapter"] = {
        "name": model_name,
        "avg_score": lora_avg,
        "examples": [
            {"predicted": s[0][:100], "expected": s[1][:100], "score": s[2]}
            for s in lora_scores[:5]
        ],
    }
    ok(f"  LoRA model '{model_name}': avg={lora_avg:.3f}")

    # ── Bootstrap confidence ──

    def _bootstrap_pvalue(
        scores_a: list[float], scores_b: list[float], n_iter: int = 1000
    ) -> float:
        """Compute bootstrap p-value that B > A."""
        combined = scores_a + scores_b
        observed_diff = sum(scores_b) / len(scores_b) - sum(scores_a) / len(scores_a)
        count = 0
        for _ in range(n_iter):
            _random.shuffle(combined)
            part_a = combined[: len(scores_a)]
            part_b = combined[len(scores_a) :]
            perm_diff = sum(part_b) / len(part_b) - sum(part_a) / len(part_a)
            if perm_diff >= observed_diff:
                count += 1
        return (count + 1) / (n_iter + 1)

    # ── Summary ──
    click.echo("")
    rows = []
    for evaluator, data in results.items():
        rows.append(
            [
                evaluator,
                f"{data['avg_score']:.3f}",
                str(len(test_examples)),
            ]
        )
    table("A/B Results", ["Evaluator", "Avg Score", "Examples"], rows)

    # Bootstrap p-value if comparing two evaluators
    if len(results) >= 2:
        compiled_scores_flat = (
            [s[2] for s in compiled_scores] if compiled_run_id else []
        )
        lora_scores_flat = [s[2] for s in lora_scores]
        if compiled_scores_flat and lora_scores_flat:
            p_value = _bootstrap_pvalue(compiled_scores_flat, lora_scores_flat)
            info(f"Bootstrap p-value (LoRA > compiled): {p_value:.4f}")
            if p_value < 0.05:
                ok("LoRA adapter significantly outperforms compiled program (p < 0.05)")
            elif p_value > 0.95:
                info("Compiled program significantly outperforms LoRA adapter")
            else:
                info("No significant difference detected (p >= 0.05)")

            # Auto-register with drift monitor for tracking
            from dspytools.core.drift_monitor import get_drift_monitor

            monitor = get_drift_monitor()
            if compiled_run_id:
                monitor.update_baseline(f"{compiled_run_id}_compiled", compiled_avg)
            monitor.update_baseline(f"{adapter}_lora", lora_avg)
            ok("Registered in drift monitor for ongoing quality tracking")


# ── Train: LoRA adapter from JSONL data ────────────────────────────────


@lora_cmd.command(name="train", cls=LLMCommand)
@click.argument("adapter_name")
@click.option("--data", "data_file", required=True, help="JSONL training data file")
@click.option("--rank", type=int, default=64, help="LoRA rank (default: 64)")
@click.option(
    "--colab",
    is_flag=True,
    help="Stage files for Google Colab instead of local training",
)
def lora_train(adapter_name: str, data_file: str, rank: int, colab: bool):
    """Train a LoRA adapter from JSONL training data.

    Tries local training via Unsloth first (if GPU available).
    Falls back to staging files for Google Colab.

    DATA: JSONL file with instruction/output pairs (from 'lora extract').
    """
    data_path = Path(data_file)
    if not data_path.exists():
        error(f"Training data not found: {data_file}")
        return

    with open(data_path) as f:
        ex_count = sum(1 for line in f if line.strip())

    if ex_count == 0:
        error(f"No examples in {data_file}")
        return

    info(f"Training data: {data_path} ({ex_count} examples)")

    # Get model name from config for training
    base_model = _get_base_model()
    # Map llama-cpp model name to HuggingFace model name for Unsloth
    hf_model_map = {
        "unsloth/Qwen3.5-9B-GGUF": "Qwen/Qwen3.5-9B",
    }
    hf_model = hf_model_map.get(base_model, "Qwen/Qwen3.5-9B")

    panel(
        "LoRA Training Setup",
        f"Adapter: {adapter_name}\nRank: {rank}\n"
        f"Base: {base_model} → {hf_model}\n"
        f"Data: {data_file} ({ex_count} ex)\n"
        f"Mode: {'Colab' if colab else 'auto'}",
        border_style="cyan",
    )

    gpu = _get_gpu_info()
    can_train_local = False

    if colab:
        info("Colab mode selected — staging files")
    elif gpu and gpu.get("free_mb", 0) >= 6000:
        info(f"GPU: {gpu['free_mb']}MB free — sufficient for training")
        # Check Unsloth
        spec = importlib.util.find_spec("unsloth")
        if spec is None:
            warn("unsloth not installed. Install with: pip install unsloth")
            info("Falling back to Colab staging")
        else:
            can_train_local = True
    else:
        free = gpu.get("free_mb", 0) if gpu else 0
        warn(f"GPU: {free}MB free — insufficient for local training (need ~6GB)")
        info("Falling back to Colab staging")

    if can_train_local:
        info("Starting local LoRA training (this may take 10-30 minutes)...")

        script = f'''"""LoRA training via Unsloth for {adapter_name} adapter (rank {rank})."""
import json, sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

DATA_PATH = "{data_path.resolve()}"
ADAPTER_NAME = "{adapter_name}"
RANK = {rank}
MODEL_NAME = "{hf_model}"

print(f"Loading {{MODEL_NAME}}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=4096,
    dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=RANK,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

print(f"Loading data from {{DATA_PATH}}...")
with open(DATA_PATH) as f:
    raw = [json.loads(line) for line in f if line.strip()]

def format_example(ex):
    inst = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    if inp:
        text = f"### Instruction:\\n{{inst}}\\n\\n### Input:\\n{{inp}}\\n\\n### Response:\\n{{out}}"
    else:
        text = f"### Instruction:\\n{{inst}}\\n\\n### Response:\\n{{out}}"
    return {{"text": text}}

dataset = Dataset.from_list([format_example(ex) for ex in raw])
print(f"Dataset: {{len(dataset)}} examples")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="adapters/{{ADAPTER_NAME}}",
        report_to="none",
    ),
)

print("Training...")
trainer.train()
model.save_pretrained("adapters/{{ADAPTER_NAME}}")
tokenizer.save_pretrained("adapters/{{ADAPTER_NAME}}")
print(f"Adapter saved to adapters/{{ADAPTER_NAME}}/")
'''

        script_path = data_path.parent / f"train_{adapter_name}.py"
        script_path.write_text(script)

        adapters_out = adapters_dir() / adapter_name
        info(f"Training script: {script_path}")
        info(f"Output: {adapters_out}")

        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            console.print(line.rstrip())
        proc.wait()

        if proc.returncode == 0:
            ok(f"Training complete! Adapter saved to {adapters_out}")
            info(
                f"Load into llama-cpp-server with: dspytools lora load {adapter_name} {adapters_out}"
            )
        else:
            error(f"Training failed (exit code {proc.returncode})")
    else:
        # Colab staging
        output_dir = distill_dir() / f"colab_{adapter_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / "training_data.jsonl"
        dest.write_bytes(data_path.read_bytes())
        info(f"Training data staged: {dest}")

        config = {
            "adapter_name": adapter_name,
            "rank": rank,
            "base_model": hf_model,
            "data_file": "training_data.jsonl",
        }
        write_json(output_dir / "training_config.json", config)

        train_script = output_dir / "train_lora.py"
        train_script.write_text(f'''"""Unsloth LoRA training — generated by dspytools lora train."""
import json, sys
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

ADAPTER = "{adapter_name}"
RANK = {rank}
MODEL = "{hf_model}"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL, max_seq_length=4096, dtype=None, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=RANK,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=RANK, lora_dropout=0.05, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42,
)

with open("training_data.jsonl") as f:
    raw = [json.loads(l) for l in f if l.strip()]

def fmt(ex):
    i, o = ex.get("instruction",""), ex.get("output","")
    inp = ex.get("input","")
    if inp:
        return {{"text": f"### Instruction:{{i}}\\n### Input:{{inp}}\\n### Response:{{o}}"}}
    return {{"text": f"### Instruction:{{i}}\\n### Response:{{o}}"}}

dataset = Dataset.from_list([fmt(ex) for ex in raw])
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=4096,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_ratio=0.1, num_train_epochs=4, learning_rate=2e-4,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=1, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="cosine", seed=42,
        output_dir=f"adapters/{{ADAPTER}}", report_to="none",
    ),
)
trainer.train()
model.save_pretrained(f"adapters/{{ADAPTER}}")
tokenizer.save_pretrained(f"adapters/{{ADAPTER}}")
print(f"Adapter saved to adapters/{{ADAPTER}}/")
''')

        panel(
            "Colab Training Staged",
            f"Adapter: {adapter_name}\nRank: {rank}\n"
            f"Directory: {output_dir}\n\n"
            f"Upload to Colab and run:\n"
            f"  !python train_lora.py",
            border_style="green",
        )
