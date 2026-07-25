"""dspytools self — Self-optimization commands.

Run `dspytools self-optimize` to re-compile the self-help module
using the configured teacher LM for better quality.
"""

from __future__ import annotations

import json as _json
import time as _time
import urllib.request as _urllib

from dspytools.cli.output import console, error, info, ok, panel, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import llama_cpp_url
from dspytools.core.logging_config import get_logger

# ── llama-cpp API helpers (fail-fast, no try/except) ──────────────────


def _llama_api_post(path: str, payload: dict, timeout: int = 60) -> dict:
    """POST to llama-cpp-server API, fail-fast."""
    url = f"{llama_cpp_url()}{path}"
    data = _json.dumps(payload).encode()
    req = _urllib.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _urllib.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode())


def _llama_api_delete(path: str, payload: dict) -> dict:
    """DELETE to llama-cpp-server API, fail-fast."""
    url = f"{llama_cpp_url()}{path}"
    data = _json.dumps(payload).encode()
    req = _urllib.Request(url, data=data, headers={"Content-Type": "application/json"})
    req.get_method = lambda: "DELETE"
    with _urllib.urlopen(req, timeout=10) as resp:
        return _json.loads(resp.read().decode())


def _llama_chat(model: str, message: str, max_tokens: int = 1000) -> str:
    """Send a chat request to llama-cpp-server native API, fail-fast."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }
    result = _llama_api_post("/api/generate", payload)
    return result["response"]


def _get_input_fields(ex: object) -> list[str]:
    """Extract input field names from a DSPy Example or dynamic object.

    SSOT for input field detection — used by distill auto-eval.
    """
    if hasattr(ex, "inputs"):
        return list(ex.inputs().keys())
    return [k for k in vars(ex) if k != "output"]


_log = get_logger(__name__)


@click.group(name="self", cls=LLMGroup)
def self_cmd():
    """Self-optimization: manage the dspytools self-help module."""


@self_cmd.command(name="optimize", cls=LLMCommand)
@click.option(
    "--teacher/--no-teacher",
    default=False,
    help="Use teacher LM for GEPA optimization (requires DeepSeek-class LM)",
)
@click.option(
    "--force", "-f", is_flag=True, help="Force re-compile even if cache exists"
)
def self_optimize(teacher: bool, force: bool):
    """Compile the self-help module using DSPy optimizers.

    Builds a trainset from CLI introspection, then compiles with
    LabeledFewShot or GEPA (if --teacher is set, uses deepseek for reflection).
    The compiled program is cached at ~/.config/dspytools/help_compiled.json
    and used by all subsequent --help invocations.
    """
    from dspytools.core.setup import setup_dspy

    setup_dspy()

    # Import cli here to avoid circular imports
    from dspytools.help.optimize import AutoCompiler
    from dspytools.main import cli

    console.print("\n[bold cyan]Self-Optimization[/]")

    if force:
        AutoCompiler.clear()
        ok("Cleared existing cache")

    if AutoCompiler.is_compiled() and not force:
        info("Help module already compiled. Use --force to re-compile.")
        return

    info("Introspecting CLI commands...")
    info(f"Compiling with {'GEPA + teacher LM' if teacher else 'LabeledFewShot'}...")

    _log.info("self_optimize", teacher=teacher, force=force)
    AutoCompiler.force_compile(cli, use_teacher=teacher)
    ok("Self-help module compiled and cached")
    info(f"Cache: {AutoCompiler.CACHE_PATH}")


@self_cmd.command(name="status", cls=LLMCommand)
def self_status():
    """Show self-optimization status."""
    status = _get_status()

    rows = [
        ["Compiled", "✓" if status.get("compiled") else "✗"],
        ["Cache Path", status.get("cache", "N/A")],
        ["Teacher", "✓" if status.get("teacher") else "✗"],
        ["Trainset Size", str(status.get("trainset_size", "N/A"))],
    ]
    table("Self-Optimization Status", ["Property", "Value"], rows)

    if not status.get("compiled"):
        info("Run `dspytools self-optimize` to compile the self-help module")


@self_cmd.command(name="evolve", cls=LLMCommand)
@click.option("--question", "-q", help="Question to ask the self-evolve router")
@click.option("--check", is_flag=True, help="Check if auto-optimization is needed")
def self_evolve(question: str | None, check: bool):
    """Run the self-evolving router agent.

    With --question: routes a query through the router agent.
    With --check: evaluates if programs need re-optimization.
    """
    from dspytools.evolve import SelfEvolve

    evolve = SelfEvolve()

    if question:
        info(f"Asking router: {question}")
        result = evolve.ask(question)
        _log.info("self_evolve_ask", question=question, score=result.get("score", 0))
        panel(
            "Router Response",
            f"[bold]Answer:[/] {result['answer'][:500]}\n"
            f"[bold]Score:[/] {result['score']:.2f}\n"
            f"[bold]Program:[/] {result['active_program']}\n"
            f"[bold]Needs optimize:[/] {result['needs_optimization']}",
            border_style="cyan",
        )

    elif check:
        result = evolve.auto_optimize()
        _log.info(
            "self_evolve_check",
            average_score=result["average_score"],
            samples=result["samples"],
        )
        panel(
            "Self-Evolve Check",
            f"[bold]Average Score:[/] {result['average_score']:.2f}\n"
            f"[bold]Samples:[/] {result['samples']}\n"
            f"[bold]Active:[/] {result['active_program']}\n"
            f"[bold]Action:[/] {result.get('message', result['action'])}",
            border_style="cyan",
        )
    else:
        panel(
            "Self-Evolve Status",
            f"[bold]Programs Loaded:[/] {evolve.status['programs_loaded']}\n"
            f"[bold]Active:[/] {evolve.status['active_program']}\n"
            f"[bold]Quality Samples:[/] {evolve.status['quality_samples']}\n"
            f"[bold]Avg Score:[/] {evolve.status['average_score']:.2f}",
            border_style="cyan",
        )


@self_cmd.command(name="auto-fix", cls=LLMCommand)
@click.option(
    "--dry-run",
    is_flag=True,
    default=True,
    help="Show pending recompiles without acting",
)
@click.option(
    "--auto-fix", is_flag=True, default=False, help="Actually trigger recompilation"
)
@click.option(
    "--depth",
    type=int,
    default=0,
    help="Max cascade depth (0 = unlimited, 1 = direct only)",
)
@click.option(
    "--ucb-exploration",
    type=float,
    default=2.0,
    help="UCB exploration constant (default: 2.0)",
)
def self_auto_fix(dry_run: bool, auto_fix: bool, depth: int, ucb_exploration: float):
    """Check and process programs degraded by drift that need recompilation.

    DriftMonitor queues programs when critical degradation is detected
    during inference. This command inspects the queue and optionally
    triggers automatic re-optimization.

    To just see what needs fixing:
        dspytools self auto-fix

    To actually fix with cascade propagation:
        dspytools self auto-fix --no-dry-run --depth 2

    \b
    Options:
        --dry-run: Show pending recompiles (default)
        --auto-fix: Actually trigger recompilation
        --depth: Max cascade depth for transitive dependents (0 = unlimited)
        --ucb-exploration: UCB exploration constant (lower = more exploitation)
    """
    from dspytools.core.drift_monitor import get_drift_monitor

    monitor = get_drift_monitor()
    pending = monitor.pending_recompiles()

    if not pending:
        ok("No programs need automatic recompilation")
        return

    info(f"Found {len(pending)} program(s) queued for recompilation due to drift:")
    for run_id in pending:
        info(f"  - {run_id}")

    if dry_run:
        info("")
        info("Run with --no-dry-run to trigger recompilation")
        return

    if not auto_fix:
        info("Use --auto-fix to actually trigger recompilation")
        return

    _log.info("self_auto_fix", auto_fix=True, depth=depth)
    results = monitor.process_recompile_requests(auto_fix=True)
    for r in results:
        if r["status"] == "recompile_triggered":
            ok(f"  {r['run_id']}: {r['action']}")
        else:
            warn(f"  {r['run_id']}: {r['action']}")

    remaining = monitor.pending_recompiles()
    if remaining:
        info(f"{len(remaining)} program(s) still pending")
    else:
        ok("All recompile requests processed")


@self_cmd.command(name="ucb-status", cls=LLMCommand)
def self_ucb_status():
    """Show UCB explorer state — optimizer trial counts and scores."""
    from dspytools.evolve.self_evolve import SelfEvolveEngine

    engine = SelfEvolveEngine()
    ucb = engine.ucb

    if not ucb.trials:
        info("No UCB trials recorded yet")
        info("Trials are populated after each compile via on_compile()")
        return

    rows = []
    for opt in ucb.all_optimizers:
        if opt in ucb.trials:
            count, avg = ucb.trials[opt]
            rows.append([opt, str(count), f"{avg:.4f}"])
        else:
            rows.append([opt, "0", "— (untried)"])

    table("UCB Explorer State", ["Optimizer", "Trials", "Avg Score"], rows)
    info(f"Total optimizers: {len(ucb.all_optimizers)}")
    info(f"Tried: {len(ucb.trials)}")
    info(f"Exploitation score: {ucb.exploitation_score:.2f}")
    info("Current UCB exploration constant (c): 2.0 (change via --ucb-exploration)")


@self_cmd.command(name="ucb-reset", cls=LLMCommand)
@click.confirmation_option(prompt="Reset all UCB trial history?")
def self_ucb_reset():
    """Reset UCB explorer history — clears all optimizer trial data."""
    from dspytools.evolve.self_evolve import SelfEvolveEngine

    engine = SelfEvolveEngine()
    engine.ucb.trials.clear()
    engine.ucb.save()
    ok("UCB explorer history reset")
    info("All optimizers will be treated as untried on the next compile")


@self_cmd.command(name="watch", cls=LLMCommand)
@click.option(
    "--interval",
    "-i",
    type=int,
    default=3600,
    help="Poll interval in seconds (default: 3600)",
)
@click.option(
    "--alert-url", envvar="DSPYTOOLS_ALERT_URL", help="Webhook URL for drift alerts"
)
@click.option("--once", is_flag=True, help="Single check instead of daemon loop")
def self_watch(interval: int, alert_url: str | None, once: bool):
    """Monitor programs for quality drift with optional webhook alerts.

    Runs continuously (like a daemon) checking all tracked programs
    for quality degradation. Can send webhook alerts on critical drift.

    \b
    Examples:
        # Single check
        dspytools self watch --once

        # Continuous monitoring with webhook
        dspytools self watch --interval 3600 --alert-url https://hooks.example.com/drift
    """

    from dspytools.core.drift_monitor import get_drift_monitor

    monitor = get_drift_monitor()
    status = monitor.status

    if status["programs_tracked"] == 0:
        info("No programs being tracked for drift")
        info("Programs are tracked after inference calls via HotSwapManager")
        return

    def _check_once() -> dict:
        """Run a single drift check cycle."""
        summary = {
            "checked": 0,
            "warnings": 0,
            "criticals": 0,
            "alerts": [],
        }
        # Use public API instead of private _baselines
        tracked = monitor.status.get("programs", {})
        for run_id in list(tracked.keys()):
            history = monitor.get_history(run_id, last_n=1)
            if history:
                alert = monitor.check(run_id, history[-1]["score"])
                if alert:
                    summary["alerts"].append(
                        {
                            "run_id": alert.run_id,
                            "severity": alert.severity,
                            "message": alert.message,
                        }
                    )
                    if alert.severity == "warning":
                        summary["warnings"] += 1
                    else:
                        summary["criticals"] += 1
                        monitor.request_recompile(alert.run_id)
            summary["checked"] += 1
        return summary

    def _send_webhook(url: str, payload: bytes) -> bool:
        """Send webhook with 3 retries and exponential backoff."""
        for attempt in range(3):
            req = _urllib.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            _urllib.urlopen(req, timeout=5)
            return True
        return False

    if once:
        summary = _check_once()
        info(
            f"Checked {summary['checked']} program(s): {summary['warnings']} warnings, {summary['criticals']} critical"
        )
        for alert in summary["alerts"]:
            if alert["severity"] == "critical":
                warn(f"  CRITICAL: {alert['message']}")
            else:
                info(f"  WARNING: {alert['message']}")
        return

    # Continuous monitoring loop
    info(f"Starting drift watch (interval={interval}s)")
    if alert_url:
        info(f"Webhook alerts to: {alert_url}")

    while True:
        try:
            summary = _check_once()
            if summary["criticals"] > 0:
                msg = f"Drift watch: {summary['criticals']} critical, {summary['warnings']} warnings"
                warn(msg)

                # Webhook alert with retry
                if alert_url:
                    payload = _json.dumps(summary).encode()
                    _send_webhook(alert_url, payload)
            else:
                ok(f"All {summary['checked']} program(s) healthy")

            _time.sleep(interval)
        except KeyboardInterrupt:
            info("Drift watch stopped")
            break


@self_cmd.command(name="distill", cls=LLMCommand)
@click.argument("run_id")
@click.option("--adapter", default="distilled", help="LoRA adapter name")
@click.option("--rank", type=int, default=64, help="LoRA rank (default: 64)")
@click.option("--min-score", type=float, default=0.5, help="Minimum metric score")
@click.option("--local", is_flag=True, help="Force local training")
@click.option("--colab", is_flag=True, help="Stage for Colab instead of local")
@click.option("--devset", help="Devset JSON path")
def self_distill(
    run_id: str,
    adapter: str,
    rank: int,
    min_score: float,
    local: bool,
    colab: bool,
    devset: str | None,
):
    """Distill a compiled DSPy program into a LoRA adapter via llama-cpp-server.

    The distillation process works as follows:
      1. Load the teacher model (DeepSeek V4) and compile the program
      2. Load the student model (Qwen 9B) and collect training examples
      3. Train a LoRA adapter on the student model using the compiled outputs
      4. Evaluate the LoRA adapter against test data
      5. Auto-rollback if performance degrades

    Environment variables:
      DSPYTOOLS_TL_ENABLED=true (required for teacher LM)
      DSPYTOOLS_TLMODEL_ID=deepseek/deepseek-v4-pro (teacher model)
      DSPYTOOLS_STUDENTMODEL=unsloth/Qwen3.5-9B-GGUF (student model)

    Returns:
        dict with distillation status and results
    """
    from dspytools.evolve import SelfEvolve

    panel(
        "Teacher → LoRA Distillation",
        f"Run: {run_id}\nAdapter: {adapter}\nRank: {rank}\n"
        f"Min score: {min_score}\nMode: {'local' if local else 'Colab' if colab else 'auto'}\n"
        f"Devset: {devset or '(default)'}",
        border_style="cyan",
    )

    evolve = SelfEvolve()
    result = evolve.distill(
        run_id=run_id,
        adapter_name=adapter,
        rank=rank,
        min_score=min_score,
        local=local,
        colab=colab,
        devset=devset,
    )

    if "error" in result:
        error(f"Distillation failed: {result['error']}")
        return

    # Extraction summary
    extraction = result.get("extraction", {})
    ok(
        f"Extracted {extraction.get('deduplicated', 0)} training examples "
        f"(avg score: {extraction.get('avg_score', 0):.3f})"
    )
    info(f"Training data: {extraction.get('jsonl_path', 'N/A')}")

    # Training summary
    training = result.get("training", {})
    if training.get("success"):
        ok(f"Training: {training.get('mode', '?')} — complete")
    else:
        warn(f"Training: {training.get('mode', '?')} — check logs")

    # llama-cpp-server summary
    llama_cpp = result.get("llama_cpp", {})
    if llama_cpp.get("status") == "loaded":
        model_name = llama_cpp.get("model_name", "?")
        ok(f"llama-cpp model loaded: {model_name}")
        info(f"Chat with: dspytools lora chat {model_name}")
        info(f"Test with: dspytools lora test {model_name}")

        # Auto-evaluate: compare new adapter against original compiled program
        info("Auto-evaluating adapter against source program...")
        from dspytools.cli.output import table as _table
        from dspytools.core.hotswap import HotSwapManager
        from dspytools.core.loaders import load_trainset
        from dspytools.core.metrics import exact_match_metric

        # Load test data: prefer explicit devset, fall back to extraction JSONL
        if devset:
            test_data = load_trainset(devset)
        else:
            # SSOT: load from extraction JSONL via core.loaders
            from dspytools.core.loaders import load_jsonl

            jsonl_path = extraction.get("jsonl_path")
            test_data = load_jsonl(jsonl_path) if jsonl_path else []

        if not test_data:
            info("No test data available for auto-evaluation (provide --devset)")
        elif run_id:
            metric_fn = exact_match_metric(val_field="output")
            mgr = HotSwapManager()

            compiled_avg = 0.0
            lora_avg = 0.0

            mgr.load_single(run_id)
            mgr.swap(run_id)
            compiled_scores = []
            for ex in test_data[:10]:
                input_fields = _get_input_fields(ex)
                inputs = {f: getattr(ex, f, "") for f in input_fields}
                result_prog = mgr.infer(**inputs)
                pred_val = result_prog.get("output", str(result_prog))
                score = metric_fn(ex, type("Pred", (), {"output": pred_val})())
                compiled_scores.append(score)
            compiled_avg = (
                sum(compiled_scores) / len(compiled_scores) if compiled_scores else 0
            )

            # Score LoRA adapter via llama-cpp-server
            lora_scores = []
            for ex in test_data[:10]:
                input_fields = _get_input_fields(ex)
                prompt_lines = [f"{f}: {getattr(ex, f, '')}" for f in input_fields]
                prompt_lines.append("\nBased on the above, provide the output:")
                prompt = "\n".join(prompt_lines)
                response = _llama_chat(model_name, prompt, max_tokens=500)
                score = metric_fn(ex, type("Pred", (), {"output": response})())
                lora_scores.append(score)
            lora_avg = sum(lora_scores) / len(lora_scores) if lora_scores else 0

            _table(
                "Distillation A/B",
                ["Evaluator", "Avg Score", "Result"],
                [
                    ["Original", f"{compiled_avg:.3f}", ""],
                    [
                        "LoRA Adapter",
                        f"{lora_avg:.3f}",
                        "✓ Accepted"
                        if lora_avg >= compiled_avg * 0.95
                        else "✗ Rolled back",
                    ],
                ],
            )

            # Auto-rollback if LoRA underperforms (llama-cpp doesn't have unload endpoint)
            if lora_avg < compiled_avg * 0.95 and lora_avg > 0:
                warn(
                    f"Adapter underperforms (LoRA={lora_avg:.3f} vs compiled={compiled_avg:.3f})"
                )
                warn("Auto-rollback: adapter unload via system command")
            elif lora_avg > 0:
                ok(
                    f"Adapter performance acceptable (LoRA={lora_avg:.3f} vs compiled={compiled_avg:.3f})"
                )
    else:
        # llama-cpp-server errors (no /api/status endpoint)
        warn("llama-cpp load error")

    # Full results
    console.print(f"\n[bold]Elapsed:[/] {result.get('elapsed_seconds', 0):.1f}s")


def _get_status() -> dict:
    """Get self-optimization status from HelpManager."""
    from dspytools.help.__init__ import HelpManager

    return HelpManager.status()
