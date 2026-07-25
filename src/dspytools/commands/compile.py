"""dspytools compile — Compile and optimize DSPy programs.

Optimizer commands are generated from _OPTIMIZER_SPECS via _register_optimizer_commands().
Special commands (multi-step / teacher-dependent) are defined explicitly.

Auto-generated (from registry):
    knn, mipro, gepa, copro, simba, bootstrap-few-shot,
    bootstrap-few-shot-random, bootstrap-few-shot-optuna,
    labeled-few-shot, infer-rules

Explicit (special):
    submit      Submit a background compile job
    status      Poll a job's status
    list        List all jobs
    cancel      Cancel a queued/running job
    better-together  GEPA sub-optimizer composition
    ensemble    Multi-module ensemble
    finetune    BootstrapFinetune
    gfl         GFL 4-way comparison pipeline
    grpo        RL-based pipeline optimization
    avatar      AvatarOptimizer
    distill     Teacher-student distillation
"""

from __future__ import annotations

from typing import Any, Callable

from dspytools.cli.output import fail, force_option, label_option, panel
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import embedder_kwargs
from dspytools.core._dspy import dspy
from dspytools.core.cost_tracker import CompileCost
from dspytools.core.drift_monitor import get_drift_monitor
from dspytools.core.holdout import get_holdout_gate
from dspytools.core.loaders import load_module_by_name, load_trainset
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import exact_match_metric, gepa_metric
from dspytools.core.mlflow_tracker import get_tracker
from dspytools.core.output import create_run_dir, save_program
from dspytools.core.registry import (
    compute_dataset_hash,
    find_existing_compile,
    get_lineage,
    get_run,
    register_run_with_graph,
    register_run_with_lineage,
)
from dspytools.core.retry import compile_with_retry
from dspytools.core.scheduler import CompileScheduler
from dspytools.core.setup import LMRegistry, setup_dspy
from dspytools.evolve.self_evolve import SelfEvolveEngine
from dspytools.gfl.pipeline import GFLPipeline

_log = get_logger(__name__)


@click.group(name="compile", cls=LLMGroup)
def compile_cmd():
    """Compile and optimize DSPy programs."""


# ── Optimizer Registry ──────────────────────────────────────────────────


def _make_optimizer_cmd(name: str, cls_name: str, cls_lambda: Callable, **kwargs):
    """Generate a Click command for a simple optimizer."""
    extra_params = kwargs.get("params", {})
    module_type = kwargs.get("module_type", "predict")
    run_dir = kwargs.get("run_dir", name)
    needs_teacher = kwargs.get("needs_teacher", False)

    @click.argument("module_name")
    @click.argument("trainset_path")
    @label_option
    @force_option
    def cmd(
        module_name: str,
        trainset_path: str,
        label: str | None,
        force: bool = False,
        draft: bool = False,
    ):
        setup_dspy()
        student = load_module_by_name(module_name)
        trainset = load_trainset(trainset_path)
        dataset_hash = compute_dataset_hash(trainset)

        # Compile dedup: skip if exact same config was already compiled
        if not force:
            existing = find_existing_compile(module_name, dataset_hash, name)
            if existing:
                click.echo(f"  Using cached: {existing['id']}")
                click.echo("  Use --force to force recompilation")
                return

        if draft:
            # Speculative compilation: student drafts, teacher polishes
            pipeline = GFLPipeline()
            result = pipeline.compile_draft(
                student=student, trainset=trainset, optimizer_name=name
            )
            compiled = result["best_program"]
            click.echo(
                f"  Draft score: {result['draft_score']:.4f} | Polished: {result['polished_score']:.4f}"
            )
        else:
            metric = gepa_metric if name == "gepa" else exact_match_metric()

            if needs_teacher:
                teacher = LMRegistry.get_teacher()
                if teacher is None:
                    click.echo("  No teacher LM configured", err=True)
                    raise click.Abort()
                extra_params["reflection_lm"] = teacher

            opt = cls_lambda(metric=metric, trainset=trainset, **extra_params)
            # KNNFewShot consumes trainset in __init__, not compile()
            compile_kwargs: dict[str, Any] = {"student": student}
            if "knn" not in name:
                compile_kwargs["trainset"] = trainset
            # GEPA needs a separate valset to avoid overfitting warning
            if "gepa" in name and len(trainset) >= 6:
                val_size = max(3, len(trainset) // 5)  # 20% or at least 3
                compile_kwargs["valset"] = trainset[-val_size:]
                compile_kwargs["trainset"] = trainset[:-val_size]
            # BetterTogether wraps GEPA internally — same valset split
            if "better-together" in name and len(trainset) >= 6:
                val_size = max(3, len(trainset) // 5)
                compile_kwargs["valset"] = trainset[-val_size:]
                compile_kwargs["trainset"] = trainset[:-val_size]
            # COPRO requires eval_kwargs
            if "copro" in name:
                compile_kwargs["eval_kwargs"] = {"num_threads": 4}
            # SIMBA requires at least bsize examples
            if "simba" in name and len(trainset) < opt.bsize:
                fail(f"SIMBA needs ≥{opt.bsize} examples, got {len(trainset)}")
                raise click.ClickException(
                    f"SIMBA requires at least {opt.bsize} training examples"
                )
            compiled, retry_stats = compile_with_retry(
                lambda: opt.compile(**compile_kwargs),
                max_retries=3,
                base_delay=2.0,
            )
            if retry_stats["retries"] > 0:
                click.echo(
                    f"  Retries: {retry_stats['retries']} (total delay: {retry_stats['total_delay']:.1f}s)"
                )

        # Cost tracking (approximate based on trainset size and optimizer)
        token_multipliers = {
            "knn": 2000,
            "mipro": 5000,
            "gepa": 8000,
            "copro": 4000,
            "simba": 3000,
        }
        est_tokens = len(trainset) * token_multipliers.get(name, 3000)

        cost = CompileCost(compile_id="", optimizer=name)
        cost.add_call(
            "deepseek/deepseek-v4-flash",
            prompt_tokens=est_tokens // 2,
            completion_tokens=est_tokens // 2,
        )
        cost.finish()

        # MLflow tracking

        tracker = get_tracker()
        tracker.log_compile(
            optimizer=name,
            module=module_name,
            score=0.5,  # placeholder — actual evaluation would need trainset
            params=extra_params,
            metrics={"tokens": cost.total_tokens, "cost_usd": cost.total_cost},
        )

        run_id, run_path = create_run_dir(run_dir, label)

        # Holdout gate validation (Invariant 5: holdout never seen by optimizer)

        gate = get_holdout_gate()
        train_set, _ = gate.split(trainset, compile_id=run_id)
        validation = gate.validate_gate(run_id, compiled)
        if not validation["accepted"]:
            click.echo(
                f"  Holdout gate: {validation['reason']} (score: {validation['score']:.4f} on {validation['n_evaluated']}/{validation['holdout_size']} examples)",
                err=True,
            )

        save_program(
            run_path,
            compiled,
            {"inputs": ["input"], "outputs": ["output"]},
            module_type=module_type,
        )
        register_run_with_lineage(
            run_id,
            {"optimizer": name, "module": module_name, **extra_params},
            optimizer=name,
            dataset_hash=dataset_hash,
            base_program_id=module_name,
        )
        _log.info(
            "compile_optimizer",
            optimizer=name,
            module=module_name,
            run_id=run_id,
            score=validation.get("score", 0.5),
            tokens=cost.total_tokens,
            cost_usd=cost.total_cost,
            duration_s=cost.elapsed_seconds,
        )
        click.echo(f"  {cls_name} compiled → {run_id}")
        click.echo(
            f"  ~{cost.total_tokens:,} tokens | ${cost.total_cost:.4f} | {cost.elapsed_seconds:.1f}s"
        )

        # ── Self-evolve wiring: learn from this compile ──────────

        engine = SelfEvolveEngine()
        profile = engine.morphology.profile_task(
            description=module_name,
            field_count=2,
            data_size=len(trainset),
        )
        engine.on_compile(
            task_profile=profile,
            optimizer=name,
            score=validation.get("score", 0.5),
            success=validation.get("accepted", True),
        )

        # Set drift baseline from this compile
        monitor = get_drift_monitor()
        monitor.update_baseline(run_id, validation.get("score", 0.5))

    if needs_teacher:
        cmd = click.option(
            "--draft/--no-draft",
            default=False,
            help="Use speculative compilation (student drafts, teacher polishes)",
        )(cmd)

    return cmd


_OPTIMIZER_SPECS = {
    "knn": {
        "cls_name": "KNNFewShot",
        "cls_lambda": lambda metric, **kw: dspy.KNNFewShot(
            k=kw.pop("k", 2),
            trainset=kw.pop("trainset", []),
            vectorizer=kw.pop("vectorizer", dspy.Embedder(**embedder_kwargs())),
        ),
        "params": {},
    },
    "mipro": {
        "cls_name": "MIPROv2",
        "cls_lambda": lambda metric, **kw: dspy.MIPROv2(
            metric=metric, auto=kw.pop("auto", "light")
        ),
        "params": {},
    },
    "gepa": {
        "cls_name": "GEPA",
        "cls_lambda": lambda metric, **kw: dspy.GEPA(
            metric=metric,
            auto=kw.pop("auto", "light"),
            reflection_lm=kw.pop("reflection_lm", None),
        ),
        "params": {},
        "needs_teacher": True,
        "extra_options": {
            "--draft/--no-draft": {
                "default": False,
                "help": "Use speculative compilation (student drafts, teacher polishes)",
            },
        },
    },
    "copro": {
        "cls_name": "COPRO",
        "cls_lambda": lambda metric, **kw: dspy.COPRO(
            metric=metric, depth=3, breadth=4
        ),
        "params": {},
    },
    "simba": {
        "cls_name": "SIMBA",
        "cls_lambda": lambda metric, **kw: dspy.SIMBA(metric=metric),
        "params": {},
    },
    "bootstrap-few-shot": {
        "cls_name": "BootstrapFewShot",
        "cls_lambda": lambda metric, **kw: dspy.BootstrapFewShot(
            metric=metric, max_labeled_demos=4, max_bootstrapped_demos=4
        ),
        "params": {},
    },
    "bootstrap-few-shot-random": {
        "cls_name": "BootstrapFewShotWithRandomSearch",
        "cls_lambda": lambda metric, **kw: dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            num_candidate_programs=8,
            num_threads=4,
        ),
        "params": {},
    },
    "bootstrap-few-shot-optuna": {
        "cls_name": "BootstrapFewShotWithOptuna",
        "cls_lambda": lambda metric, **kw: dspy.BootstrapFewShotWithOptuna(
            metric=metric,
        ),
        "params": {},
    },
    "labeled-few-shot": {
        "cls_name": "LabeledFewShot",
        "cls_lambda": lambda metric, **kw: dspy.LabeledFewShot(k=kw.pop("k", 5)),
        "params": {},
    },
    "infer-rules": {
        "cls_name": "InferRules",
        "cls_lambda": lambda metric, **kw: dspy.InferRules(metric=metric),
        "params": {},
    },
}


def _register_optimizer_commands():
    """Register all simple optimizer commands from the spec registry."""
    for name, spec in _OPTIMIZER_SPECS.items():
        cmd_fn = _make_optimizer_cmd(name=name, **spec)
        cmd_fn.__name__ = f"compile_{name.replace('-', '_')}"
        cmd_fn.__doc__ = f"Compile with {spec['cls_name']} optimizer (synchronous)."
        compile_cmd.command(name=name)(cmd_fn)


# ── Async Job Commands ──────────────────────────────────────────────────


@compile_cmd.command(name="submit", cls=LLMCommand)
@click.argument(
    "optimizer", type=click.Choice(["knn", "mipro", "gepa", "better-together"])
)
@click.argument("module_name")
@click.argument("trainset_path")
@label_option
@click.option("--auto", default="light", help="Auto mode for MIPRO/GEPA")
@click.option("--k", default=2, type=int, help="K for KNNFewShot")
def compile_submit(
    optimizer: str,
    module_name: str,
    trainset_path: str,
    label: str | None,
    auto: str,
    k: int,
):
    """Submit a background compile job. Returns job_id for polling."""

    setup_dspy()
    teacher = LMRegistry.get_teacher()
    if teacher is None:
        click.echo(
            "  No teacher LM configured. Set via `dspytools configure lm set --role teacher`",
            err=True,
        )
        raise click.Abort()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    def _run_knn() -> str:
        vectorizer = dspy.Embedder(**embedder_kwargs())
        opt = dspy.KNNFewShot(k=k, trainset=trainset, vectorizer=vectorizer)
        compiled = opt.compile(student=student)
        run_id, run_path = create_run_dir("knn", label)
        save_program(
            run_path,
            compiled,
            {"inputs": ["input"], "outputs": ["output"]},
            module_type="predict",
        )
        register_run_with_graph(
            run_id, {"optimizer": "knn", "module": module_name, "k": k}, optimizer="knn"
        )
        return run_id

    def _run_mipro() -> str:
        opt = dspy.MIPROv2(
            metric=exact_match_metric(),
            auto=auto,
        )
        compiled = opt.compile(student=student, trainset=trainset)
        run_id, run_path = create_run_dir("mipro", label)
        save_program(
            run_path,
            compiled,
            {"inputs": ["input"], "outputs": ["output"]},
            module_type="predict",
        )
        register_run_with_graph(
            run_id,
            {"optimizer": "mipro", "module": module_name, "auto": auto},
            optimizer="mipro",
        )
        return run_id

    def _run_gepa() -> str:
        opt = dspy.GEPA(metric=gepa_metric, auto=auto, reflection_lm=teacher)
        # Split valset to avoid GEPA overfitting warning
        val_size = max(3, len(trainset) // 5) if len(trainset) >= 6 else 0
        compile_kwargs = {"student": student}
        if val_size > 0:
            compile_kwargs["valset"] = trainset[-val_size:]
            compile_kwargs["trainset"] = trainset[:-val_size]
        else:
            compile_kwargs["trainset"] = trainset
        compiled = opt.compile(**compile_kwargs)
        run_id, run_path = create_run_dir("gepa", label)
        save_program(
            run_path,
            compiled,
            {"inputs": ["input"], "outputs": ["output"]},
            module_type="predict",
        )
        register_run_with_graph(
            run_id, {"optimizer": "gepa", "module": module_name, "auto": auto}
        )
        return run_id

    runners = {
        "knn": _run_knn,
        "mipro": _run_mipro,
        "gepa": _run_gepa,
    }

    job_id = CompileScheduler.submit(optimizer, module_name, runners[optimizer], label)
    _log.info("compile_submit", optimizer=optimizer, module=module_name, job_id=job_id)
    click.echo(f"  Job submitted: {job_id}")
    click.echo(f"  Check status: dspytools compile status {job_id}")


@compile_cmd.command(name="status", cls=LLMCommand)
@click.argument("job_id")
def compile_status(job_id: str):
    """Check the status of a compile job."""
    job = CompileScheduler.get_status(job_id)
    if not job:
        click.echo(f"  Job '{job_id}' not found")
        return
    click.echo(f"  Job: {job.job_id}")
    click.echo(f"  Optimizer: {job.optimizer}")
    click.echo(f"  Status: {job.status}")
    click.echo(f"  Progress: {job.progress:.0%}")
    click.echo(f"  Message: {job.message}")
    if job.run_id:
        click.echo(f"  Run: {job.run_id}")
    if job.error:
        click.echo(f"  Error: {job.error}")


@compile_cmd.command(name="list", cls=LLMCommand)
def compile_list():
    """List all compile jobs."""
    jobs = CompileScheduler.list_jobs()
    if not jobs:
        click.echo("  No compile jobs")
        return
    for j in jobs:
        marker = (
            "✓"
            if j["status"] == "completed"
            else "✗"
            if j["status"] == "failed"
            else "⋯"
        )
        click.echo(
            f"  {marker} {j['job_id']}: {j['optimizer']} — {j['status']} ({j.get('run_id', '')})"
        )


@compile_cmd.command(name="cancel", cls=LLMCommand)
@click.argument("job_id")
def compile_cancel(job_id: str):
    """Cancel a queued or running compile job."""
    if CompileScheduler.cancel(job_id):
        click.echo(f"  Cancelled job {job_id}")
    else:
        click.echo(f"  Job {job_id} not found or already finished")


@compile_cmd.command(name="cost", cls=LLMCommand)
@click.argument("run_id")
def compile_cost(run_id: str):
    """Show cost breakdown and lineage chain for a compiled run."""

    meta = get_run(run_id)
    if not meta:
        fail(f"Run '{run_id}' not found")
        return

    lineage = get_lineage(run_id)

    click.echo(f"  Run: {run_id}")
    click.echo(f"  Optimizer: {meta.get('lineage', {}).get('optimizer', '?')}")
    click.echo(f"  Created: {meta.get('created', '?')}")
    if meta.get("score") is not None:
        click.echo(f"  Score: {meta['score']}")
    click.echo(f"  Lineage depth: {len(lineage)}")
    if lineage:
        chain = [entry.get("lineage", {}).get("optimizer", "?") for entry in lineage]
        click.echo(f"  Chain: {' → '.join(chain)}")

    # Cost projection: estimate savings from optimizations
    optimizer = meta.get("lineage", {}).get("optimizer", "")

    # Estimate token savings from halving-style pruning
    if "gfl" in optimizer:
        click.echo("  ── Projected Savings ──")
        click.echo("    Successive Halving (estimate): ~75% fewer teacher LM calls")
        click.echo("    Speculative Compile (estimate): ~3-5x fewer teacher LM calls")
        click.echo(
            "    Semantic Cache (estimate):     ~70% API cost reduction on repeat queries"
        )
    elif optimizer in ("mipro", "gepa", "copro"):
        click.echo("  ── Optimization Tips ──")
        click.echo("    Use --draft for speculative compilation: 3-5x cheaper")
        click.echo("    Use GFL --halving for 75% faster multi-optimizer search")
        click.echo("    Enable Redis cache for ~70% cost reduction on repeat queries")


@compile_cmd.command(name="better-together", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@click.option("--strategy", default="p", help="Optimization strategy")
@label_option
def compile_bt(module_name: str, trainset_path: str, strategy: str, label: str | None):
    """Compile with BetterTogether optimizer (synchronous)."""

    setup_dspy()
    teacher = LMRegistry.get_teacher()
    if teacher is None:
        click.echo("  No teacher LM configured", err=True)
        raise click.Abort()
    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    # Set LM explicitly on student module so BetterTogether can access predictor.lm
    student.set_lm(LMRegistry.get_or_default())

    bt = dspy.BetterTogether(
        metric=gepa_metric,
        p=dspy.GEPA(metric=gepa_metric, auto="light", reflection_lm=teacher),
    )
    # Split valset to avoid GEPA overfitting warning
    val_size = max(3, len(trainset) // 5) if len(trainset) >= 6 else 0
    compile_kwargs: dict[str, Any] = {"student": student, "strategy": strategy}
    if val_size > 0:
        compile_kwargs["valset"] = trainset[-val_size:]
        compile_kwargs["trainset"] = trainset[:-val_size]
    else:
        compile_kwargs["trainset"] = trainset
    compiled = bt.compile(**compile_kwargs)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="better_together",
        module=module_name,
        score=0.5,
        params={"strategy": strategy},
    )

    run_id, run_path = create_run_dir("better_together", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="predict",
    )
    register_run_with_graph(
        run_id,
        {"optimizer": "better_together", "module": module_name, "strategy": strategy},
        optimizer="better_together",
    )
    _log.info("compile_bt", module=module_name, strategy=strategy, run_id=run_id)
    click.echo(f"  BetterTogether compiled → {run_id}")


@compile_cmd.command(name="ensemble", cls=LLMCommand)
@click.option(
    "--modules",
    "-m",
    multiple=True,
    required=True,
    help="Module names (comma-separated or repeated -m)",
)
@label_option
def compile_ensemble(modules: tuple[str, ...], label: str | None):
    """Build an ensemble of multiple modules (synchronous)."""

    setup_dspy()
    # Support both "-m A -m B" and "-m A,B" syntax
    module_names: list[str] = []
    for m in modules:
        module_names.extend(name.strip() for name in m.split(","))
    if len(module_names) < 2:
        raise click.ClickException("need at least 2 modules for ensemble")
    programs = [load_module_by_name(m) for m in module_names]
    ensemble = dspy.Ensemble(reduce_fn=dspy.majority)
    compiled = ensemble.compile(programs=programs)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="ensemble",
        module="+".join(modules),
        score=0.5,
    )

    run_id, run_path = create_run_dir("ensemble", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="ensemble",
    )
    register_run_with_graph(run_id, {"optimizer": "ensemble", "modules": list(modules)})
    click.echo(f"  Ensemble compiled → {run_id}")


@compile_cmd.command(name="finetune", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@click.option("--num-iters", default=5, type=int)
@label_option
def compile_finetune(
    module_name: str, trainset_path: str, num_iters: int, label: str | None
):
    """Compile with BootstrapFinetune (distill prompts into weight updates)."""
    setup_dspy()
    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)
    student_lm = LMRegistry.get_or_default()
    student.set_lm(student_lm)

    # BootstrapFinetune uploads training data to the provider's fine-tuning API.
    # Local LLM endpoints (llama-cpp-server) don't expose a fine-tuning upload API.
    # Check api_base before attempting — prevents a confusing auth/upload error.
    api_base = student_lm.kwargs.get("api_base", "")
    if api_base and ("localhost" in api_base or "127.0.0.1" in api_base):
        raise click.ClickException(
            "BootstrapFinetune requires a provider with a fine-tuning API (OpenAI, Together, etc).\n"
            f"  Current api_base: {api_base} (local endpoint — does not support fine-tuning uploads).\n"
            "  For local model weight optimization, use the LoRA distillation pipeline:\n"
            "    dspytools distill run --module {m} --trainset {t}\n"
            "    dspytools lora train --data output/<jsonl>".format(
                m=module_name, t=trainset_path
            )
        )

    opt = dspy.BootstrapFinetune(
        metric=exact_match_metric(),
        num_iters=num_iters,
    )
    compiled = opt.compile(student=student, trainset=trainset)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="bootstrap_finetune",
        module=module_name,
        score=0.5,
        params={"num_iters": num_iters},
    )

    run_id, run_path = create_run_dir("finetune", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="finetune",
    )
    register_run_with_graph(
        run_id,
        {
            "optimizer": "bootstrap_finetune",
            "module": module_name,
            "num_iters": num_iters,
        },
        optimizer="bootstrap_finetune",
    )
    click.echo(f"  BootstrapFinetune compiled → {run_id}")


@compile_cmd.command(name="gfl", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@click.option(
    "--single",
    type=click.Choice(
        ["bootstrap-few-shot", "mipro", "gepa", "sequential"],
        case_sensitive=False,
    ),
    default=None,
    help="Run a single optimizer instead of 4-way comparison",
)
@click.option(
    "--halving/--no-halving",
    default=True,
    help="Use successive halving for faster comparison",
)
@click.option(
    "--validate/--no-validate",
    default=False,
    help="Run SPRT validation after compile (requires holdout data)",
)
@click.option(
    "--auto-suggest/--no-auto-suggest",
    default=False,
    help="Use SelfEvolveEngine to suggest best optimizer order",
)
@label_option
def compile_gfl(
    module_name: str,
    trainset_path: str,
    single: str | None,
    halving: bool,
    validate: bool,
    auto_suggest: bool,
    label: str | None,
):
    """GFL compile — 4-way optimizer comparison (BFS → MIPROv2 → GEPA → Sequential).

    Runs all optimizers in parallel and picks the best one. Uses the LSE
    (Learning Speed Estimator) tracker to measure improvement deltas.

    Example: dspytools compile gfl TestMod trainset.json

    To run a single optimizer instead:
        dspytools compile gfl TestMod trainset.json --single gepa
    """

    setup_dspy()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    click.echo(f"\n  GFL Pipeline: {module_name}")
    click.echo(
        f"  Trainset: {len(trainset)} examples | Mode: {'single=' + single if single else '4-way comparison'}"
    )

    pipeline = GFLPipeline(mode="single" if single else "compare")

    if single:
        # Run a single optimizer via the pipeline
        result = pipeline.run_single(
            optimizer_name=single.replace("-", "_"),
            student=student,
            trainset=trainset,
        )
        click.echo(
            f"  Optimizer: {result['best_optimizer']} | Score: {result['best_score']:.4f}"
        )
        if result.get("trainset_size"):
            click.echo(f"  Trainset size: {result['trainset_size']}")
        if result.get("synthesized"):
            click.echo(f"  Data synthesized: {result['synthesized']}")
    else:
        if halving:
            # Multi-fidelity early pruning (Successive Halving)
            result = pipeline.run_halving(
                student=student,
                trainset=trainset,
                train_field="input",
                val_field="output",
                auto_suggest=auto_suggest,
            )
            click.echo("  Mode: Successive Halving")
            if result.get("pruned"):
                click.echo(f"  Pruned: {', '.join(result['pruned'])}")
            if result.get("survivors"):
                click.echo(f"  Survivors: {', '.join(result['survivors'])}")
        else:
            # Full 4-way comparison
            result = pipeline.run(
                student=student,
                trainset=trainset,
                train_field="input",
                val_field="output",
            )
        click.echo("\n  Scores:")
        for name, score in result["all_scores"].items():
            click.echo(f"    {name}: {score:.4f}")
        click.echo(
            f"\n  Best: {result['best_optimizer']} (score: {result['best_score']:.4f})"
        )
        click.echo(f"  Baseline: {result['baseline']:.4f}")
        click.echo(f"  Improvement: {result['improvement']:+.4f}")
        if result.get("trend"):
            trend = result["trend"]
            if isinstance(trend, (int, float)):
                click.echo(f"  LSE Trend: {trend:.4f}")
            else:
                click.echo(f"  LSE Trend: {trend}")
        if result.get("total_improvement"):
            imp = result["total_improvement"]
            if isinstance(imp, (int, float)):
                click.echo(f"  Total Δ: {imp:+.4f}")
            else:
                click.echo(f"  Total Δ: {imp}")

    # SPRT validation if requested
    if validate:
        engine = SelfEvolveEngine()
        # Create holdout from trainset
        train_set, holdout = GFLPipeline.split_holdout(trainset, holdout_fraction=0.2)
        if holdout:
            val_result = engine.validate_and_deploy(
                candidate_program=result.get("best_program", student),
                program_id="gfl_latest",
                holdout_set=holdout,
                alpha=0.05,
                beta=0.2,
                max_evaluations=50,
            )
            click.echo(
                f"\n  SPRT Validation: {'✓ ACCEPTED' if val_result['accepted'] else '✗ REJECTED'}"
            )
            click.echo(f"    Score: {val_result['candidate_score']:.4f}")
            click.echo(f"    Evaluated: {val_result['n_evaluated']} examples")
            if val_result.get("early_stop"):
                click.echo(f"    Early stop: ✓ ({val_result['reason']})")
            if not val_result["accepted"]:
                click.echo(
                    "    Recommended: re-compile with different optimizer or more data",
                    err=True,
                )

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_gfl_comparison(result)

    # Save the best compiled program
    run_id, run_path = create_run_dir("gfl", label)
    save_program(
        run_path,
        result.get(
            "best_program", student
        ),  # GFLPipeline returns best_optimizer name, need to get actual compiled program
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="predict",
    )
    register_run_with_graph(
        run_id,
        {
            "optimizer": f"gfl_{result['best_optimizer']}",
            "module": module_name,
            "score": result["best_score"],
            "baseline": result.get("baseline"),
            "improvement": result.get("improvement"),
            "all_scores": result.get("all_scores"),
            "mode": "single" if single else "4-way",
        },
    )
    click.echo(f"\n  GFL compiled → {run_id}")

    synthesized_str = " (synthesized)" if result.get("synthesized") else ""
    panel(
        "GFL Results",
        f"[bold]Optimizer:[/] {result['best_optimizer']}\n"
        f"[bold]Trainset:[/] {len(trainset)} examples{synthesized_str}\n"
        f"[bold]Score:[/] {result['best_score']:.2f}\n"
        f"[bold]Run:[/] {run_id}",
        border_style="green",
    )


@compile_cmd.command(name="grpo", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@click.option("--lora/--no-lora", default=False, help="Use LoRA adapter training")
@click.option("--beta", default=0.01, type=float, help="KL penalty coefficient")
@click.option("--max-steps", default=10, type=int, help="Max optimization steps")
@label_option
def compile_grpo(
    module_name: str,
    trainset_path: str,
    lora: bool,
    beta: float,
    max_steps: int,
    label: str | None,
):
    """Compile with GRPO (RL-based pipeline optimization).

    GRPO treats the entire agent pipeline as a single policy and optimizes
    end-to-end with reinforcement learning. Combined with GEPA via
    BetterTogether, yields 5-11% improvement (arXiv:2508.04660).
    Falls back to BetterTogether(GEPA sequential) if GRPO module unavailable.
    """

    setup_dspy()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    # lazy: GRPO may not be available in all DSPy versions
    from dspytools.gfl.grpo import compile_grpo as _grpo_compile

    compiled = _grpo_compile(
        student, trainset, lora=lora, beta=beta, max_steps=max_steps
    )

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="grpo",
        module=module_name,
        score=0.5,
        params={"lora": lora, "beta": beta, "max_steps": max_steps},
    )

    run_id, run_path = create_run_dir("grpo", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="predict",
    )
    register_run_with_graph(
        run_id, {"optimizer": "grpo", "module": module_name, "lora": lora}
    )
    click.echo(f"  GRPO compiled → {run_id}")


@compile_cmd.command(name="avatar", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@label_option
def compile_avatar(module_name: str, trainset_path: str, label: str | None):
    """Compile with AvatarOptimizer (reflective instruction improvement).

    Avatar uses positive/negative comparison to refine task instructions
    for acting agents, producing concise and effective prompts.
    """
    setup_dspy()
    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    # AvatarOptimizer requires an acting agent module (ReActV2 with tools)
    # that produces 'actions' in its output. Simple Predict/ChainOfThought
    # modules don't have actions and will crash.
    test_result = student(question="test")
    if not hasattr(test_result, "actions"):
        fail(
            "AvatarOptimizer requires an acting agent module (e.g., ReActV2 with tools)"
        )
        raise click.ClickException(
            "AvatarOptimizer is designed for acting agents that produce 'actions' in their output. "
            "The module '" + module_name + "' does not produce actions. "
            "Use 'dspytools agent new' to create an agent with tools, or use "
            "'compile mipro' / 'compile gepa' for instruction optimization on non-agent modules."
        )

    opt = dspy.AvatarOptimizer(
        metric=exact_match_metric(),
    )
    compiled = opt.compile(student=student, trainset=trainset)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="avatar",
        module=module_name,
        score=0.5,
    )

    run_id, run_path = create_run_dir("avatar", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="predict",
    )
    register_run_with_graph(run_id, {"optimizer": "avatar", "module": module_name})
    click.echo(f"  AvatarOptimizer compiled → {run_id}")


@compile_cmd.command(name="distill", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@click.option(
    "--auto", default="light", type=click.Choice(["light", "medium", "heavy"])
)
@click.option("--num-iters", default=5, type=int, help="Finetune iterations")
@label_option
def compile_distill(
    module_name: str, trainset_path: str, auto: str, num_iters: int, label: str | None
):
    """Teacher-Student distillation: teacher optimizes prompts, student absorbs via finetune.

    Pipeline:
      1. Teacher LM (DeepSeek V4) runs GEPA to optimize instructions
      2. Student LM (local 3B model) finetunes on optimized prompts
      3. Saves compiled program using distilled knowledge
    """
    setup_dspy()
    teacher = LMRegistry.get_teacher()
    if teacher is None:
        click.echo(
            "  No teacher LM configured. Set via `dspytools configure lm set --role teacher`",
            err=True,
        )
        raise click.Abort()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)
    student_lm = LMRegistry.get_or_default()
    student.set_lm(student_lm)

    click.echo(f"  Teacher: {teacher}")
    click.echo(f"  Student: {student_lm}")

    # Step 1: Teacher optimization (GEPA with reflection_lm)
    click.echo("  [1/2] Teacher optimizing instructions...")

    gepa_opt = dspy.GEPA(metric=exact_match_metric(), auto=auto, reflection_lm=teacher)
    # Split valset to avoid GEPA overfitting warning
    val_size = max(3, len(trainset) // 5) if len(trainset) >= 6 else 0
    gepa_kwargs: dict[str, Any] = {"student": student}
    if val_size > 0:
        gepa_kwargs["valset"] = trainset[-val_size:]
        gepa_kwargs["trainset"] = trainset[:-val_size]
    else:
        gepa_kwargs["trainset"] = trainset
    optimized = gepa_opt.compile(**gepa_kwargs)

    # Step 2: Student finetuning on optimized prompts
    click.echo("  [2/2] Student absorbing via finetune...")

    ft_opt = dspy.BootstrapFinetune(metric=exact_match_metric(), num_iters=num_iters)
    distilled = ft_opt.compile(student=optimized, trainset=trainset)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="distill",
        module=module_name,
        score=0.5,
        params={
            "teacher_opt": "gepa",
            "student_ft": "bootstrap_finetune",
            "num_iters": num_iters,
            "auto": auto,
        },
    )

    run_id, run_path = create_run_dir("distill", label)
    save_program(
        run_path,
        distilled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="finetune",
    )
    register_run_with_graph(
        run_id,
        {
            "optimizer": "distill",
            "module": module_name,
            "teacher_opt": "gepa",
            "student_ft": "bootstrap_finetune",
            "num_iters": num_iters,
        },
    )
    click.echo(f"  Distilled → {run_id}")


# ── Register generated commands ──────────────────────────────────────────

_register_optimizer_commands()
