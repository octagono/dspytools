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

import inspect
import json
from typing import Any, Callable

from dspytools.cli.output import fail, force_option, label_option, panel, table
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.config.settings import embedder_kwargs
from dspytools.core._dspy import dspy
from dspytools.core.cost_tracker import CompileCost
from dspytools.core.drift_monitor import get_drift_monitor
from dspytools.core.holdout import get_holdout_gate
from dspytools.core.loaders import load_module_by_name, load_trainset
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import (
    exact_match_metric,
    gepa_metric,
    gepa_metric_cost_aware,
    gepa_metric_float,
)
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

# ── GEPA kwargs plumbing ────────────────────────────────────────────────
# The knob lists below are derived from the *installed* DSPy signature instead
# of being pinned to one DSPy version, so `dspytools compile` accepts every
# kwarg the running dspy.GEPA understands and keeps working across upgrades.

# Knobs promoted to first-class CLI options (only forwarded when they differ
# from dspy.GEPA's own defaults, so the CLI defaults stay behaviour-neutral).
_GEPA_KNOB_NAMES = (
    "seed",
    "use_merge",
    "component_selector",
    "perfect_score",
    "track_stats",
    "track_best_outputs",
)

# gepa.optimize kwargs that dspy.GEPA passes explicitly itself (a duplicate
# would be a TypeError) — route them through --gepa-kwarg / named options.
_GEPA_DSPY_OWNED_OPTIMIZE_KWARGS = frozenset(
    {
        "seed_candidate",
        "trainset",
        "valset",
        "adapter",
        "reflection_lm",
        "candidate_selection_strategy",
        "skip_perfect_score",
        "reflection_minibatch_size",
        "module_selector",
        "perfect_score",
        "use_merge",
        "max_merge_invocations",
        "max_metric_calls",
        "logger",
        "run_dir",
        "use_wandb",
        "wandb_api_key",
        "wandb_init_kwargs",
        "use_mlflow",
        "track_best_outputs",
        "display_progress_bar",
        "raise_on_exception",
        "seed",
    }
)

# kwargs dspy.GEPA's constructor rejects outright, and why.
_GEPA_UNUSABLE_OPTIMIZE_KWARGS = {
    "max_reflection_cost": (
        "not supported by dspy.GEPA (DSPy LMs expose no cumulative cost budget)"
    ),
    "reflection_prompt_template": (
        "unused by dspy.GEPA (DspyAdapter supplies its own proposer) —"
        " pass instruction_proposer instead"
    ),
}


def _gepa_init_params() -> dict[str, inspect.Parameter]:
    """Keyword parameters the installed ``dspy.GEPA.__init__`` accepts."""
    params = inspect.signature(dspy.GEPA.__init__).parameters
    return {
        name: param
        for name, param in params.items()
        if name not in {"self", "metric"}
        and param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def _parse_kv_tuples(pairs: tuple[str, ...]) -> dict[str, Any]:
    """Parse ``key=value`` CLI tuples, JSON-decoding each value when possible."""
    out: dict[str, Any] = {}
    for kv in pairs:
        key, _, value = kv.partition("=")
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
        out[key.strip()] = value
    return out


def _parse_gepa_kwargs(gepa_kwargs: tuple[str, ...]) -> dict[str, Any] | None:
    """Parse ``--gepa-kwargs key=value`` into ``dspy.GEPA(gepa_kwargs=...)``.

    These are forwarded verbatim to ``gepa.optimize``, so keys are validated
    against that callable's signature: unknown keys and keys dspy.GEPA already
    supplies itself fail here with a readable CLI error instead of a TypeError
    (or a silent duplicate) deep inside the optimizer.
    """
    gk = _parse_kv_tuples(gepa_kwargs)
    if not gk:
        return None

    from gepa import optimize as gepa_optimize

    accepted = set(inspect.signature(gepa_optimize).parameters)
    unknown = sorted(set(gk) - accepted)
    if unknown:
        raise click.ClickException(
            f"unknown GEPA kwargs: {', '.join(unknown)}"
            " (not accepted by gepa.optimize in the installed dspy/gepa)"
        )
    unusable = sorted(set(gk) & set(_GEPA_UNUSABLE_OPTIMIZE_KWARGS))
    if unusable:
        details = "; ".join(
            f"{name}: {_GEPA_UNUSABLE_OPTIMIZE_KWARGS[name]}" for name in unusable
        )
        raise click.ClickException(f"GEPA kwarg(s) unusable here — {details}")
    owned = sorted(set(gk) & _GEPA_DSPY_OWNED_OPTIMIZE_KWARGS)
    if owned:
        raise click.ClickException(
            f"GEPA kwargs supplied by dspy.GEPA itself: {', '.join(owned)}"
            " (use the matching CLI option, e.g. --seed or --use-merge,"
            " or pass a GEPA constructor kwarg via --gepa-kwarg)"
        )
    return gk


def _parse_gepa_ctor_kwargs(gepa_kwarg: tuple[str, ...]) -> dict[str, Any]:
    """Parse repeatable ``--gepa-kwarg key=value`` into constructor kwargs.

    Escape hatch for any ``dspy.GEPA.__init__`` parameter the named CLI options
    do not cover (e.g. ``--gepa-kwarg max_metric_calls=250``).
    """
    kwargs = _parse_kv_tuples(gepa_kwarg)
    accepted = _gepa_init_params()
    unknown = sorted(set(kwargs) - set(accepted))
    if unknown:
        raise click.ClickException(
            f"unknown dspy.GEPA constructor kwargs: {', '.join(unknown)}"
            f" (accepted: {', '.join(sorted(accepted))})"
        )
    return kwargs


def _gepa_ctor_kwargs(**overrides: Any) -> dict[str, Any]:
    """Keep only GEPA knobs whose value differs from ``dspy.GEPA``'s default.

    Guarantees the CLI defaults reproduce un-flagged behaviour exactly, while
    still forwarding any knob the user chose explicitly.
    """
    params = _gepa_init_params()
    selected: dict[str, Any] = {}
    for name, value in overrides.items():
        if value is None:
            continue
        default = params[name].default
        if default is inspect.Parameter.empty or value != default:
            selected[name] = value
    return selected


def _gepa_cli_kwargs(
    *,
    gepa_kwarg: tuple[str, ...],
    seed: int | None,
    use_merge: bool,
    component_selector: str,
    perfect_score: float,
    track_stats: bool,
    track_best_outputs: bool,
    gepa_skills: bool = False,
) -> dict[str, Any]:
    """Collect GEPA constructor kwargs chosen on the command line."""
    selected = _gepa_ctor_kwargs(
        seed=seed,
        use_merge=use_merge,
        component_selector=component_selector,
        perfect_score=perfect_score,
        track_stats=track_stats,
        track_best_outputs=track_best_outputs,
    )
    # Explicit --gepa-kwarg wins over the named options.
    selected.update(_parse_gepa_ctor_kwargs(gepa_kwarg))
    if gepa_skills:
        selected["instruction_proposer"] = _instruction_proposer()
    return selected


def _instruction_proposer() -> Any:
    """Build GEPA's ``InstructionProposer`` over the local skills library.

    ``~/.config/dspytools/skills/*/SKILL.md`` is exactly the directory layout
    ``InstructionProposer(skills=...)`` consumes, so the teacher's reflections
    get the mined Trace2Skill patterns as priors. ``truncate_history_outputs``
    keeps the reflection prompt inside the budget of a small reflection model.
    """
    from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

    from dspytools.config.settings import skills_dir

    skills_path = skills_dir()
    if not skills_path.exists():
        raise click.ClickException(
            f"no skills library at {skills_path} — create one with `dspytools skills create`"
        )
    return InstructionProposer(
        skills=skills_path,
        max_chars=2000,
        truncate_history_outputs=True,
    )


def _json_safe_params(params: dict[str, Any]) -> dict[str, Any]:
    """Filter optimizer params down to values that can be logged as JSON.

    ``run_params`` carries live objects (``reflection_lm`` is a ``dspy.LM``,
    ``gepa_kwargs`` may hold callables). Splatting them into
    ``register_run_with_lineage``/``log_compile`` wrote ``"<LM repr>"`` into
    ``compiled/index.json`` and the MLflow run params.
    """
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            safe[key] = value
            continue
        if isinstance(value, (list, tuple, dict)):
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            safe[key] = list(value) if isinstance(value, tuple) else value
    return safe


def _optimizer_run_params(spec_params: dict[str, Any]) -> dict[str, Any]:
    """Copy a spec's ``params`` dict for one command invocation.

    The spec dicts live for the life of the process (they are captured when the
    Click commands are registered), so writing into them leaked a
    ``reflection_lm`` from one invocation into the next.
    """
    return dict(spec_params)


def _valset_split(dataset: list, min_train: int = 2) -> tuple[list, list | None]:
    """Split a validation slice off ``dataset`` without starving the optimizer.

    Returns ``(train, valset_or_None)``. Small datasets keep everything for
    training — GEPA/MIPROv2 fall back to their own internal split.
    """
    if len(dataset) >= 6:
        val_size = max(3, len(dataset) // 5)
        train = dataset[:-val_size]
        if len(train) >= min_train:
            return train, dataset[-val_size:]
    return list(dataset), None


def _detailed_results_score(compiled: Any, default: float = 0.5) -> float:
    """Best validation score from ``compiled.detailed_results`` (GEPA stats).

    Returns ``default`` when ``track_stats`` was off or the optimizer does not
    attach detailed results — that placeholder used to be the only value
    recorded for every compile.
    """
    details = getattr(compiled, "detailed_results", None)
    if details is None:
        return default
    scores = getattr(details, "val_aggregate_scores", None)
    if not scores:
        return default
    best_idx = getattr(details, "best_idx", None)
    try:
        if isinstance(best_idx, int) and 0 <= best_idx < len(scores):
            return float(scores[best_idx])
        return float(max(scores))
    except (TypeError, ValueError):
        return default


def _instrument_metric(
    metric: Callable[..., float], val_field: str = "output"
) -> tuple[Callable[..., float], dict[str, Any]]:
    """Wrap a metric so a compile can report how many traces it accepted.

    Bootstrap-based optimizers silently emit an *uncompiled* program when the
    metric accepts zero demos; the returned stats let the caller say so out loud
    (metric name + one sample mismatch) instead of reporting success.
    """
    stats: dict[str, Any] = {
        "scored": 0,
        "matched": 0,
        "first_mismatch": None,
        "name": getattr(metric, "__name__", type(metric).__name__),
    }

    def observed(example: Any, prediction: Any, trace: Any = None) -> float:
        score = metric(example, prediction, trace)
        stats["scored"] += 1
        if score >= 1.0:
            stats["matched"] += 1
        elif stats["first_mismatch"] is None:
            stats["first_mismatch"] = (
                getattr(example, val_field, None),
                getattr(prediction, val_field, None),
            )
        return score

    return observed, stats


def _warn_if_no_bootstrap(stats: dict[str, Any] | None, optimizer: str) -> None:
    """Warn when a bootstrap compile accepted zero traces (no demos learned)."""
    if not stats or not stats["scored"] or stats["matched"]:
        return
    gold, pred = stats["first_mismatch"] or (None, None)
    click.echo(
        f"  Warning: {optimizer} bootstrapped 0/{stats['scored']} successful traces "
        f"({stats.get('name', 'metric')}) — the compiled program carries no demos. "
        f"Sample mismatch: expected {gold!r}, got {pred!r}.",
        err=True,
    )


def _require_finetune_capable_lm(lm: Any, module_name: str, trainset_path: str) -> None:
    """Fail fast when the student LM has no provider fine-tuning API.

    Local llama.cpp endpoints (``localhost``/``127.0.0.1``) cannot accept a
    fine-tuning upload, so BootstrapFinetune/GRPO would die inside the provider
    call with a confusing auth error.
    """
    api_base = lm.kwargs.get("api_base", "")
    if api_base and ("localhost" in api_base or "127.0.0.1" in api_base):
        raise click.ClickException(
            "this optimizer needs a provider with a fine-tuning API "
            "(OpenAI, Together, etc).\n"
            f"  Current api_base: {api_base} (local endpoint — no fine-tuning upload).\n"
            "  For local model weight optimization, use the LoRA distillation pipeline:\n"
            "    dspytools distill run --module {m} --trainset {t}\n"
            "    dspytools lora train --data output/<jsonl>".format(
                m=module_name, t=trainset_path
            )
        )


def _build_gepa(metric: Any, **kwargs: Any) -> Any:
    """Construct ``dspy.GEPA``, keeping every kwarg signature-valid.

    Named CLI knobs and ``--gepa-kwarg`` values are collected, stray
    compile-time arguments (``trainset``, ``k``, ...) are ignored, and the
    budget selector is resolved to exactly one of ``auto`` /
    ``max_metric_calls`` / ``max_full_evals`` as dspy requires.
    """
    accepted = _gepa_init_params()

    ctor: dict[str, Any] = {}
    for name in _GEPA_KNOB_NAMES:
        if name in kwargs:
            ctor[name] = kwargs.pop(name)
    for name in list(kwargs):
        if name in accepted and name != "metric":
            ctor[name] = kwargs.pop(name)

    if ctor.get("track_best_outputs"):
        # dspy asserts track_stats whenever track_best_outputs is requested.
        ctor.setdefault("track_stats", True)
    # dspy.GEPA asserts exactly one of auto / max_metric_calls / max_full_evals.
    # Two explicit budgets used to reach that bare assert (a crash, not a CLI
    # error), so resolve the conflict here.
    budgets = {
        name: ctor[name]
        for name in ("auto", "max_metric_calls", "max_full_evals")
        if ctor.get(name) is not None
    }
    if len(budgets) > 1:
        raise click.ClickException(
            "conflicting GEPA budgets: "
            + ", ".join(sorted(budgets))
            + " — auto, max_metric_calls and max_full_evals are mutually exclusive"
            " (pass at most one; omitting all three uses auto='light')"
        )
    if not budgets:
        ctor["auto"] = "light"

    return dspy.GEPA(metric=metric, **ctor)


_GEPA_KNOB_OPTIONS: dict[str, dict[str, Any]] = {
    "--seed": {
        "type": int,
        "default": None,
        "help": "GEPA RNG seed (omit to use dspy.GEPA's own default)",
    },
    "--use-merge/--no-use-merge": {
        "default": True,
        "help": "GEPA merge-based optimization (default: enabled)",
    },
    "--component-selector": {
        "default": "round_robin",
        "help": "GEPA component selector ('round_robin' or 'all')",
    },
    "--perfect-score": {
        "type": float,
        "default": 1.0,
        "help": "Metric score GEPA treats as perfect",
    },
    "--track-stats/--no-track-stats": {
        "default": False,
        "help": "Attach GEPA detailed_results to the compiled program",
    },
    "--track-best-outputs/--no-track-best-outputs": {
        "default": False,
        "help": "Track best valset outputs (implies --track-stats)",
    },
    "--gepa-kwarg": {
        "multiple": True,
        "help": "Any dspy.GEPA constructor kwarg as key=value (values JSON-parsed)",
    },
    "--gepa-skills/--no-gepa-skills": {
        "default": False,
        "help": "Feed the local skills library to GEPA's InstructionProposer",
    },
}


def _with_gepa_knobs(cmd: Callable) -> Callable:
    """Attach the shared GEPA knob options to an explicit Click command."""
    for opt_str, opt_kwargs in _GEPA_KNOB_OPTIONS.items():
        cmd = click.option(opt_str, **opt_kwargs)(cmd)
    return cmd


# Estimated token cost per optimizer (per trainset example). Follows the
# established cost-tracking convention: real LM usage isn't surfaced by the
# optimizers, so cost is estimated by optimizer intensity × dataset size.
_OPTIMIZER_TOKEN_MULTIPLIERS: dict[str, int] = {
    "knn": 2000,
    "mipro": 5000,
    "gepa": 8000,
    "copro": 4000,
    "simba": 3000,
    "bootstrap_few_shot": 3000,
    "sequential": 5000,
    "labeled_few_shot": 500,
}


def _estimate_optimizer_cost(
    optimizer_name: str, trainset_size: int
) -> tuple[int, float]:
    """Estimate (tokens, cost_usd) for a compile with the given optimizer.

    Uses per-optimizer token multipliers × dataset size and the DeepSeek
    pricing table (SSOT: core.cost_tracker PRICING). Optimizer names that map
    to compile subcommands are normalized (hyphens → underscores).
    """
    from dspytools.core.cost_tracker import TokenUsage

    norm = optimizer_name.replace("-", "_")
    tokens = trainset_size * _OPTIMIZER_TOKEN_MULTIPLIERS.get(norm, 3000)
    usage = TokenUsage.estimate(
        "deepseek/deepseek-v4-flash",
        prompt_tokens=tokens // 2,
        completion_tokens=tokens // 2,
    )
    return tokens, usage.cost_estimate


@click.group(name="compile", cls=LLMGroup)
def compile_cmd():
    """Compile and optimize DSPy programs."""


# ── Optimizer Registry ──────────────────────────────────────────────────


def _make_optimizer_cmd(name: str, cls_name: str, cls_lambda: Callable, **kwargs):
    """Generate a Click command for a simple optimizer."""
    spec_params = kwargs.get("params", {})
    module_type = kwargs.get("module_type", "predict")
    run_dir = kwargs.get("run_dir", name)
    needs_teacher = kwargs.get("needs_teacher", False)
    extra_options = kwargs.get("extra_options", {})

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
        use_mlflow: bool = False,
        gepa_kwargs: tuple[str, ...] = (),
        gepa_kwarg: tuple[str, ...] = (),
        seed: int | None = None,
        use_merge: bool = True,
        component_selector: str = "round_robin",
        perfect_score: float = 1.0,
        track_stats: bool = False,
        track_best_outputs: bool = False,
        gepa_skills: bool = False,
        bsize: int | None = None,
        max_steps: int | None = None,
    ):
        setup_dspy()
        student = load_module_by_name(module_name)
        # `total_trainset` is the real dataset; the optimizer only ever sees the
        # gate's train split.
        total_trainset = load_trainset(trainset_path)
        dataset_hash = compute_dataset_hash(total_trainset)

        # Compile dedup: skip if exact same config was already compiled
        if not force:
            existing = find_existing_compile(module_name, dataset_hash, name)
            if existing:
                click.echo(f"  Using cached: {existing['id']}")
                click.echo("  Use --force to force recompilation")
                return

        # ── Holdout split FIRST (Golden Rule 3) ─────────────────────────
        # The gate's holdout must never reach an optimizer input — including
        # GEPA's/MIPROv2's valset. Splitting after the compile (the old order)
        # meant the optimizer trained on the exact examples used to "validate".
        gate = get_holdout_gate()
        run_id, run_path = create_run_dir(run_dir, label)
        train_set, holdout = gate.split(total_trainset, compile_id=run_id)
        if not train_set:  # degenerate 1-example dataset: no split possible
            train_set = list(total_trainset)

        # Per-invocation copy of the registry's params — never mutate the spec.
        run_params = _optimizer_run_params(spec_params)
        metric: Callable[..., Any] = exact_match_metric()
        metric_stats: dict[str, Any] | None = None

        if draft:
            # Speculative compilation: student drafts, teacher polishes
            pipeline = GFLPipeline()
            result = pipeline.compile_draft(
                student=student, trainset=train_set, optimizer_name=name
            )
            compiled = result["best_program"]
            click.echo(
                f"  Draft score: {result['draft_score']:.4f} | Polished: {result['polished_score']:.4f}"
            )
        else:
            if name == "gepa":
                metric = gepa_metric
            else:
                metric, metric_stats = _instrument_metric(metric)

            if needs_teacher:
                teacher = LMRegistry.get_teacher()
                if teacher is None:
                    click.echo("  No teacher LM configured", err=True)
                    raise click.Abort()
                run_params["reflection_lm"] = teacher

            # GEPA: forward --use-mlflow, --gepa-kwargs and the GEPA knobs
            if name == "gepa":
                run_params["gepa_kwargs"] = _parse_gepa_kwargs(gepa_kwargs)
                run_params["use_mlflow"] = use_mlflow
                run_params.update(
                    _gepa_cli_kwargs(
                        gepa_kwarg=gepa_kwarg,
                        seed=seed,
                        use_merge=use_merge,
                        component_selector=component_selector,
                        perfect_score=perfect_score,
                        track_stats=track_stats,
                        track_best_outputs=track_best_outputs,
                        gepa_skills=gepa_skills,
                    )
                )

            # SIMBA needs `bsize` examples per step: derive a workable default
            # from the data instead of erroring out below 32 examples.
            if name == "simba":
                run_params["bsize"] = bsize or max(1, min(32, len(train_set)))
                run_params["max_steps"] = max_steps or 8
                if len(train_set) < run_params["bsize"]:
                    raise click.ClickException(
                        f"simba requires at least {run_params['bsize']} training "
                        f"examples (got {len(train_set)} after the holdout split)"
                    )

            opt = cls_lambda(metric=metric, trainset=train_set, **run_params)

            # KNNFewShot consumes trainset in __init__, not compile()
            compile_kwargs: dict[str, Any] = {"student": student}
            if "knn" not in name:
                compile_kwargs["trainset"] = train_set
            # GEPA (and BetterTogether's GEPA stage) need a separate valset; it
            # comes out of the train split, never out of the holdout.
            if name.startswith("gepa") or name == "better-together":
                gepa_train, gepa_valset = _valset_split(train_set)
                compile_kwargs["trainset"] = gepa_train
                if gepa_valset:
                    compile_kwargs["valset"] = gepa_valset
            # MIPROv2 hands 80% of the trainset to validation when no valset is
            # given, leaving 2-4 examples for demo construction on small sets.
            if name == "mipro":
                mipro_train, mipro_valset = _valset_split(train_set)
                compile_kwargs["trainset"] = mipro_train
                compile_kwargs["valset"] = (
                    mipro_valset or list(mipro_train[-1:]) or list(train_set)
                )
                # upstream MIN_MINIBATCH_SIZE is 50
                compile_kwargs["minibatch"] = len(mipro_train) > 50
                compile_kwargs["max_bootstrapped_demos"] = min(4, len(mipro_train))
                compile_kwargs["max_labeled_demos"] = min(16, len(mipro_train))
            # COPRO requires eval_kwargs
            if "copro" in name:
                compile_kwargs["eval_kwargs"] = {"num_threads": 4}
            # BootstrapFewShotWithOptuna requires the demo cap at compile time
            if "optuna" in name:
                compile_kwargs["max_demos"] = 4
            compiled, retry_stats = compile_with_retry(
                lambda: opt.compile(**compile_kwargs),
                max_retries=3,
                base_delay=2.0,
            )
            if retry_stats["retries"] > 0:
                click.echo(
                    f"  Retries: {retry_stats['retries']} (total delay: {retry_stats['total_delay']:.1f}s)"
                )
            _warn_if_no_bootstrap(metric_stats, name)

        # Cost tracking (approximate based on trainset size and optimizer).
        # Single canonical path: one estimate → one CompileCost.add_call →
        # TokenUsage.estimate. DSPy's Usage totals already fold cached, tool-use
        # and reasoning/audio tokens into prompt/completion counts, so no extra
        # counters are added here (that would double-count).
        est_tokens, _ = _estimate_optimizer_cost(name, len(total_trainset))

        cost = CompileCost(compile_id="", optimizer=name)
        cost.add_call(
            "deepseek/deepseek-v4-flash",
            prompt_tokens=est_tokens // 2,
            completion_tokens=est_tokens // 2,
        )
        cost.finish()

        # Real score when the optimizer attached stats (GEPA --track-stats);
        # 0.5 stays the documented fallback for optimizers that report nothing.
        compile_score = _detailed_results_score(compiled)

        # MLflow tracking — JSON-safe params only (no live LM objects)
        logged_params = _json_safe_params(run_params)

        tracker = get_tracker()
        tracker.log_compile(
            optimizer=name,
            module=module_name,
            score=compile_score,
            params=logged_params,
            metrics={
                "tokens": cost.total_tokens,
                "cost_usd": cost.total_cost,
                "trainset_size": len(total_trainset),
                "train_split_size": len(train_set),
                "holdout_size": len(holdout),
            },
        )

        # Holdout gate validation (Invariant 5: holdout never seen by optimizer)
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
            {
                "optimizer": name,
                "module": module_name,
                "score": validation.get("score", compile_score),
                "trainset_size": len(total_trainset),
                "holdout_size": len(holdout),
                **logged_params,
            },
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
            trainset_size=len(total_trainset),
            holdout_size=len(holdout),
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
            data_size=len(total_trainset),
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

    # Apply per-optimizer extra options (e.g. --use-mlflow, --gepa-kwargs for GEPA)
    for opt_str, opt_kwargs in extra_options.items():
        cmd = click.option(opt_str, **opt_kwargs)(cmd)

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
        "cls_lambda": lambda metric, **kw: _build_gepa(metric, **kw),
        "params": {},
        "needs_teacher": True,
        "extra_options": {
            "--use-mlflow/--no-use-mlflow": {
                "default": False,
                "help": "Enable MLflow tracking for GEPA optimization",
            },
            "--gepa-kwargs": {
                "multiple": True,
                "help": "Extra kwargs for gepa.optimize as key=value (values JSON-parsed)",
            },
            **_GEPA_KNOB_OPTIONS,
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
        "cls_lambda": lambda metric, **kw: dspy.SIMBA(
            metric=metric,
            bsize=kw.pop("bsize", 32),
            max_steps=kw.pop("max_steps", 8),
        ),
        "params": {},
        "extra_options": {
            "--bsize": {
                "type": int,
                "default": None,
                "help": "Examples sampled per SIMBA step (default: min(32, dataset size))",
            },
            "--max-steps": {
                "type": int,
                "default": None,
                "help": "SIMBA optimization steps (default: 8)",
            },
        },
    },
    "bootstrap-few-shot": {
        "cls_name": "BootstrapFewShot",
        # Demo caps stay at DSPy's defaults (max_labeled_demos=16,
        # max_bootstrapped_demos=4); pinning both to 4 dropped the labeled budget.
        "cls_lambda": lambda metric, **kw: dspy.BootstrapFewShot(metric=metric),
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
@click.option(
    "--gepa-kwargs",
    multiple=True,
    help="Extra kwargs for gepa.optimize as key=value (values JSON-parsed), e.g. --gepa-kwargs use_cloudpickle=true",
)
@click.option(
    "--use-mlflow/--no-use-mlflow",
    default=False,
    help="Enable MLflow tracking for GEPA optimization",
)
@_with_gepa_knobs
def compile_submit(
    optimizer: str,
    module_name: str,
    trainset_path: str,
    label: str | None,
    auto: str,
    k: int,
    gepa_kwargs: tuple[str, ...],
    use_mlflow: bool,
    gepa_kwarg: tuple[str, ...] = (),
    seed: int | None = None,
    use_merge: bool = True,
    component_selector: str = "round_robin",
    perfect_score: float = 1.0,
    track_stats: bool = False,
    track_best_outputs: bool = False,
    gepa_skills: bool = False,
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
        # Explicit valset: without one MIPROv2 hands 80% of the trainset to
        # validation, leaving almost nothing for demo construction.
        mipro_train, mipro_valset = _valset_split(trainset)
        compiled = opt.compile(
            student=student,
            trainset=mipro_train,
            valset=mipro_valset or list(mipro_train[-1:]),
            minibatch=len(mipro_train) > 50,
            max_bootstrapped_demos=min(4, len(mipro_train)),
            max_labeled_demos=min(16, len(mipro_train)),
        )
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
        gk = _parse_gepa_kwargs(gepa_kwargs)
        opt = _build_gepa(
            gepa_metric,
            auto=auto,
            reflection_lm=teacher,
            use_mlflow=use_mlflow,
            gepa_kwargs=gk,
            **_gepa_cli_kwargs(
                gepa_kwarg=gepa_kwarg,
                seed=seed,
                use_merge=use_merge,
                component_selector=component_selector,
                perfect_score=perfect_score,
                track_stats=track_stats,
                track_best_outputs=track_best_outputs,
                gepa_skills=gepa_skills,
            ),
        )
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

    def _run_better_together() -> str:
        # BetterTogether evaluates with its own metric via dspy.Evaluate, which
        # sums scores — the plain-float metric is required there. Its GEPA stage
        # keeps the feedback-returning metric for reflection.
        student.set_lm(LMRegistry.get_or_default())
        opt = dspy.BetterTogether(
            metric=gepa_metric_float,
            p=_build_gepa(gepa_metric, auto=auto, reflection_lm=teacher),
        )
        bt_train, bt_valset = _valset_split(trainset)
        compile_kwargs = {"student": student, "trainset": bt_train, "strategy": "p"}
        if bt_valset:
            compile_kwargs["valset"] = bt_valset
        compiled = opt.compile(**compile_kwargs)
        run_id, run_path = create_run_dir("better_together", label)
        save_program(
            run_path,
            compiled,
            {"inputs": ["input"], "outputs": ["output"]},
            module_type="predict",
        )
        register_run_with_graph(
            run_id,
            {"optimizer": "better_together", "module": module_name, "auto": auto},
            optimizer="better_together",
        )
        return run_id

    runners = {
        "knn": _run_knn,
        "mipro": _run_mipro,
        "gepa": _run_gepa,
        "better-together": _run_better_together,
    }

    job_id = CompileScheduler.submit(optimizer, module_name, runners[optimizer], label)
    _log.info("compile_submit", optimizer=optimizer, module=module_name, job_id=job_id)
    click.echo(f"  Job submitted: {job_id}")
    click.echo(f"  Check status: dspytools compile status {job_id}")


@compile_cmd.command(name="flex", cls=LLMCommand)
@click.argument("module_name")
@click.argument("trainset_path")
@label_option
@click.option("--auto", default="light", help="Auto mode for GEPA (light/medium/heavy)")
@click.option(
    "--tools",
    "-t",
    multiple=True,
    help="Tools as server:tool_name or module.attr (e.g. math.sqrt)",
)
@click.option(
    "--max-predictor-calls",
    default=100,
    type=int,
    help="Cap predictor calls per Flex forward (None = unlimited)",
)
@click.option(
    "--use-mlflow/--no-use-mlflow",
    default=False,
    help="Enable MLflow tracking for GEPA optimization",
)
@_with_gepa_knobs
def compile_flex(
    module_name: str,
    trainset_path: str,
    label: str | None,
    auto: str,
    tools: tuple[str, ...],
    max_predictor_calls: int,
    use_mlflow: bool,
    gepa_kwarg: tuple[str, ...] = (),
    seed: int | None = None,
    use_merge: bool = True,
    component_selector: str = "round_robin",
    perfect_score: float = 1.0,
    track_stats: bool = False,
    track_best_outputs: bool = False,
    gepa_skills: bool = False,
):
    """Optimize a module's implementation code with dspy.Flex + GEPA.

    Wraps the module in a dspy.Flex program (implementation source code is
    the optimizable unit) and runs GEPA with a cost-aware metric that
    penalizes excessive LM calls via the program_trace parameter. The
    optimized program is saved with module_type='flex' for hot-swap loading.

    \b
    Examples:
        dspytools compile flex my_module trainset.jsonl --auto light
        dspytools compile flex my_module trainset.jsonl -t math.sqrt
        dspytools compile flex my_module trainset.jsonl --use-mlflow
    """
    setup_dspy()
    teacher = LMRegistry.get_teacher()
    if teacher is None:
        click.echo("  No teacher LM configured", err=True)
        raise click.Abort()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)

    # Derive the Flex signature from the module's signature fields
    sig = getattr(student, "signature", None)
    sig_inputs = list(getattr(sig, "input_fields", {}).keys()) or ["input"]
    sig_outputs = list(getattr(sig, "output_fields", {}).keys()) or ["output"]
    sig_str = f"{', '.join(sig_inputs)} -> {', '.join(sig_outputs)}"

    from dspytools.commands.run import _load_tools

    dspy_tools = _load_tools(tools)

    def _run_flex() -> str:
        if dspy_tools:
            prog = dspy.Flex(
                sig_str,
                tools=dspy_tools,
                max_predictor_calls=max_predictor_calls,
            )
        else:
            prog = dspy.Flex(sig_str, max_predictor_calls=max_predictor_calls)

        opt = _build_gepa(
            gepa_metric_cost_aware(),
            auto=auto,
            reflection_lm=teacher,
            use_mlflow=use_mlflow,
            **_gepa_cli_kwargs(
                gepa_kwarg=gepa_kwarg,
                seed=seed,
                use_merge=use_merge,
                component_selector=component_selector,
                perfect_score=perfect_score,
                track_stats=track_stats,
                track_best_outputs=track_best_outputs,
                gepa_skills=gepa_skills,
            ),
        )
        # Split valset to avoid GEPA overfitting warning
        val_size = max(3, len(trainset) // 5) if len(trainset) >= 6 else 0
        compile_kwargs = {"student": prog}
        if val_size > 0:
            compile_kwargs["valset"] = trainset[-val_size:]
            compile_kwargs["trainset"] = trainset[:-val_size]
        else:
            compile_kwargs["trainset"] = trainset
        compiled = opt.compile(**compile_kwargs)
        run_id, run_path = create_run_dir("gepa-flex", label)
        save_program(
            run_path,
            compiled,
            {"inputs": sig_inputs, "outputs": sig_outputs},
            module_type="flex",
        )
        register_run_with_graph(
            run_id,
            {"optimizer": "gepa-flex", "module": module_name, "auto": auto},
            optimizer="gepa",
        )
        return run_id

    job_id = CompileScheduler.submit("gepa-flex", module_name, _run_flex, label)
    _log.info("compile_flex", optimizer="gepa-flex", module=module_name, job_id=job_id)
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
        # dspy.Evaluate (BetterTogether's scorer) sums metric returns, so the
        # float variant is required here; GEPA's own stage keeps the feedback one.
        metric=gepa_metric_float,
        p=_build_gepa(gepa_metric, auto="light", reflection_lm=teacher),
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
        score=_detailed_results_score(compiled),
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
@click.option(
    "--num-iters",
    default=5,
    type=int,
    help="Training epochs, forwarded to the provider as train_kwargs['n_epochs']",
)
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
    _require_finetune_capable_lm(student_lm, module_name, trainset_path)

    # dspy.BootstrapFinetune has no num_iters kwarg: training hyper-parameters go
    # to the provider through train_kwargs (num_iters → n_epochs).
    opt = dspy.BootstrapFinetune(
        metric=exact_match_metric(),
        train_kwargs={"n_epochs": num_iters},
    )
    compiled = opt.compile(student=student, trainset=trainset)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="bootstrap_finetune",
        module=module_name,
        score=_detailed_results_score(compiled),
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
        # auto_meta=False: the user explicitly requested an optimizer — the
        # MetaOptimizer must not silently swap it (it overrode --single gepa
        # to bootstrap_few_shot before this fix).
        result = pipeline.run_single(
            optimizer_name=single.replace("-", "_"),
            student=student,
            trainset=trainset,
            auto_meta=False,
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

        # Cost/quality view: estimated $ and $/score per optimizer. Surfaces
        # the cost-aware tradeoff so users can pick a cheaper optimizer when
        # scores are near-equal (Pareto-style lens on the comparison).
        click.echo("\n  Cost / quality (estimated):")
        cost_rows = []
        for name, score in sorted(
            result["all_scores"].items(), key=lambda kv: kv[1], reverse=True
        ):
            _, est_cost = _estimate_optimizer_cost(name, len(trainset))
            dollars_per_score = est_cost / score if score > 0 else float("inf")
            cost_rows.append(
                [
                    name,
                    f"{score:.4f}",
                    f"${est_cost:.4f}",
                    f"${dollars_per_score:.4f}"
                    if dollars_per_score != float("inf")
                    else "—",
                ]
            )
        table(
            "Optimizer cost/quality",
            ["Optimizer", "Score", "Est. $", "$/score"],
            cost_rows,
        )
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
@click.option(
    "--max-steps", default=10, type=int, help="GRPO train steps (num_train_steps)"
)
@click.option("--num-rollouts", default=1, type=int, help="Rollouts per GRPO step")
@click.option(
    "--num-threads",
    default=1,
    type=int,
    help="Rollout parallelism (1 for a local model)",
)
@label_option
def compile_grpo(
    module_name: str,
    trainset_path: str,
    max_steps: int,
    num_rollouts: int,
    num_threads: int,
    label: str | None,
):
    """Compile with GRPO (RL-based optimization via LM fine-tuning).

    GRPO treats the program as a single policy and updates its predictor LMs
    through ``LM.finetune()``. That requires a provider with a fine-tuning API
    (OpenAI/Together/etc); local llama.cpp endpoints fail fast with a pointer to
    the LoRA distillation pipeline instead.
    """

    setup_dspy()

    student = load_module_by_name(module_name)
    trainset = load_trainset(trainset_path)
    student_lm = LMRegistry.get_or_default()
    student.set_lm(student_lm)
    _require_finetune_capable_lm(student_lm, module_name, trainset_path)

    # lazy: GRPO is only exported from dspy.teleprompt.grpo in 3.5
    from dspytools.gfl.grpo import compile_grpo as _grpo_compile

    compiled = _grpo_compile(
        student,
        trainset,
        max_steps=max_steps,
        num_rollouts=num_rollouts,
        num_threads=num_threads,
    )

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="grpo",
        module=module_name,
        score=0.5,
        params={
            "max_steps": max_steps,
            "num_rollouts": num_rollouts,
            "num_threads": num_threads,
        },
    )

    run_id, run_path = create_run_dir("grpo", label)
    save_program(
        run_path,
        compiled,
        {"inputs": ["input"], "outputs": ["output"]},
        module_type="predict",
    )
    register_run_with_graph(
        run_id,
        {
            "optimizer": "grpo",
            "module": module_name,
            "max_steps": max_steps,
            "num_rollouts": num_rollouts,
        },
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

    # dspy.AvatarOptimizer builds its comparator with dspy.TypedPredictor, which
    # current DSPy releases no longer ship. Fail here with an actionable message
    # instead of an AttributeError half-way through the compile.
    if not hasattr(dspy, "TypedPredictor"):
        raise click.ClickException(
            "AvatarOptimizer is unavailable in the installed DSPy version "
            "(it depends on dspy.TypedPredictor, which this release no longer ships). "
            "Use 'compile gepa' or 'compile mipro' for instruction optimization instead."
        )

    # AvatarOptimizer requires an acting agent module (dspy.ReAct with tools)
    # that produces 'actions' in its output. Simple Predict/ChainOfThought
    # modules don't have actions and will crash.
    test_result = student(question="test")
    if not hasattr(test_result, "actions"):
        fail(
            "AvatarOptimizer requires an acting agent module (e.g., dspy.ReAct with tools)"
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
        score=_detailed_results_score(compiled),
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
    # Step 2 fine-tunes the student LM — same provider requirement as
    # `compile finetune`, so fail fast on a local llama.cpp endpoint.
    _require_finetune_capable_lm(student_lm, module_name, trainset_path)

    click.echo(f"  Teacher: {teacher}")
    click.echo(f"  Student: {student_lm}")

    # Step 1: Teacher optimization (GEPA with reflection_lm)
    click.echo("  [1/2] Teacher optimizing instructions...")

    gepa_opt = _build_gepa(gepa_metric, auto=auto, reflection_lm=teacher)
    # Split valset to avoid GEPA overfitting warning
    val_size = max(3, len(trainset) // 5) if len(trainset) >= 6 else 0
    gepa_kwargs: dict[str, Any] = {"student": student}
    if val_size > 0:
        gepa_kwargs["valset"] = trainset[-val_size:]
        gepa_kwargs["trainset"] = trainset[:-val_size]
    else:
        gepa_kwargs["trainset"] = trainset
    optimized = gepa_opt.compile(**gepa_kwargs)

    # Step 2: Student finetuning on optimized prompts. GEPA's valset is held out
    # from the fine-tuning data — it exists to score candidates, not to train.
    finetune_data = gepa_kwargs["trainset"]
    click.echo("  [2/2] Student absorbing via finetune...")

    ft_opt = dspy.BootstrapFinetune(
        metric=exact_match_metric(), train_kwargs={"n_epochs": num_iters}
    )
    distilled = ft_opt.compile(student=optimized, trainset=finetune_data)

    # MLflow tracking

    tracker = get_tracker()
    tracker.log_compile(
        optimizer="distill",
        module=module_name,
        score=_detailed_results_score(optimized),
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
