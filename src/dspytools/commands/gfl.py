"""dspytools gfl — Generative Feedback Loop commands.

Synthesize data, meta-learn optimizers, decompose tasks, monitor quality.
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _P

from dspytools.cli.output import console, header, info, ok, panel, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core._io import read_json as _read_json
from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)


@click.group(name="gfl", cls=LLMGroup)
def gfl_cmd():
    """Generative Feedback Loop: synthesize, meta-learn, decompose, monitor."""


@gfl_cmd.command(name="synthesize", cls=LLMCommand)
@click.argument("seed_path")
@click.option(
    "--target", default=10, type=int, help="Target number of synthetic examples"
)
@click.option("--output", "-o", help="Output JSON path")
def gfl_synthesize(seed_path: str, target: int, output: str | None):
    """Generate synthetic training data from seed examples."""
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.synthetic import DataSynthesizer

    setup_dspy()
    synth = DataSynthesizer()
    result = synth.generate(seed_path, target_count=target, output_path=output)
    panel(
        "Synthetic Data Generated",
        f"[bold]Generated:[/] {result['generated']} examples\n"
        f"[bold]Saved to:[/] {result['output_path']}",
        border_style="green",
    )
    _log.info(
        "gfl_synthesize",
        generated=result["generated"],
        output_path=result.get("output_path"),
    )


@gfl_cmd.command(name="meta-optimize", cls=LLMCommand)
@click.argument("program")
@click.option("--dataset-size", default=10, type=int)
@click.option(
    "--complexity", default="medium", type=click.Choice(["simple", "medium", "complex"])
)
def gfl_meta_optimize(program: str, dataset_size: int, complexity: str):
    """Select the best optimizer for a task using meta-learning."""
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.meta_learn import MetaOptimizer

    setup_dspy()
    meta = MetaOptimizer()
    result = meta.select_optimizer(program, dataset_size, complexity)
    panel(
        "Meta-Optimizer Selection",
        f"[bold]Program:[/] {result['program']}\n"
        f"[bold]Optimizer:[/] {result['optimizer']}\n"
        f"[bold]Reason:[/] {result['reason']}",
        border_style="cyan",
    )


@gfl_cmd.command(name="decompose", cls=LLMCommand)
@click.argument("task")
@click.option("--output", "-o", help="Save generated code")
def gfl_decompose(task: str, output: str | None):
    """Decompose a complex task into DSPy sub-modules (Self-Discover)."""
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.decompose import TaskDecomposer

    setup_dspy()
    dec = TaskDecomposer()
    result = dec.decompose(task)

    _log.info("gfl_decompose", task=task, sub_tasks=len(result.get("sub_tasks", [])))
    console.print(f"\n[bold cyan]Task Decomposition:[/] {task}")
    console.print(f"\n[bold]Sub-tasks ({len(result['sub_tasks'])}):[/]")
    for sub in result["sub_tasks"]:
        depends = f" ← {', '.join(sub['depends_on'])}" if sub["depends_on"] else ""
        console.print(f"  • {sub['name']}: {sub['signature']}{depends}")

    if result.get("parallel"):
        console.print(
            f"\n[bold]Parallel:[/] {', '.join(str(p) for p in result['parallel'])}"
        )
    if result.get("sequential"):
        console.print(f"[bold]Sequential:[/] {', '.join(result['sequential'])}")

    if output:
        code = dec.generate_code(result, output.replace(".py", ""))

        _P(output).write_text(code)
        ok(f"Generated code saved to {output}")


@gfl_cmd.command(name="ab-test", cls=LLMCommand)
@click.argument("program_a")
@click.argument("program_b")
@click.option("--trials", "-n", default=20, type=int, help="Number of trials")
@click.option(
    "--confidence", "-c", default=0.9, type=float, help="Confidence threshold"
)
@click.option(
    "--auto-deploy/--no-auto-deploy", default=False, help="Auto-deploy winner"
)
def gfl_ab_test(
    program_a: str, program_b: str, trials: int, confidence: float, auto_deploy: bool
):
    """A/B test two compiled programs with statistical significance.

    PROGRAM_A and PROGRAM_B are compiled program run IDs.
    After N trials, deploys the winner if auto-deploy is set and win-rate ≥ confidence.
    """
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.ab_test import ABTest, auto_deploy_if_better

    setup_dspy()
    mgr = HotSwapManager()
    mgr.load_all()

    if program_a not in [p["id"] for p in mgr.list()]:
        console.print(f"[red]Program '{program_a}' not loaded[/red]")
        return
    if program_b not in [p["id"] for p in mgr.list()]:
        console.print(f"[red]Program '{program_b}' not loaded[/red]")
        return

    info(f"Testing: {program_a} vs {program_b} ({trials} trials)")

    test = ABTest(None, None, confidence=confidence)
    # Get the actual programs
    pa = (
        mgr._programs[program_a] if program_a in getattr(mgr, "_programs", {}) else None
    )
    pb = (
        mgr._programs[program_b] if program_b in getattr(mgr, "_programs", {}) else None
    )

    if pa is None or pb is None:
        console.print(
            "[red]Could not load programs. Use 'dspytools server list' to see loaded programs.[/red]"
        )
        return

    test.prog_a = pa
    test.prog_b = pb
    result = test.run([], n_trials=trials)

    panel(
        "A/B Test Results",
        f"[bold]Winner:[/] {result['winner'] or 'none'}\n"
        f"[bold]A wins:[/] {result['wins_a']} ({result['rate_a']:.0%})\n"
        f"[bold]B wins:[/] {result['wins_b']} ({result['rate_b']:.0%})\n"
        f"[bold]Draws:[/] {result['draws']}\n"
        f"[bold]Significant:[/] {result['significant']}\n"
        f"[bold]Recommendation:[/] {result['recommendation']}",
        border_style="green" if result.get("winner") else "yellow",
    )

    if auto_deploy and result.get("significant"):
        deployed = auto_deploy_if_better(result, mgr, program_a, program_b)
        if deployed:
            ok(f"Auto-deployed: {deployed}")


@gfl_cmd.command(name="consolidate", cls=LLMCommand)
@click.argument("program_id")
@click.option(
    "--tasks", "-t", help="JSON array of tasks [{input: dict, expected: str}]"
)
@click.option("--skill-name", default="trace2skill", help="Name for the evolved skill")
@click.option(
    "--mode",
    default="creation",
    type=click.Choice(["creation", "deepening"]),
    help="Creation: generate from scratch. Deepening: refine existing skill.",
)
@click.option(
    "--transfer/--no-transfer",
    default=False,
    help="Run transfer validation across models",
)
def gfl_consolidate(
    program_id: str, tasks: str | None, skill_name: str, mode: str, transfer: bool
):
    """Trace2Skill: evolve agent skills from execution trajectories.

    arXiv 2603.25158: 3-stage pipeline — parallel rollout, LLM-driven analysis,
    hierarchical consolidation with inductive reasoning.
    """
    from dspytools.core.hotswap import HotSwapManager
    from dspytools.core.metrics import exact_match_metric
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.consolidation import SkillConsolidator

    setup_dspy()

    mgr = HotSwapManager()
    mgr.load_all()

    mgr.swap(program_id)

    compiled_program = mgr.active_program

    # Build tasks
    if tasks:
        if _P(tasks).exists():
            task_list = _read_json(tasks)
        else:
            task_list = _json.loads(tasks)
    else:
        # Auto-detect input field from signature.json
        from dspytools.config.settings import compiled_dir

        sig_path = compiled_dir() / program_id / "signature.json"
        input_field = "input"
        if sig_path.exists():
            sig_data = _read_json(sig_path)
            inputs = sig_data.get("inputs", ["input"])
            input_field = inputs[0] if inputs else "input"

        task_list = [
            {"input": {input_field: "hello"}, "expected": "Hello, World!"},
            {"input": {input_field: "test"}, "expected": "Test output"},
        ]

    metric = exact_match_metric()
    consolidator = SkillConsolidator()
    info(
        f"Trace2Skill evolving '{skill_name}' ({mode} mode, {len(task_list)} tasks)..."
    )

    if mode == "deepening":
        existing = (consolidator.SKILLS_DIR / skill_name / "SKILL.md").read_text()
        result = consolidator.evolve(
            program=compiled_program,
            tasks=task_list,
            metric=metric,
            skill_name=skill_name,
            skill_content=existing,
            mode=mode,
        )
    else:
        result = consolidator.evolve(
            program=compiled_program,
            tasks=task_list,
            metric=metric,
            skill_name=skill_name,
            mode=mode,
        )

    panel(
        "Trace2Skill Consolidation Complete",
        f"[bold]Skill:[/] {result.skill_name}\n"
        f"[bold]Mode:[/] {result.mode}\n"
        f"[bold]Trajectories:[/] {result.trajectories_analyzed} "
        f"({result.success_trajectories} success, {result.error_trajectories} error)\n"
        f"[bold]Patches:[/] {result.patches_generated} generated, "
        f"{result.patches_accepted} accepted, {result.patches_discarded} discarded\n"
        f"[bold]Quality dropped:[/] {result.quality_dropped} trajectories\n"
        f"[bold]Guardrail failures:[/] {result.guardrail_failures}\n"
        f"[bold]Elapsed:[/] {result.elapsed_seconds:.1f}s\n"
        f"[bold]Skill preview:[/]\n{result.evolved_skill[:300]}...",
        border_style="green",
    )
    _log.info(
        "gfl_consolidate",
        program_id=program_id,
        skill_name=skill_name,
        mode=mode,
        trajectories_analyzed=result.trajectories_analyzed,
        patches_accepted=result.patches_accepted,
        elapsed_seconds=result.elapsed_seconds,
    )

    if transfer:
        scores = SkillConsolidator.validate_transfer(
            skill_name,
            compiled_program,
            task_list[:5],
            metric,
        )
        panel(
            "Transfer Validation",
            "\n".join(f"  {m}: {s:.3f}" for m, s in scores.items()),
            border_style="cyan",
        )


@gfl_cmd.command(name="spin", cls=LLMCommand)
@click.argument("module_name")
@click.option(
    "--iterations", "-n", default=3, type=int, help="Number of SPIN iterations"
)
@click.option("--trainset", "-t", help="JSON array of training examples")
def gfl_spin(module_name: str, iterations: int, trainset: str | None):
    """SPIN self-play optimization — arXiv 2401.01335.

    Self-Play fIne-tuNing: generates candidate outputs, then discriminates
    model outputs from gold using teacher LM.
    """
    from dspytools.core._dspy import dspy
    from dspytools.core.loaders import load_module_by_name
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.paper_optimizers import SPINOptimizer

    setup_dspy()
    student = load_module_by_name(module_name)

    if trainset:
        if _P(trainset).exists():
            raw = _read_json(trainset)
        else:
            raw = _json.loads(trainset)
        examples = [
            dspy.Example(**item).with_inputs(list(item.keys())[0]) for item in raw
        ]
    else:
        examples = [dspy.Example(input="hello", output="world").with_inputs("input")]

    opt = SPINOptimizer(student=student)
    info(f"SPIN self-play on '{module_name}' ({iterations} iterations)...")
    result = opt.iterate(examples, num_iterations=iterations)

    panel(
        "SPIN Self-Play Results",
        f"[bold]Module:[/] {module_name}\n"
        f"[bold]Iterations:[/] {iterations}\n"
        f"[bold]Final score:[/] {result['final_score']:.3f}\n"
        f"[bold]Improvement:[/] {result['improvement']:.3f}\n"
        f"[bold]History:[/] {len(result['iterations'])} steps",
        border_style="cyan",
    )


@gfl_cmd.command(name="opsd", cls=LLMCommand)
@click.argument("module_name")
@click.option(
    "--iterations", "-n", default=3, type=int, help="Number of purification iterations"
)
@click.option("--trainset", "-t", help="JSON array of training examples")
@click.option("--beta", default=1.0, type=float, help="PMI correction strength")
@click.option("--clip-c", default=10.0, type=float, help="Tanh soft clipping threshold")
@click.option("--wrap", "-w", default=None, help="Base optimizer to wrap (e.g. 'spin')")
def gfl_opsd(
    module_name: str,
    iterations: int,
    trainset: str | None,
    beta: float,
    clip_c: float,
    wrap: str | None,
):
    """Purified OPSD — arXiv 2607.02234.

    On-Policy Self-Distillation Without Losing How to Think.
    Replaces raw teacher distribution with PMI target that strips
    reference-induced shortcuts, preserving reflective reasoning.

    Can wrap any existing optimizer (--wrap spin, --wrap gepa) or
    run standalone to purify student inference.
    """
    from dspytools.core._dspy import dspy
    from dspytools.core.loaders import load_module_by_name
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.paper_optimizers import PurifiedOPSDOptimizer

    setup_dspy()
    student = load_module_by_name(module_name)

    if trainset:
        if _P(trainset).exists():
            raw = _read_json(trainset)
        else:
            raw = _json.loads(trainset)
        examples = [
            dspy.Example(**item).with_inputs(list(item.keys())[0]) for item in raw
        ]
    else:
        examples = [dspy.Example(input="hello", output="world").with_inputs("input")]

    # Wrap base optimizer if specified
    base_opt = None
    if wrap:
        if wrap.lower() == "spin":
            from dspytools.gfl.paper_optimizers import SPINOptimizer

            base_opt = SPINOptimizer(student=student)
        else:
            warn(f"Unknown base optimizer '{wrap}', running standalone purification")

    opt = PurifiedOPSDOptimizer(
        student=student,
        base_optimizer=base_opt,
        beta=beta,
        clip_c=clip_c,
    )
    mode = f"wrapped {wrap}" if wrap else "standalone"
    info(
        f"Purified OPSD ({mode}) on '{module_name}' ({iterations} iterations, β={beta})..."
    )
    result = opt.iterate(examples, num_iterations=iterations)

    stats = result["purification_stats"]
    panel(
        "Purified OPSD Results (arXiv 2607.02234)",
        f"[bold]Module:[/] {module_name}\n"
        f"[bold]Mode:[/] {mode}\n"
        f"[bold]Iterations:[/] {iterations}\n"
        f"[bold]Final score:[/] {result['final_score']:.3f}\n"
        f"[bold]Improvement:[/] {result['improvement']:.3f}\n"
        f"[bold]Avg PMI weight:[/] {stats['avg_pmi_weight']:.4f}\n"
        f"[bold]Positive PMI:[/] {stats['positive_pmi_count']}/{stats['total_pmi_signals']} "
        f"(question-conditioned)\n"
        f"[bold]Negative PMI:[/] {stats['negative_pmi_count']}/{stats['total_pmi_signals']} "
        f"(reference-dominated)\n"
        f"[bold]β:[/] {beta}, [bold]clip_c:[/] {clip_c}",
        border_style="cyan",
    )


@gfl_cmd.command(name="lse", cls=LLMCommand)
@click.argument("module_name")
@click.option("--depth", "-d", default=3, type=int, help="Max tree depth")
@click.option("--trainset", "-t", help="JSON array of training examples")
def gfl_lse(module_name: str, depth: int, trainset: str | None):
    """LSE tree-guided evolution — arXiv 2603.18620.

    Tree-guided evolution with UCB selection. Rewards improvement, not score.
    Each node represents an optimization attempt.
    """
    from dspytools.core._dspy import dspy
    from dspytools.core.loaders import load_module_by_name
    from dspytools.core.setup import setup_dspy
    from dspytools.gfl.paper_optimizers import LSETreeExplorer
    from dspytools.gfl.pipeline import GFLPipeline

    setup_dspy()
    student = load_module_by_name(module_name)

    # Infer input/output field names from the module's DSPy signature
    sub = student.predictor if hasattr(student, "predictor") else student
    sig = getattr(sub, "signature", None) or getattr(
        getattr(sub, "predict", None), "signature", None
    )
    if sig:
        in_fields = list(sig.input_fields.keys())
        out_fields = list(sig.output_fields.keys())
    else:
        in_fields, out_fields = ["input"], ["output"]

    if trainset:
        if _P(trainset).exists():
            raw = _read_json(trainset)
        else:
            raw = _json.loads(trainset)
        examples = [dspy.Example(**item).with_inputs(*in_fields) for item in raw]
    else:
        dummy = {f: "test" for f in in_fields}
        dummy.update({f: "output" for f in out_fields})
        # Use 3 copies so MIPROv2 (needs ≥2) and valset splitting work
        examples = [dspy.Example(**dummy).with_inputs(*in_fields) for _ in range(3)]

    lse = LSETreeExplorer(max_depth=depth)
    root = lse.new_root()
    info(f"LSE tree on '{module_name}' (depth={depth})...")

    t_field = in_fields[0] if in_fields else "input"
    v_field = out_fields[0] if out_fields else "output"

    for d in range(depth):
        for opt_name in ["bootstrap_few_shot", "mipro", "gepa"]:
            result = GFLPipeline().run_single(
                opt_name,
                student,
                examples,
                auto_synthesize=False,
                auto_meta=False,
                train_field=t_field,
                val_field=v_field,
            )
            score = result.get("best_score", 0.5)
            node = lse.expand(root, opt_name, score, f"depth={d}")
            _ = lse.select(node)
            ok(f"  Depth {d}: {opt_name} = {score:.3f}")

    panel(
        "LSE Tree Results",
        f"[bold]Module:[/] {module_name}\n"
        f"[bold]Tree size:[/] {len(lse.tree)} nodes\n"
        f"[bold]Max depth:[/] {depth}\n"
        f"[bold]Baseline:[/] root score = {lse.tree.get(root, {}).get('score', 'N/A')}",
        border_style="green",
    )


@gfl_cmd.command(name="gepa", cls=LLMCommand)
@click.argument("module_name")
@click.option("--scores", "-s", help="JSON array of {optimizer, score, feedback}")
def gfl_gepa(module_name: str, scores: str | None):
    """GEPA Pareto frontier — arXiv 2507.19457.

    Pareto frontier optimization with coverage-weighted selection.
    """
    from dspytools.gfl.paper_optimizers import GEPAParetoFrontier

    frontier = GEPAParetoFrontier()

    if scores:
        candidates = _json.loads(scores)
        for c in candidates:
            frontier.add(
                optimizer=c.get("optimizer", "unknown"),
                score=c.get("score", 0.0),
                feedback=c.get("feedback", ""),
            )
    else:
        frontier.add("bootstrap", 0.60, "Baseline")
        frontier.add("mipro", 0.75, "Bayesian improvement")
        frontier.add("gepa", 0.88, "Evolutionary with teacher LM")
        frontier.add("coprolite", 0.70, "Lower score, dominated")

    next_opt = frontier.select_next()
    panel(
        "GEPA Pareto Frontier",
        f"[bold]Frontier size:[/] {len(frontier.frontier)} candidates\n"
        f"[bold]Total candidates:[/] {len(frontier.candidates)}\n"
        f"[bold]Next optimizer:[/] {next_opt or 'none'}\n"
        f"\n[bold]Frontier:[/]"
        + "".join(
            f"\n  • {c['optimizer']} (score={c['score']:.2f}, coverage={c.get('coverage', 0)})"
            for c in frontier.frontier
        ),
        border_style="green",
    )


@gfl_cmd.command(name="status", cls=LLMCommand)
def gfl_status():
    """Show GFL pipeline status and configuration."""
    from dspytools.core.registry import list_compiled_runs
    from dspytools.core.setup import LMRegistry

    header("GFL Pipeline Status")

    # LM configuration
    student = LMRegistry.get_or_default()
    teacher = LMRegistry.get_teacher()
    if teacher:
        ok(f"Teacher LM: {teacher.model}")
    else:
        warn("Teacher LM: not configured")
    if student:
        ok(f"Student LM: {student.model}")
    else:
        warn("Student LM: not configured")

    # Compiled runs
    runs = list_compiled_runs()
    if runs:
        ok(f"Compiled programs: {len(runs)}")
        best = max(
            (r for r in runs if r.get("score") is not None),
            key=lambda r: float(r["score"]),
            default=None,
        )
        if best:
            info(f"Best score: {best['score']} ({best.get('optimizer', '?')})")
    else:
        info("No compiled programs yet")
