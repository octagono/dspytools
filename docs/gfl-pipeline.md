# GFL Pipeline

The **Generative Feedback Loop** (GFL) is a 4-way optimizer comparison engine with early pruning and speculative compilation.

## Core Entry Points

### `GFLPipeline.run()` — 4-Way Comparison
```python
from dspytools.gfl.pipeline import GFLPipeline

pipeline = GFLPipeline()
result = pipeline.run(student=my_module, trainset=data)
```
Evaluates all optimizers on 10% of data, then runs survivors on full dataset. Returns best optimizer + scores.

### `run_halving()` — Successive Halving
Prunes worst 50% each round. Saves up to 75% of compile time vs full 4-way:
```python
result = pipeline.run_halving(student=my_module, trainset=data)
# → {"best_optimizer": "gepa", "best_score": 0.87, "survivors": [...], "pruned": [...]}
```
Configurable optimizer list via `optimizers` parameter. `--auto-suggest` CLI flag uses `MetaOptimizer` to select the best optimizer list by dataset size/complexity.

### `compile_draft()` — Speculative Compile
Quick draft compile using BootstrapFewShot, suitable for rapid iteration:
```python
draft = pipeline.compile_draft(student=my_module, trainset=data[:50])
# → round_scores array with per-round scores for granular tracking
```
Configurable `draft_rounds` (default 3) and `polish_rounds` (default 1). Phase 1 uses student LM; phase 2 uses teacher LM for polish, reducing API costs by 3-5x.

### `run_single()` — Single Optimizer
```python
result = pipeline.run_single(student=my_module, trainset=data, optimizer="mipro")
```

## Data Management

- **`split_holdout()`**: Splits off 20% holdout set before optimizer ever sees data. In `GFLPipeline.run()`, holdout is split at the START (Invariant 5) and used only for SPRT validation, never by compile.
- **`gate_promotion()`**: Checks if new program beats previous baseline at 95% confidence before promoting.
- **SPRT post-compile validation**: After best optimizer is selected, `run()` calls `SelfEvolveEngine.validate_and_deploy()` (SPRT, p0=0.50, p1=0.65, α=0.05, β=0.20) on the reserved holdout. Returns `{validation: {accepted, score, p_value, n_evaluated, early_stop}}`. Falls back to baseline if rejected.

## Synthetic Data & Co-Evolution

### `ChallengerSolver`
R-Zero co-evolution: a Challenger program generates harder tasks, a Solver program attempts them, and both improve iteratively:
```python
from dspytools.gfl.synthetic import ChallengerSolver

cs = ChallengerSolver(challenger_program, solver_program)
result = cs.co_evolve(num_rounds=5)
```

### `DataSynthesizer`
Generates synthetic training data from seed examples:
```python
from dspytools.gfl.synthetic import DataSynthesizer

synth = DataSynthesizer()
data = synth.generate("seed.json", target_count=50)
```

## Paper Optimizers

| Module | Source |
|--------|--------|
| `SPINOptimizer` | SPIN: Self-Play Fine-Tuning (arXiv 2401.01335) |
| `MetaPromptOptimizer` | MetaSPO: Bilevel meta-prompt optimization (arXiv 2505.09666) |
| `GEPAParetoFrontier` | GEPA: Guided Evolutionary Program Adaptation (arXiv 2507.19457) |
| `LSETreeExplorer` | LSE: Learned Skill Evolution (arXiv 2603.18620) |
| `GRAOMetaOptimizer` | GRAO: Graph-based Recursive Architecture Optimization |
| `SkillConsolidator` | Trace2Skill: Rollout → Analyze → Consolidate (arXiv 2603.25158) |
| `PurifiedOPSDOptimizer` | Purified OPSD: On-Policy Self-Distillation Without Losing How to Think (arXiv 2607.02234) |

## Resources

- Source: `src/dspytools/gfl/pipeline.py`
- Budget: `src/dspytools/gfl/budget.py`
- Tracker: `src/dspytools/gfl/tracker.py`
- Paper optimizers: `src/dspytools/gfl/paper_optimizers.py`
- Synthetic data: `src/dspytools/gfl/synthetic.py`
- Trace2Skill: `src/dspytools/gfl/consolidation.py`
