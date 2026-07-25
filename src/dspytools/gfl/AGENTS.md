# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Purpose

The GFL (Generative Feedback Loop) directory implements the full self-meta-learning loop: **Generate → Evaluate → Keep → Learn → Deploy**. It is the optimization engine of the DSPyTools CLI — every `compile`, `gfl loop`, `skills optimize`, and `self evolve` command routes through this package.

## Ownership

- `GFLPipeline` — primary interface for optimizer comparison and single-optimizer mode (compile gfl command, skills manager). Used by `commands/gfl.py` and `skills/manager.py`. Supports `mode="compare"` (4-way, default) or `mode="single"`.
- `LSETracker` — improvement delta tracker, used by pipeline and paper_optimizers. Shared state lives at `~/.config/dspytools/lse_log.json`.
- `MetaOptimizer` — meta-learner that selects best optimizer by dataset size/complexity. Shared state at `~/.config/dspytools/meta_optimizer.json`.
- `DataSynthesizer` — auto-generates training examples from seed data. Uses teacher LM for paraphrasing, domain variation, and complexity variation.
- `ABTest` — statistical A/B testing for compiled programs (win-rate with confidence threshold). Used by hot-swap deploy pipeline.
- `TaskDecomposer` — Self-Discover style decomposition into DSPy sub-modules. Used by `self evolve`.
- `SkillConsolidator` — Trace2Skill pattern mining from execution trajectories (arXiv 2603.25158). 3-stage pipeline: Rollout → Analyze → Consolidate. Outputs skills to `~/.config/dspytools/skills/`. Used by `GFLPipeline._consolidate_trajectories()` automatically after each `run()` and `run_halving()`.
- `generate_feedback` — scalar scores → structured diagnostic text for GEPA's `reflection_lm`.
- `ResourceBudget` — hard limits on LLM calls, wall time, agents, and iterations. Prevents runaway costs in autonomous operation. Used by pipeline for pre-flight checks.

## Local Contracts

### Primary interface contract
- `GFLPipeline(student, trainset, train_field, val_field)` → `{best_optimizer, best_program, best_score, baseline, improvement, all_scores, budget, trend, total_improvement}`
- **SPRT post-compile validation**: Holdout is split via `HoldoutGate.split()` BEFORE any optimizer runs (Invariant 5: optimizer never sees holdout). After the best optimizer is selected, the pipeline runs `SelfEvolveEngine.validate_and_deploy()` (SPRT, p0=0.50, p1=0.65, α=0.05, β=0.20) on that holdout set. Result includes `{validation: {accepted, score, p_value, n_evaluated, early_stop}}`. If SPRT rejects the best candidate, the pipeline falls back to the baseline program (fail-safe, never deploys a regressor).
- Runs 4 optimizers in order: BootstrapFewShot → MIPROv2 → GEPA → Sequential (BetterTogether)
- Each optimizer receives a fresh `metric` function that compares `val_field` between prediction and gold
- GEPA and Sequential receive `reflection_lm=teacher` via `LMRegistry.get_teacher()`
- `ResourceBudget.check()` called before each optimizer — raises `RuntimeError` on budget exceeded
- Best optimizer selected by max score across all completed optimizers
- Degraded runs (exception) fall back to student program with current best score
- `GFLPipeline.run_single(optimizer_name, student, trainset, auto_synthesize, auto_meta, min_examples)` → `{best_optimizer, best_score, best_program, trainset_size, synthesized, all_scores}`
- `run_single()` auto-synthesizes data when `len(trainset) < min_examples` and meta-learns best optimizer via `MetaOptimizer` when `auto_meta=True`
- `GFLPipeline.split_holdout(trainset, holdout_fraction=0.2)` → `(train, holdout)` — deterministic split with seed 42
- `GFLPipeline.gate_promotion(candidate, baseline, holdout, min_improvement=0.02)` → `{promoted, candidate_score, baseline_score, improvement}` — CI gate: only promotes if candidate beats baseline on hold-out by `min_improvement`

### Score contract
- Quality scores are `0.0-1.0` float, tracked via `LSETracker`
- Baseline measured on first 3 training examples before any optimizer
- `_evaluate` uses `generate_feedback(prediction)["score"]`
- LSE delta = `score - baseline`, only positive deltas count toward `total_improvement`

### Teacher LM contract
- Teacher LM (DeepSeek V4 Flash) used **only** for reflection in GEPA path
- Accessed via `LMRegistry.get_teacher()` — returns `None` if no teacher configured, caller must handle gracefully
- `DataSynthesizer` and `TaskDecomposer` also use teacher LM for generation and decomposition

### ResourceBudget contract
- Default limits: 100 LLM calls, 300s wall time, 10 agents, 20 iterations
- `PRODUCTION`: 200/600/20 — `LIGHT`: 50/180/5
- `check()` throws `RuntimeError` with message including the exceeded limit name and value
- `spend_tokens(tokens)` — track estimated token spend (e.g. 5K per optimizer compile per example)
- `tokens_spent` — property returning cumulative estimated tokens
- Budget tracking is called after each `_dispatch_optimizer()` in `run()`, `run_halving()`, and `compile_draft()`

### Paper-verified patterns — all compilable DSPy modules

All LLM-driven components use typed `dspy.Signature` + `dspy.Module` that can be compiled with any DSPy optimizer (`dspytools compile gepa <ModuleName> trainset.json`):

| Paper | Module | Signature | File |
|-------|--------|-----------|------|
| **LSE** (arXiv 2603.18620) | `LSESelfEvolveModule` | `current_context, performance_summary → new_context, improvement_estimate, changes_made` | `paper_optimizers.py` |
| **LSE** | `LSETreeExplorer` | Tree-guided evolution with UCB selection. `r_LSE = R̄(c₁) − R̄(c₀)` — rewards improvement, not absolute score. Uses `LSESelfEvolveModule` for context evolution (paper Alg.1 line 8) + holdout evaluation (paper Eq.4). | `paper_optimizers.py` |
| **SPIN** (arXiv 2401.01335) | `SpinDiscriminateModule` | `gold_output, generated_output → score, rationale` | `paper_optimizers.py` |
| **SPIN** | `SPINOptimizer` | Self-play discrimination loop using `SpinDiscriminateModule` for teacher LM judgment. | `paper_optimizers.py` |
| **MetaSPO** (arXiv 2505.09666) | `MetaPromptOptimizer` | Bilevel system prompt optimization. Returns `{iterations, final_score, final_meta_prompt}`. | `paper_optimizers.py` |
| **Trace2Skill** (arXiv 2603.25158) | `ErrorAnalystModule` | `trajectory_raw, expected_output, current_skill, iteration → patch_content, patch_section, decision, confidence, root_cause` | `consolidation.py` |
| **Trace2Skill** | `SuccessAnalystModule` | `task_description, output, score, current_skill → patterns_json, count` | `consolidation.py` |
| **Trace2Skill** | `MergeOperatorModule` | `patch_a/b_section/content/confidence, current_skill → merged_content, decision, confidence` | `consolidation.py` |
| **Trace2Skill** | `SkillConsolidator` | 3-stage pipeline (Rollout → Analyze → Consolidate). Supports `merge_width` (B_merge, default 2) for configurable hierarchical merge branching. Outputs skills to `~/.config/dspytools/skills/`. | `consolidation.py` |
| **GEPA** (arXiv 2507.19457) | `GEPAParetoFrontier` | Pareto frontier with coverage-weighted selection. Candidates dominate if score not exceeded by any frontier member. | `paper_optimizers.py` |
| **GRAO** | `GRAOMetaOptimizer` | Meta-learner tracking `improvement_rate` per (task_type, optimizer) pair. | `paper_optimizers.py` |
| **OPSD** (arXiv 2607.02234) | `PurifiedOPSDModule` | `reference_output, gold_output, question → purified_reference, original_weight, reference_weight, question_weight` | `paper_optimizers.py` |
| **R-Zero** (arXiv 2508.05004) | `ChallengerSolver` | Zero-data co-evolution. Two-model curriculum using `dspy.Predict` programs. `co_evolve()` runs rounds reporting accuracy. | `synthetic.py` |

### Shared state files
All persistent state lives under `~/.config/dspytools/`:
| File | Owner | Content |
|------|-------|---------|
| `lse_log.json` | LSETracker | Iteration history, best score, total improvement |
| `meta_optimizer.json` | MetaOptimizer | Trial history, per-complexity recommendations |
| `grao_log.json` | GRAOMetaOptimizer | Trial history, learned strategies, error patterns |
| `skills/*.json` | SkillConsolidator | Mined success/error patterns per source |

### Legacy compatibility
- Old `compile()` was `(compiled_program, result_dict)` — new `run_single()` returns `dict` with `best_program` key instead.
- `_dispatch_optimizer()` now centralizes all optimizer dispatch for both `run()` and `run_single()`.
- Supports 6 optimizers via `_dispatch_optimizer`: bootstrap_few_shot, mipro, gepa, sequential, knn, labeled_few_shot
- Falls back to `LabeledFewShot` for unrecognized optimizer names

### Multi-Fidelity Early Pruning (Successive Halving)
- `GFLPipeline.run_halving(student, trainset, train_field, val_field, prune_fraction, probe_fraction, min_examples)` → `{best_optimizer, best_score, best_program, baseline, improvement, all_scores, probe_scores, full_scores, survivors, pruned, budget, trend, total_improvement}`
- Runs 4 optimizers on ~10% probe subset first, prunes bottom `prune_fraction` (default 50%, keeping 2), runs survivors on full dataset
- Deterministic probe split with seed 42
- Falls back to probe scores for optimizers that fail during full run
- Best overall score across both phases (prefers full-score, uses probe as fallback)
- Configurable optimizer list via `optimizers` parameter — not hardcoded to 4 defaults
- `--auto-suggest` CLI flag uses `MetaOptimizer` to select best optimizer list by dataset size/complexity (additional 2x speedup on top of halving's 75%)
- See `run_halving()` for details and `run()` for standard 4-way comparison

### Speculative Compilation (Drafting)
- `GFLPipeline.compile_draft(student, trainset, optimizer_name, draft_rounds, polish_rounds)` → `{optimizer, draft_score, polished_score, improvement, draft_rounds, polish_rounds, best_program, teacher_used}`
- Phase 1: `draft_rounds` (default 3) optimization passes with student LM (`LMRegistry.get_or_default()`)
- Phase 2: `polish_rounds` (default 1) optimization passes with teacher LM (`LMRegistry.get_teacher()`)
- Uses `dspy.context(lm=..., temperature=...)` to switch LM for each phase
- Returns `round_scores` array with per-round scores for granular tracking
- Reduces teacher LM API costs by 3-5x for iterative optimizers like GEPA
- Falls back to draft result gracefully if no teacher LM configured

### SPRT post-compile validation
- `GFLPipeline.run()` splits holdout via `HoldoutGate.split()` BEFORE any optimizer runs (Invariant 5: optimizer trains on `train_data` only, holdout reserved for SPRT)
- After best optimizer is selected, runs `SelfEvolveEngine.validate_and_deploy()` (SPRT) on the reserved holdout
- Returns `{accepted, score, p_value, n_evaluated, early_stop}` — provides deploy decision and confidence metric
- Cost-aware: max 50 evaluations by default, early termination on clear wins (typically ~12 examples)

### CLI integration
- `gfl synthesize` → `DataSynthesizer.generate()`
- `gfl meta-optimize` → `MetaOptimizer.select_optimizer()`
- `gfl decompose` → `TaskDecomposer.decompose()`
- `gfl ab-test` → `ABTest.run()`
- `gfl consolidate` → `SkillConsolidator.consolidate()` (Trace2Skill)
- `gfl spin` → `SPINOptimizer.optimize()` (arXiv 2401.01335)
- `gfl lse` → `LSETreeExplorer.explore()` (arXiv 2603.18620)
- `gfl gepa` → `GEPAParetoFrontier.select_next()` (arXiv 2507.19457)
- `gfl opsd` → `PurifiedOPSDOptimizer.iterate()` (arXiv 2607.02234)
- `compile gfl` → `GFLPipeline.run()` (4-way comparison), `GFLPipeline.run_halving()` (when `--halving`), or `GFLPipeline.run_single()` when `--single` is specified
- `compile gepa --draft` → `GFLPipeline.compile_draft()` for speculative compilation
- `gfl opsd` → `PurifiedOPSDOptimizer.iterate()` (arXiv 2607.02234)

## Work Guidance

- All LLM-driven paper components use compilable `dspy.Module` subclasses — never raw `lm(prompt)` calls. Compile them: `dspytools compile gepa ErrorAnalystModule trainset.json`
- **Fail-fast pattern**: DSPy module constructors (`ErrorAnalystModule`, `SuccessAnalystModule`, `MergeOperatorModule`, `LSESelfEvolveModule`, `SpinDiscriminateModule`) instantiate directly — no try/except masking errors. Only LM *calls* and runtime operations (Redis down, network timeout) use graceful degradation via try/except.
- Always prefer `GFLPipeline(mode="single").run_single()` for single-optimizer compiles
- When adding a new optimizer to `GFLPipeline.run()`, add it to the `optimizers` dict in `_dispatch_optimizer()` with a try/except block — this auto-enables it for both `run()` and `run_single()`
- New optimizers must respect `ResourceBudget.check()` and return a compiled program compatible with `_evaluate()`
- LSE pattern: use `LSETracker.record()` after each optimization step to maintain delta history
- Quality scores must be deterministic for the same input — avoid randomness in `generate_feedback`
- When using `GEPAParetoFrontier`, call `add()` after each candidate and `select_next()` before the next mutation
- `GRAOMetaOptimizer.learn_from_trial()` should be called at the end of each optimization cycle to build meta-knowledge
- State files are JSON — handle parse errors gracefully (log and continue)
- Do not hardcode LM model names — always use `LMRegistry.get_teacher()` or `LMRegistry.get_or_default()`

## Verification

No dedicated test suite exists yet. Verification is manual via CLI commands:
```bash
dspytools compile gfl --program test --trainset data.json
dspytools gfl spin --module test --trainset data.json
dspytools gfl lse --module test --trainset data.json
dspytools gfl gepa --module test --trainset data.json
dspytools gfl opsd --module test --trainset data.json --beta 1.0
dspytools gfl consolidate --program-id test
```

Quality contracts to verify manually:
- Scores are always `0.0-1.0` float
- `GFLPipeline.run()` returns all expected keys
- `ResourceBudget.check()` raises `RuntimeError` on limit breach with descriptive message
- `LSETracker` persists and reloads from `lse_log.json`
- `compile_grpo()` falls back gracefully if GRPO is unavailable
- All shared state files survive partial writes (JSON not written until fully constructed)

## Child DOX Index

No subdirectories — all files are flat in `src/dspytools/gfl/`. No child AGENTS.md files exist.
