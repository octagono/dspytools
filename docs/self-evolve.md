# Self-Evolve Engine

The self-evolving agent system enables DSPyTools to improve its own programs through morphology tracking, UCB exploration, SPRT validation, and Meta Agent Search.

## Core Components

### `SelfEvolveEngine`
The central engine orchestrating all self-evolution capabilities:
```python
from dspytools.evolve.self_evolve import SelfEvolveEngine

engine = SelfEvolveEngine()
```

#### `validate_and_deploy()` — SPRT Validation
Uses Sequential Probability Ratio Test (SPRT) to determine if a compiled program beats the baseline with minimal evaluations:
```python
result = engine.validate_and_deploy(
    baseline_program, new_program_id, holdout_set,
    alpha=0.05, beta=0.2
)
# → {"deployed": True, "p_value": 0.003, "evidence_ratio": 12.4}
```
Early stopping on clear wins/losses saves API tokens.

#### `archive_search()` — Meta Agent Search
Searches the compiled program archive for past programs similar to a query, returning candidate optimizers and architectures:
```python
results = engine.archive_search("extract function docstrings", top_k=3)
# → [{"program_id": "...", "optimizer": "mipro", "score": 0.92}, ...]
```

#### `suggest_optimizer()`
Recommends the best optimizer for a given task description based on archive history.

### `MorphologyTracker`
Tracks program architecture changes across generations — detects growth, shrinkage, and structural drift.

### `UCBExplorer`
Upper Confidence Bound explorer for optimizer selection. Balances exploration of untested optimizers against exploitation of known good ones.

### `SkillGraph`
Graph of skill dependencies. When a skill improves, all downstream skills can be queued for re-optimization.

## Drift→Recompile Cascade

When `SelfEvolveEngine.on_compile()` records a successful optimization, it also:
1. Calls `SkillGraph.transitive_dependents()` to find all downstream skills
2. Calls `DriftMonitor.request_recompile()` for each downstream skill
3. CLI: `dspytools self auto-fix --no-dry-run --auto-fix` processes the queue
4. MCP tool: `drift_auto_fix` inspects and applies pending recompiles

This ensures that improving one skill automatically propagates optimization to everything that depends on it.

## Harness Layers

| Layer | File | Role |
|-------|------|------|
| H1 Action | `layers/action.py` | Primitive tool execution |
| H2 Contract | `layers/contract.py` | Pre/post-condition contracts on tools |
| H3 Trajectory | `layers/trajectory.py` | Multi-step plan decomposition |

## Resources

- Engine: `src/dspytools/evolve/self_evolve.py`
- Router: `src/dspytools/evolve/router.py`
- Layers: `src/dspytools/evolve/layers/`
