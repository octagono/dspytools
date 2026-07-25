---
name: dspy-gfl-pipeline
description: GFL (Generative Feedback Loop) pipeline — 4-way optimizer comparison with self-evolve learning
metadata:
  audience: developers
  workflow: optimization
---

## GFL Pipeline

The GFL pipeline runs a 4-way comparison of optimizers and tracks results for self-evolution.

### Standard Run
```bash
dspytools compile gfl <module> <trainset.json>
```
Runs: BootstrapFewShot → MIPROv2 → GEPA → Sequential. Picks best score.

### Fast Run (Recommended)
```bash
dspytools compile gfl <module> <trainset.json> --halving
```
Successive Halving: probes on 10% data, prunes bottom 50%, full run on survivors. Saves 2-3 teacher LM calls.

### Single Optimizer
```bash
dspytools compile gfl <module> <trainset.json> --single gepa
```

### Speculative Compilation (Cheapest)
```bash
dspytools compile gepa <module> <trainset.json> --draft
```
Student drafts 3 rounds, teacher polishes 1 round. 3-5x cost reduction.

### Paper-Backed Optimizers
- GEPA (arXiv 2507.19457): Pareto frontier, reflection LM
- SPIN (arXiv 2401.01335): Self-play discrimination
- MetaSPO (arXiv 2505.09666): Bilevel meta-learning
- R-Zero (arXiv 2508.05004): Challenger-Solver co-evolution

### Cost Awareness
Teacher LM (DeepSeek V4 Pro): ~$0.14/$0.28 per 1M tokens
Student LM (Qwen 3B local): free (vLLM)
