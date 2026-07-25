# DSPyTools Documentation

## Setup
- [Quick Start](../README.md#quick-start) — install, configure, run
- [dev-local](dev-local.md) — one-command local development stack
- [CLI Reference](../README.md#features) — all 23 command groups with subcommand counts

## Architecture
- [Package Map](architecture.md) — module dependency graph, data flow, 96 source files
- [GFL Pipeline](gfl-pipeline.md) — 4-way comparison, Successive Halving, Speculative Compile
- [Self-Evolve Engine](self-evolve.md) — Gödel Agent, SPRT validation, Meta Agent Search
- [MCP Server](mcp-server.md) — 65 tools, 9 resources, 3 prompts, transport
- [LoRA Integration](lora-integration.md) — adapter management, distillation, Colab training
- [Cost Tracking](architecture.md#cost-tracking) — token counting and cost estimation
- [Drift Detection](architecture.md#drift-detection) — quality monitoring (core/drift_monitor.py)
- [Holdout Gate](architecture.md#holdout-gate) — programmatic Invariant 5 enforcement
- [Caching](architecture.md#caching) — AST-based dependency caching (generate/cache.py)

## Papers
- [arXiv Implementations](../README.md#gfl-pipeline-generative-feedback-loop) — 11 arXiv papers implemented in dspytools

## Development
- [DOX Tree](../AGENTS.md) — 12 child AGENTS.md files governing the codebase
