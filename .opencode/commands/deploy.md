---
description: Validate deployment readiness of a compiled program
agent: dspy-evaluator
subtask: true
---

Validate a compiled program before deployment.

1. List available programs: use the list_compiled_runs MCP tool
2. Check for quality drift: use the drift_status MCP tool
3. Run SPRT validation: use the validate_deploy MCP tool
4. Compare if multiple candidates: `scripts/dspytools compare [run_a] [run_b] [devset.json]`
5. Report: DEPLOY / HOLD / REJECT with evidence
