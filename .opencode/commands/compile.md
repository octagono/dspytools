---
description: Compile a DSPy module with optimizer selection
agent: dspy-optimizer
subtask: true
---

Compile a DSPy module with the appropriate optimizer.

1. Check available modules: `ls modules/`
2. Check available optimizers via MCP: use the list_optimizers tool
3. If unsure which optimizer fits the task profile, consult the optimizer selection guide in the dspy-optimizer agent prompt
4. Run compilation: `scripts/dspytools compile [optimizer] [module_name] [trainset.json]`
5. Verify the compiled run: `scripts/dspytools compile list`
6. Check cost: use the compile_cost MCP tool
7. Validate with SPRT: use the validate_deploy MCP tool
