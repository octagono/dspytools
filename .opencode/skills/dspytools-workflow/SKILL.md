---
name: dspytools-workflow
description: Core dspytools development workflow — compile, evaluate, validate, deploy cycle
metadata:
  audience: developers
  workflow: dspytools
---

## DSPyTools Development Workflow

Follow this cycle when building and optimizing DSPy programs:

### 1. Create Signature
Generate a DSPy signature using the LLM-powered generator:
```bash
dspytools signature new "question: str = The user's question -> answer: str = The final answer" -n QASignature
dspytools signature new "question -> answer" --no-llm   # faster, no API call
```

The generator is itself a DSPy ChainOfThought module — run `dspytools compile` on `signature.py` to optimize code quality.

### 2. Create Module
Generate a DSPy module with typed forward():
```bash
dspytools module new QAModule --signature QASignature -t ChainOfThought
dspytools module new QA -t ChainOfThought --from-prompt "question -> answer"  # inline
dspytools module new Planner --no-llm  # rule-based fallback
```

The generator is itself a compilable DSPy module — `_ModuleGeneratorDSPy` in `module.py`.

### 3. Compile
Choose an optimizer based on dataset size:

| Size | Optimizer | Command |
|------|-----------|---------|
| <10 | labeled-few-shot | `dspytools compile labeled-few-shot <module> data.json` |
| 10-50 | bootstrap-few-shot | `dspytools compile bootstrap-few-shot <module> data.json` |
| 50-200 | mipro or gepa | `dspytools compile gepa <module> data.json` |
| >200 | gfl --halving | `dspytools compile gfl <module> data.json` |

### 4. Evaluate
```bash
dspytools evaluate run <run_id> <devset.json>
dspytools compile cost <run_id>
```

### 5. Validate with SPRT
Deployment gate — statistical validation:
```bash
dspytools validate_deploy <program_id>
```

### 6. Monitor for Drift
```bash
dspytools drift_status
dspytools drift_history <run_id>
```

## Generator Architecture (signature.py, module.py)

Both use the same 100% DSPy-native design:

- **`_SignatureGeneratorDSPy`** / **`_ModuleGeneratorDSPy`** — inline `dspy.ChainOfThought` classes that produce Python code via the LLM. **No string-formatting fallback.** The LLM IS the generator.
- Both use `dspy.ChainOfThought` with descriptive signatures — compilable with any DSPy optimizer.
- **No import from `modules/`**: defined inline in the command file (`.gitignore`d directory avoided).
- **`task_instructions`**: used instead of `instructions` (reserved name in `dspy.Signature`).
- `from dspytools.core._dspy import dspy` — never bare `import dspy`.
- Bracket-aware field splitting for `list[str]`, `dict[str, Any]`.
- Type alias conversion: `string → str`, `integer → int`, `boolean → bool`.

### Fast Path (Single Optimizer)
```bash
dspytools compile gfl <module> data.json --single gepa --halving
```
