# DOX — generators

## Purpose

DSPy-native code generators — compilable `dspy.ChainOfThought` modules that produce Python code for DSPy signatures and modules via the LLM. These generators are themselves compilable with any DSPy optimizer (BootstrapFewShot, MIPROv2, GEPA, etc.).

Consumed by `commands/signature.py` and `commands/module.py` — the generators are defined here as an extractable, compilable package rather than inline in the command files.

## Ownership

Three source files:

| File | Role |
|------|------|
| `__init__.py` | Public API — re-exports `SignatureGeneratorDSPy`, `ModuleGeneratorDSPy` |
| `signature_generator.py` | `SignatureGeneratorDSPy` — compilable `dspy.ChainOfThought` that generates DSPy signature class definitions |
| `module_generator.py` | `ModuleGeneratorDSPy` — compilable `dspy.ChainOfThought` that generates DSPy module class definitions |

## Local Contracts

### `signature_generator.py` — SignatureGeneratorDSPy

- `SignatureGeneratorDSPy` is a **real `dspy.Module` subclass** (`super().__init__()` + `self.predictor = dspy.ChainOfThought(...)`) — `named_predictors()` and `forward` exist, so optimizers can actually compile it. Before, it was a plain class whose docstrings and this doc claimed compilability that DSPy cannot use.
- `forward(prompt_text: str, class_name: str, task_instructions: str) → dspy.Prediction`
  - Input: DSPy signature string (`"question -> answer"`), class name, task instructions
  - Output: `generated_code` (Python class definition), `field_count` (count of input + output fields), `warnings` (any issues)
- `generate(...) → (code, field_count, warnings)` — tuple wrapper over `forward`, used by `commands/signature.py`.
- **100% LLM-generated code** — no string formatting fallback. The LLM produces valid Python.
- **Compilable**: `dspy.BootstrapFewShot().compile(gen, trainset=...)` traces the real predictor.

### `module_generator.py` — ModuleGeneratorDSPy

- `ModuleGeneratorDSPy` is a **real `dspy.Module` subclass** with `forward` + `named_predictors()`.
- `forward(module_name: str, signature_str: str, module_type: str, description: str) → dspy.Prediction`
  - Input: module name, DSPy signature, module type (e.g., `"ChainOfThought"`), description
  - Output: `module_code` (Python module definition), `has_tools` (bool), `out_fields` (list of output field names)
- `generate(...) → (code, has_tools, out_fields)` — tuple wrapper over `forward`, used by `commands/module.py`.
- **100% LLM-generated code** — the LLM produces complete DSPy module definitions.
- **Compilable**: `dspy.MIPROv2().compile(gen, trainset=...)` traces the real predictor.

### Shared contracts

- Both generators use `from dspytools.core._dspy import dspy` — never `import dspy` directly.
- Both use `dspy.ChainOfThought` with descriptive signatures.
- The generators are the **single source of truth** for DSPy code generation. `commands/signature.py` and `commands/module.py` import from here rather than defining inline generators.
- Reserved field names (`instructions`, `output_fields`) are avoided in generator signatures — use `task_instructions` instead.

## Work Guidance

- Keep generators as pure DSPy modules — no file I/O, no CLI integration.
- When improving generation quality, compile the generators against a curated trainset of high-quality signature/module definitions.
- New generator types (e.g., optimizer generator, pipeline generator) should follow the same pattern: compilable `dspy.ChainOfThought` modules in this directory.

## Verification

`uv run --no-sync pytest tests/test_serving_integrity.py -q` exercises the generator contracts indirectly (`module new --signature` passes `Signature.signature` to `ModuleGeneratorDSPy.generate`).

Manual smoke test (needs an LM — not part of the offline suite):
```python
from dspytools.generators import SignatureGeneratorDSPy, ModuleGeneratorDSPy

sig_gen = SignatureGeneratorDSPy()
assert "predictor" in dict(sig_gen.named_predictors())  # a real dspy.Module
result = sig_gen.generate("question -> answer", "QASig", "Answer questions")
assert result[0] and "class" in result[0]

mod_gen = ModuleGeneratorDSPy()
code, has_tools, out_fields = mod_gen.generate(
    "QA", "question -> answer", "ChainOfThought", "QA module"
)
assert code
```

## Child DOX Index

No subdirectories — this directory contains only three `.py` files and this `AGENTS.md`.
