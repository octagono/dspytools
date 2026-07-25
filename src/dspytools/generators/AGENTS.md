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

- `SignatureGeneratorDSPy` is a `dspy.Module` subclass using `dspy.ChainOfThought`.
- `forward(signature_str: str, class_name: str, instructions: str) → dspy.Prediction`
  - Input: DSPy signature string (`"question -> answer"`), class name, task instructions
  - Output: `code` (Python class definition), `num_fields` (count of input + output fields), `warnings` (any issues)
- **100% LLM-generated code** — no string formatting fallback. The LLM produces valid Python.
- **Compilable**: can be optimized with `dspytools compile` to improve generation quality over time.

### `module_generator.py` — ModuleGeneratorDSPy

- `ModuleGeneratorDSPy` is a `dspy.Module` subclass using `dspy.ChainOfThought`.
- `forward(module_name: str, signature_str: str, module_type: str, description: str) → dspy.Prediction`
  - Input: module name, DSPy signature, module type (e.g., `"ChainOfThought"`), description
  - Output: `code` (Python module definition), `has_tools` (bool), `input_fields` (comma-separated)
- **100% LLM-generated code** — the LLM produces complete DSPy module definitions.
- **Compilable**: can be optimized with `dspytools compile` for higher quality code generation.

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

No automated tests exist yet. Manual smoke test:
```python
from dspytools.generators import SignatureGeneratorDSPy, ModuleGeneratorDSPy

sig_gen = SignatureGeneratorDSPy()
result = sig_gen("question -> answer", "QASig", "Answer questions")
assert result.code and "class" in result.code

mod_gen = ModuleGeneratorDSPy()
result = mod_gen("QA", "question -> answer", "ChainOfThought", "QA module")
assert result.code and result.has_tools is not None
```

## Child DOX Index

No subdirectories — this directory contains only three `.py` files and this `AGENTS.md`.
