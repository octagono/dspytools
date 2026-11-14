"""ModuleGeneratorDSPy — compilable ChainOfThought module that generates
module Python code via the LLM.

The generator is a real ``dspy.Module`` (``forward`` + ``named_predictors``), so
any DSPy optimizer can compile it:
    from dspytools.generators import ModuleGeneratorDSPy
    from dspytools.core._dspy import dspy

    gen = ModuleGeneratorDSPy()
    compiled = dspy.MIPROv2().compile(gen, trainset=...)
"""

from __future__ import annotations

from dspytools.core._dspy import dspy


def _strip_code_fences(code: str) -> str:
    """Remove markdown code fences that LLMs often wrap code output in."""
    code = code.strip()
    if code.startswith("```"):
        for prefix in ("```python", "```py", "```"):
            if code.startswith(prefix):
                code = code[len(prefix) :]
                break
        code = code.removesuffix("```")
    return code.strip()


class ModuleGeneratorDSPy(dspy.Module):
    """DSPy module for generating module Python code via LLM.

    Uses ChainOfThought with a descriptive signature so the LLM outputs
    valid complete Python code for a dspy.Module subclass with a typed
    forward() method.

    ``forward`` returns the raw ``dspy.Prediction`` (what optimizers trace and
    score); ``generate`` is the tuple-returning convenience wrapper the CLI
    uses.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            "module_name: str, sig_string: str, mod_type: str, "
            "task_instructions: str -> "
            "module_code: str, has_tools: bool, out_fields: list[str]"
        )

    def forward(
        self, module_name: str, sig_string: str, mod_type: str, task_instructions: str
    ) -> dspy.Prediction:
        """Generate module Python code using the LLM.

        Args:
            module_name: PascalCase name for the generated Module class
            sig_string: Signature string (e.g. "question -> answer")
            mod_type: Module type ("Predict", "ChainOfThought", etc.)
            task_instructions: Instructions/docstring for the module

        Returns:
            ``dspy.Prediction`` with ``module_code``, ``has_tools`` and
            ``out_fields``.
        """
        result = self.predictor(
            module_name=module_name,
            sig_string=sig_string,
            mod_type=mod_type,
            task_instructions=task_instructions,
        )
        return dspy.Prediction(
            module_code=_strip_code_fences(getattr(result, "module_code", "")),
            has_tools=bool(getattr(result, "has_tools", False) or False),
            out_fields=list(getattr(result, "out_fields", []) or []),
        )

    def generate(
        self, module_name: str, sig_string: str, mod_type: str, task_instructions: str
    ) -> tuple[str, bool, list[str]]:
        """Tuple-returning wrapper around ``forward`` for CLI callers."""
        prediction = self(module_name, sig_string, mod_type, task_instructions)
        return prediction.module_code, prediction.has_tools, prediction.out_fields
