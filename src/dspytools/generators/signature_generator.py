"""SignatureGeneratorDSPy — compilable ChainOfThought module that generates
signature Python code via the LLM.

The generator is a real ``dspy.Module`` (``forward`` + ``named_predictors``), so
any DSPy optimizer can compile it:
    from dspytools.generators import SignatureGeneratorDSPy
    from dspytools.core._dspy import dspy

    gen = SignatureGeneratorDSPy()
    compiled = dspy.BootstrapFewShot().compile(gen, trainset=...)
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


class SignatureGeneratorDSPy(dspy.Module):
    """DSPy module that generates signature Python code via LLM.

    Uses ChainOfThought with a descriptive signature so the LLM outputs
    valid complete Python code for a Signature class.

    ``forward`` returns the raw ``dspy.Prediction`` (what optimizers trace and
    score); ``generate`` is the tuple-returning convenience wrapper the CLI
    uses.
    """

    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(
            "prompt_text: str, class_name: str, task_instructions: str -> "
            "generated_code: str, field_count: int, warnings: list[str]"
        )

    def forward(
        self, prompt_text: str, class_name: str, task_instructions: str
    ) -> dspy.Prediction:
        """Generate signature Python code.

        Args:
            prompt_text: The raw prompt string (e.g. "question -> answer")
            class_name: PascalCase name for the generated Signature class
            task_instructions: Instructions/docstring for the signature

        Returns:
            ``dspy.Prediction`` with ``generated_code``, ``field_count`` and
            ``warnings``.
        """
        result = self.predictor(
            prompt_text=prompt_text,
            class_name=class_name,
            task_instructions=task_instructions,
        )
        return dspy.Prediction(
            generated_code=_strip_code_fences(getattr(result, "generated_code", "")),
            field_count=int(getattr(result, "field_count", 0) or 0),
            warnings=list(getattr(result, "warnings", []) or []),
        )

    def generate(
        self, prompt_text: str, class_name: str, task_instructions: str
    ) -> tuple[str, int, list[str]]:
        """Tuple-returning wrapper around ``forward`` for CLI callers."""
        prediction = self(prompt_text, class_name, task_instructions)
        return (
            prediction.generated_code,
            prediction.field_count,
            prediction.warnings,
        )
