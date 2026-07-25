"""SignatureGeneratorDSPy — compilable ChainOfThought module that generates
signature Python code via the LLM.

This module is itself a DSPy program — it can be compiled with any DSPy optimizer:
    from dspytools.generators import SignatureGeneratorDSPy
    from dspytools.core._dspy import dspy

    gen = SignatureGeneratorDSPy()
    compiled = dspy.BootstrapFewShot().compile(gen, trainset=...)
"""

from __future__ import annotations


class SignatureGeneratorDSPy:
    """DSPy module for generating signature Python code via LLM.

    Uses ChainOfThought with a descriptive signature so the LLM outputs
    valid complete Python code for a Signature class.

    The generator is itself a compilable DSPy module — extract it and run
    `dspytools compile` on it to boost code quality.
    """

    def __init__(self):
        from dspytools.core._dspy import dspy

        self.predictor = dspy.ChainOfThought(
            "prompt_text: str, class_name: str, task_instructions: str -> "
            "generated_code: str, field_count: int, warnings: list[str]"
        )

    def generate(
        self, prompt_text: str, class_name: str, task_instructions: str
    ) -> tuple[str, int, list[str]]:
        """Generate signature Python code using the LLM.

        Args:
            prompt_text: The raw prompt string (e.g. "question -> answer")
            class_name: PascalCase name for the generated Signature class
            task_instructions: Instructions/docstring for the signature

        Returns:
            (code, field_count, warnings) tuple.
        """
        result = self.predictor(
            prompt_text=prompt_text,
            class_name=class_name,
            task_instructions=task_instructions,
        )
        code = getattr(result, "generated_code", "")
        count = int(getattr(result, "field_count", 0) or 0)
        warns = list(getattr(result, "warnings", []) or [])
        code = code.strip()
        # Strip markdown code fences that LLMs often wrap output in
        if code.startswith("```"):
            for prefix in ("```python", "```py", "```"):
                if code.startswith(prefix):
                    code = code[len(prefix) :]
                    break
            if code.endswith("```"):
                code = code[:-3]
        return code.strip(), count, warns
