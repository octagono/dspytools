"""ModuleGeneratorDSPy — compilable ChainOfThought module that generates
module Python code via the LLM.

This module is itself a DSPy program — it can be compiled with any DSPy optimizer:
    from dspytools.generators import ModuleGeneratorDSPy
    from dspytools.core._dspy import dspy

    gen = ModuleGeneratorDSPy()
    compiled = dspy.MIPROv2().compile(gen, trainset=...)
"""

from __future__ import annotations


class ModuleGeneratorDSPy:
    """DSPy module for generating module Python code via LLM.

    Uses ChainOfThought with a descriptive signature so the LLM outputs
    valid complete Python code for a dspy.Module subclass with a typed
    forward() method.

    The generator is itself a compilable DSPy module — extract it and run
    `dspytools compile` on it to boost code quality.
    """

    def __init__(self):
        from dspytools.core._dspy import dspy

        self.predictor = dspy.ChainOfThought(
            "module_name: str, sig_string: str, mod_type: str, "
            "task_instructions: str -> "
            "module_code: str, has_tools: bool, out_fields: list[str]"
        )

    def generate(
        self, module_name: str, sig_string: str, mod_type: str, task_instructions: str
    ) -> tuple[str, bool, list[str]]:
        """Generate module Python code using the LLM.

        Args:
            module_name: PascalCase name for the generated Module class
            sig_string: Signature string (e.g. "question -> answer")
            mod_type: Module type ("Predict", "ChainOfThought", etc.)
            task_instructions: Instructions/docstring for the module

        Returns:
            (code, has_tools, out_fields) tuple.
        """
        result = self.predictor(
            module_name=module_name,
            sig_string=sig_string,
            mod_type=mod_type,
            task_instructions=task_instructions,
        )
        code = getattr(result, "module_code", "")
        ht = bool(getattr(result, "has_tools", False) or False)
        of = list(getattr(result, "out_fields", []) or [])
        code = code.strip()
        # Strip markdown code fences that LLMs often wrap output in
        if code.startswith("```"):
            for prefix in ("```python", "```py", "```"):
                if code.startswith(prefix):
                    code = code[len(prefix) :]
                    break
            if code.endswith("```"):
                code = code[:-3]
        return code.strip(), ht, of
