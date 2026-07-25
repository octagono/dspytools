"""DSPy-native generators — compilable ChainOfThought modules for code generation.

These generators produce Python code via the LLM and are themselves compilable
with any DSPy optimizer (BootstrapFewShot, MIPROv2, GEPA, etc.).

Usage:
    from dspytools.generators import SignatureGeneratorDSPy, ModuleGeneratorDSPy

    sig_gen = SignatureGeneratorDSPy()
    code, count, warns = sig_gen.generate("question -> answer", "QASig",
                                          "Given a question, produce an answer")

    mod_gen = ModuleGeneratorDSPy()
    code, has_tools, fields = mod_gen.generate("QA", "question -> answer",
                                               "ChainOfThought", "QA module")
"""

from dspytools.generators.module_generator import ModuleGeneratorDSPy
from dspytools.generators.signature_generator import SignatureGeneratorDSPy

__all__ = [
    "SignatureGeneratorDSPy",
    "ModuleGeneratorDSPy",
]
