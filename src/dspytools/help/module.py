"""DSPy HelpModule — answers CLI questions via local LLM.

Signature: "command, subcommands, examples -> answer: str"
Self-optimized via LabeledFewShot + GEPA on first --help invocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy


class HelpModule(dspy.Module):
    """DSPy module that generates contextual CLI help.

    Uses ChainOfThought for step-by-step reasoning about what the
    user needs to know for a given command.
    """

    def __init__(self):
        super().__init__()
        self.answer = dspy.ChainOfThought(
            dspy.Signature(
                "command, subcommands, examples -> answer: str",
                "You are a CLI help assistant for dspytools, a tool that manages "
                "DSPy programs, MCP agents, LLM servers, and hot-swap inference. "
                "Given the command the user typed and the available subcommands, "
                "provide clear, concise help. Include examples when relevant. "
                "Format the response as plain text, no markdown.",
            )
        )

    def forward(
        self, command: str, subcommands: str, examples: str = ""
    ) -> dspy.Prediction:
        return self.answer(command=command, subcommands=subcommands, examples=examples)
