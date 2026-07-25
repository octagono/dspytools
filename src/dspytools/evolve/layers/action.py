"""H1 Action Layer — Wraps DSPy modules as callable actions with metadata.

harness-so pattern: every DSPy program becomes an Action with
  - invoke(inputs) → structured output
  - bind(program) → Action
  - with_tools(tools) → Action with tools
  - Metadata: name, description, input_schema, output_schema
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dspytools.core.loaders import prediction_to_dict


@dataclass
class Action:
    """A callable DSPy action with metadata and tool binding."""

    name: str
    description: str = ""
    program: Any = None
    tools: list = field(default_factory=list)
    input_schema: dict = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def invoke(self, **inputs: Any) -> dict[str, Any]:
        """Execute the action with inputs, return structured output."""
        if self.program is None:
            raise RuntimeError(f"Action '{self.name}' has no bound program")

        result = self.program(**inputs)
        return prediction_to_dict(result)

    def bind(self, program: Any) -> Action:
        """Bind a DSPy program to this action."""
        self.program = program
        return self

    def with_tools(self, *tools: Any) -> Action:
        """Add tools to this action."""
        self.tools.extend(tools)
        return self

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
            "has_tools": len(self.tools) > 0,
        }


class ActionLayer:
    """H1: Manages actions — the API binding layer.

    Every DSPy program, compiled program, and skill becomes an Action.
    Agents compose actions into pipelines.
    """

    def __init__(self):
        self.actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        self.actions[action.name] = action

    def get(self, name: str) -> Action | None:
        return self.actions.get(name)

    def list_actions(self) -> list[dict]:
        return [a.to_dict() for a in self.actions.values()]

    def invoke(self, name: str, **inputs: Any) -> dict[str, Any]:
        action = self.get(name)
        if action is None:
            raise KeyError(f"Action '{name}' not found")
        return action.invoke(**inputs)

    def from_program(
        self,
        name: str,
        program: Any,
        description: str = "",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
    ) -> Action:
        """Create an action from a DSPy program."""
        action = Action(
            name=name,
            description=description,
            program=program,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
        )
        self.register(action)
        return action

    def from_skill(self, skill_name: str) -> Action | None:
        """Create an action from a compiled skill."""
        from dspytools.skills import SkillManager

        mgr = SkillManager()
        skill = mgr.loader.get(skill_name)
        if skill is None or not skill.has_program:
            return None

        if skill.path is None:
            return None

        from dspytools.core._dspy import dspy

        sig = skill.frontmatter.get("signature", "question -> answer")
        program = dspy.Predict(sig)
        program.load(str(skill.path / "program.json"))

        action = Action(
            name=skill_name,
            description=skill.description,
            program=program,
            input_schema={
                "inputs": skill.frontmatter.get("signature", "").split("->")[0].strip()
            },
            output_schema={
                "outputs": skill.frontmatter.get("signature", "").split("->")[1].strip()
                if "->" in skill.frontmatter.get("signature", "")
                else "output"
            },
        )
        self.register(action)
        return action

    def from_compiled_run(self, run_id: str) -> Action | None:
        """Create an action from a compiled program run."""
        from dspytools.core.hotswap import _load_program_from_run

        program = _load_program_from_run(run_id)
        if program is None:
            return None

        action = Action(
            name=run_id,
            description=f"Compiled program: {run_id}",
            program=program,
        )
        self.register(action)
        return action
