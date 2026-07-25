"""Self-Discover decomposition — auto-break complex tasks into sub-modules.

GFL Stage: Compose

Like Hurricane/STORM, uses teacher LM to propose task decomposition.
"""

from __future__ import annotations

from dspytools.core.setup import LMRegistry


class TaskDecomposer:
    """Decompose complex tasks into DSPy sub-modules (Self-Discover pattern).

    Teacher LM proposes: sub-tasks, their signatures, and how they connect.
    """

    def __init__(self):

        self.teacher = LMRegistry.get_teacher() or LMRegistry.get_or_default()

    def decompose(self, task_description: str) -> dict:
        """Break a complex task into DSPy sub-modules."""
        # Simplified: use pattern matching for common tasks
        if (
            "llms.txt" in task_description.lower()
            or "repository" in task_description.lower()
        ):
            return {
                "task": task_description,
                "sub_tasks": [
                    {
                        "name": "AnalyzeRepository",
                        "signature": "repo_url, file_tree, readme_content -> purpose, concepts, architecture",
                        "type": "cot",
                        "depends_on": [],
                    },
                    {
                        "name": "AnalyzeStructure",
                        "signature": "file_tree, package_files -> directories, entry_points, dev_info",
                        "type": "cot",
                        "depends_on": [],
                    },
                    {
                        "name": "GenerateDoc",
                        "signature": "purpose, concepts, architecture, directories, entry_points, dev_info -> llms_txt",
                        "type": "cot",
                        "depends_on": ["AnalyzeRepository", "AnalyzeStructure"],
                    },
                ],
                "parallel": [["AnalyzeRepository", "AnalyzeStructure"]],
                "sequential": ["GenerateDoc"],
            }

        # Generic decomposition
        len(task_description.split())
        return {
            "task": task_description,
            "sub_tasks": [
                {
                    "name": "Analyze",
                    "signature": "input -> analysis",
                    "type": "cot",
                    "depends_on": [],
                },
                {
                    "name": "Generate",
                    "signature": "analysis -> output",
                    "type": "cot",
                    "depends_on": ["Analyze"],
                },
            ],
            "parallel": [],
            "sequential": ["Analyze", "Generate"],
        }

    def generate_code(
        self, decomposition: dict, module_name: str = "GeneratedModule"
    ) -> str:
        """Generate DSPy module code from decomposition with correct data flow chaining."""
        sub_tasks = decomposition["sub_tasks"]

        # Parse input and output fields for each subtask
        def _parse_sig(sig: str):
            left, _, right = sig.partition("->")
            inputs = [f.strip() for f in left.split(",") if f.strip()]
            outputs = [f.strip() for f in right.split(",") if f.strip()]
            return inputs, outputs

        # Build output map: field → producing subtask
        all_outputs: dict[str, str] = {}
        for task in sub_tasks:
            _, outputs = _parse_sig(task["signature"])
            for out in outputs:
                # Strip type annotations (e.g. "field: str" -> "field")
                field_name = out.split(":")[0].strip()
                all_outputs[field_name] = task["name"].lower()

        # Collect all input fields across all subtasks
        all_inputs: set[str] = set()
        for task in sub_tasks:
            inp, _ = _parse_sig(task["signature"])
            for field in inp:
                field_name = field.split(":")[0].strip()
                all_inputs.add(field_name)

        # External inputs: fields that are not produced by any subtask
        external_inputs = [f for f in sorted(all_inputs) if f not in all_outputs]
        external_params = ", ".join(f"{f}: str" for f in external_inputs)

        # Determine terminal tasks (nothing depends on them)
        dependents: set[str] = set()
        for task in sub_tasks:
            for dep in task.get("depends_on", []):
                dependents.add(dep.lower())
        terminal = [t["name"] for t in sub_tasks if t["name"].lower() not in dependents]

        code = f'''"""Auto-generated DSPy module from task decomposition."""


from dspytools.core._dspy import dspy


class {module_name}(dspy.Module):
    """Auto-generated from: {decomposition["task"][:80]}"""

    def __init__(self):
        super().__init__()
'''
        for task in sub_tasks:
            sig = task["signature"]
            code += (
                f'        self.{task["name"].lower()} = dspy.ChainOfThought("{sig}")\n'
            )

        code += f"""
    def forward(
        self,
        {external_params},
    ) -> dspy.Prediction:
"""
        # Generate call blocks per subtask
        for task in sub_tasks:
            name_l = task["name"].lower()
            inp, _ = _parse_sig(task["signature"])
            task.get("depends_on", [])

            # Determine input field values: from external params or dependency outputs
            call_args: list[str] = []
            for field in inp:
                field_name = field.split(":")[0].strip()
                if field_name in all_outputs and all_outputs[field_name] != name_l:
                    # Comes from a dependency subtask's output
                    source_task = all_outputs[field_name]
                    call_args.append(f"{field_name}={source_task}.{field_name},")
                else:
                    # Comes from external forward param
                    call_args.append(f"{field_name}={field_name},")

            args_str = "\n            ".join(call_args)
            code += f"""        {name_l} = self.{name_l}(
            {args_str}
        )

"""

        # Determine what to return: group terminal task outputs
        return_fields: list[str] = []
        for task in sub_tasks:
            name_l = task["name"].lower()
            if task["name"] in terminal:
                _, outputs = _parse_sig(task["signature"])
                for out in outputs:
                    field_name = out.split(":")[0].strip()
                    return_fields.append(f"{field_name}={name_l}.{field_name},")

        # If no terminal outputs found, return the last sub_task's outputs
        if not return_fields:
            last = sub_tasks[-1]
            _, outputs = _parse_sig(last["signature"])
            name_l = last["name"].lower()
            for out in outputs:
                field_name = out.split(":")[0].strip()
                return_fields.append(f"{field_name}={name_l}.{field_name},")

        ret_str = "\n            ".join(return_fields)
        code += f"""        return dspy.Prediction(
            {ret_str}
        )
"""
        return code
