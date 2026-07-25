"""H2 Contract Layer — type contracts and runtime validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContractViolation:
    field: str
    expected: str
    actual: str
    message: str = ""


@dataclass
class ContractResult:
    valid: bool
    violations: list[ContractViolation] = field(default_factory=list)


class ContractLayer:
    TYPES = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    @classmethod
    def validate_inputs(cls, inputs: dict[str, Any], schema: dict) -> ContractResult:
        violations = []
        if not schema:
            return ContractResult(valid=True)
        required = [
            k.strip()
            for k in schema.get("inputs", schema.get("required", "")).split(",")
            if k.strip()
        ]
        for fname in required:  # noqa: F402
            if fname not in inputs:
                violations.append(
                    ContractViolation(
                        field=fname,
                        expected="present",
                        actual="missing",
                        message=f"Required '{fname}' is missing",
                    )
                )
            else:
                tname = schema.get("types", {}).get(fname, "str")
                expected = cls.TYPES.get(tname, str)
                if not isinstance(inputs[fname], expected):
                    violations.append(
                        ContractViolation(
                            field=fname,
                            expected=tname,
                            actual=type(inputs[fname]).__name__,
                            message=f"'{fname}' expected {tname}",
                        )
                    )
        return ContractResult(valid=len(violations) == 0, violations=violations)

    @classmethod
    def validate_outputs(cls, outputs: dict[str, Any], schema: dict) -> ContractResult:
        violations = []
        if not schema:
            return ContractResult(valid=True)
        expected_keys = [
            k.strip()
            for k in schema.get("outputs", schema.get("expected", "")).split(",")
            if k.strip()
        ]
        for key in expected_keys:
            if key not in outputs:
                violations.append(
                    ContractViolation(
                        field=key,
                        expected="present",
                        actual="missing",
                        message=f"Expected '{key}' is missing",
                    )
                )
            elif outputs[key] is None or outputs[key] == "":
                violations.append(
                    ContractViolation(
                        field=key,
                        expected="non-empty",
                        actual="empty",
                        message=f"'{key}' is empty",
                    )
                )
        return ContractResult(valid=len(violations) == 0, violations=violations)

    @classmethod
    def infer_schema(cls, signature_str: str) -> dict:
        if "->" not in signature_str:
            return {}
        inputs_part, outputs_part = signature_str.split("->", 1)
        input_types, input_list = {}, []
        for fld in inputs_part.split(","):  # noqa: F402
            fld = fld.strip()
            if ":" in fld:
                name, tpart = fld.split(":", 1)
                input_types[name.strip()] = tpart.strip()
                input_list.append(name.strip())
            else:
                input_list.append(fld)
                input_types[fld] = "str"
        output_types, output_list = {}, []
        for fld in outputs_part.split(","):  # noqa: F402
            fld = fld.strip()
            if ":" in fld:
                name, tpart = fld.split(":", 1)
                output_types[name.strip()] = tpart.strip()
                output_list.append(name.strip())
            else:
                output_list.append(fld)
                output_types[fld] = "str"
        return {
            "inputs": ", ".join(input_list),
            "outputs": ", ".join(output_list),
            "types": {**input_types, **output_types},
        }
