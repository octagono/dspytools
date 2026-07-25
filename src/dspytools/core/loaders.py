"""Shared dataset and module loading — Single Source of Truth.

All commands that load trainsets or modules should use these functions.
"""

from __future__ import annotations

import importlib.util as _util
import json as _json
import sys as _sys
from pathlib import Path

from dspytools.config.settings import modules_dir
from dspytools.core._dspy import dspy

# Optimization: cache for loaded DSPy modules — avoids re-import on repeated calls
_module_cache: dict[str, object] = {}


def get_example_inputs(ex) -> dict:
    """Extract input fields from a dspy.Example. SSOT for Example input access."""
    return ex.inputs()


def join_inputs(ex, exclude: str | None = None) -> str:
    """Extract input fields from a dspy.Example and join as newline-separated string.

    SSOT for converting example inputs to a prompt-ready string.
    Use exclude to skip a specific field (e.g., the output field).

    Falls back to vars(ex) if ex.inputs() is not available.
    """
    if hasattr(ex, "inputs"):
        inputs = ex.inputs()
    else:
        inputs = {k: getattr(ex, k) for k in vars(ex) if not k.startswith("_")}
    if exclude and exclude in inputs:
        del inputs[exclude]
    return "\n".join(str(v) for v in inputs.values())


def prediction_to_dict(pred) -> dict:
    """Convert a dspy.Prediction to a plain dict. SSOT for Prediction serialization.

    Handles all DSPy result types: Prediction with _output_field_names,
    dict-like objects, dataclass-like objects with _fields, and fallback str().
    """
    if hasattr(pred, "_output_field_names"):
        return {k: getattr(pred, k) for k in pred._output_field_names}
    if hasattr(pred, "items"):
        return dict(pred.items())
    if hasattr(pred, "_fields"):
        return {f: getattr(pred, f) for f in pred._fields}
    return {"output": str(pred)}


def _load_json_array(text: str) -> list[dict]:
    """Parse a JSON array string into list of dicts."""
    return _json.loads(text)


def _load_jsonl(text: str) -> list[dict]:
    """Parse JSONL text (one JSON object per line) into list of dicts."""
    items = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(_json.loads(line))
    return items


def load_trainset(path: str) -> list:
    """Load a DSPy trainset from a JSON or JSONL file. Auto-detects format.

    Supported formats (auto-detected by first non-whitespace char):
      - JSON array: [{"field1": "value1", ...}, ...]
      - JSONL:      {"field1": "value1", ...}\\n{"field1": "value2", ...}

    Input detection:
      - If "input" key present: uses .with_inputs("input")
      - If "inputs" key present: uses .with_inputs(*item["inputs"])
      - Otherwise: uses the first key as input
    """
    text = Path(path).read_text().strip()
    if not text:
        return []

    # Auto-detect JSONL (starts with '{') vs JSON array (starts with '[')
    if text[0] == "{":
        data = _load_jsonl(text)
    else:
        data = _load_json_array(text)

    examples = []
    for item in data:
        ex = dspy.Example(**item)
        if "input" in item:
            ex = ex.with_inputs("input")
        elif "inputs" in item:
            ex = ex.with_inputs(*item["inputs"])
        elif item:
            first_key = list(item.keys())[0]
            ex = ex.with_inputs(first_key)
        examples.append(ex)
    return examples


def load_jsonl(path: str) -> list:
    """Load examples from a JSONL file (one JSON object per line).

    Parses each line as a dict and returns lightweight attribute objects.
    If the 'input' field is a JSON string, it's parsed into individual fields.

    JSONL format (e.g. distill extraction):
        {"instruction": "...", "input": "{\\"question\\": \\"...\\"}", "output": "...", "score": 0.8}

    Returns objects with parsed input fields + output as attributes.
    """
    examples = []
    with open(Path(path)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = _json.loads(line)
            # If 'input' is a JSON string, parse it into individual fields
            raw_input = item.get("input", "{}")
            input_dict = (
                _json.loads(raw_input) if isinstance(raw_input, str) else raw_input
            )
            # Merge parsed inputs + output into a single attribute object
            attrs = {**input_dict, "output": item.get("output", "")}
            examples.append(type("JsonlExample", (), attrs)())
    return examples


def load_module_by_name(name: str):
    """Load a DSPy module from its generated .py file in modules_dir.

    Returns an instantiated module instance.
    Optimization: caches loaded instances by name — avoids re-importing
    and re-scanning dir() on repeated calls to the same module.
    """
    # Check cache first
    if name in _module_cache:
        return _module_cache[name]

    mod_path = modules_dir() / f"{name.lower()}.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Module '{name}' not found at {mod_path}")

    spec = _util.spec_from_file_location(name, str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module '{name}' from {mod_path}")

    mod = _util.module_from_spec(spec)
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)

    # Try exact match first, then PascalCase (e.g. "bugfinder" → "BugFinder",
    # "code_analyzer" → "CodeAnalyzer"), then scan all attrs for any DSPy Module subclass
    candidates = [name]
    pascal = "".join(w.capitalize() for w in name.replace("-", "_").split("_"))
    if pascal != name:
        candidates.append(pascal)
    for c in candidates:
        if hasattr(mod, c):
            cls = getattr(mod, c)
            if isinstance(cls, type) and issubclass(cls, dspy.Module):
                instance = cls()
                _module_cache[name] = instance
                return instance
    # Final fallback: scan all module attributes for a DSPy Module subclass
    for attr_name in dir(mod):
        cls = getattr(mod, attr_name, None)
        if (
            isinstance(cls, type)
            and issubclass(cls, dspy.Module)
            and cls is not dspy.Module
        ):
            instance = cls()
            _module_cache[name] = instance
            return instance
    raise ImportError(
        f"Module '{name}' has no DSPy module class (tried: {', '.join(candidates)})"
    )
