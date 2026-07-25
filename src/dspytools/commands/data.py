"""dspytools data — Load, inspect, and manage datasets."""

from __future__ import annotations

import csv as _csv
import json
from os.path import getsize
from pathlib import Path

from dspytools.cli.output import console, error, header, info, ok, panel, table, warn
from dspytools.cli.rich_config import LLMCommand, LLMGroup, click
from dspytools.core._io import read_json, write_json

HF_RAW_BASE = "https://huggingface.co/datasets/{source}/resolve/main"
HF_API_BASE = "https://huggingface.co/api/datasets/{source}"


def _hf_list_files(source: str, split: str, limit: int | None) -> list[str]:
    """List raw JSONL files for a HuggingFace dataset matching the given split."""
    import requests as _requests

    resp = _requests.get(HF_API_BASE.format(source=source), timeout=30)
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])

    # Split-guided prefixes: which paths to include for each split
    split_prefixes = {
        "train": ("train", "data", "pi-traces", "pi_traces", "examples"),
        "test": ("test",),
        "validation": ("validation", "valid", "dev"),
    }
    prefixes = split_prefixes.get(split, (split,))

    candidates = [
        s["rfilename"]
        for s in siblings
        if s["rfilename"].endswith(".jsonl")
        and any(p in s["rfilename"] for p in prefixes)
        and not s["rfilename"].startswith("claude/")
    ]

    if not candidates:
        # Fall back to all JSONL files if split-guess landed on nothing
        candidates = [
            s["rfilename"]
            for s in siblings
            if s["rfilename"].endswith(".jsonl")
            and not s["rfilename"].startswith("claude/")
        ]

    candidates = sorted(candidates)

    if limit and len(candidates) > limit:
        candidates = candidates[:limit]

    if not candidates:
        raise click.ClickException(
            f"No JSONL files found for '{source}' split '{split}'.\n"
            "  Verify the dataset name and split on huggingface.co/datasets/{source}"
        )

    return candidates


def _hf_download_parse(source: str, files: list[str], fields: list[str] | None) -> list:
    """Download and parse multi-line JSONL session files into DSPy Examples."""
    import requests as _requests

    from dspytools.core._dspy import dspy

    base = HF_RAW_BASE.format(source=source)
    examples: list = []

    for fname in files:
        try:
            resp = _requests.get(f"{base}/{fname}", timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
        except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
            warn(f"Skipping {fname}: {exc}")
            continue

        # Assemble a single session from multi-line JSONL
        session: dict = {}
        msg_list: list = []
        for line in lines:
            if not line.strip():
                continue
            entry = json.loads(line)
            match entry.get("type"):
                case "session":
                    session["session_id"] = entry.get("id")
                    session["timestamp"] = entry.get("timestamp")
                case "model_change":
                    session["model"] = entry.get("modelId")
                case "thinking_level_change":
                    session["thinking_level"] = entry.get("thinkingLevel")
                case "message":
                    msg = entry.get("message", {})
                    role = msg.get("role")
                    content = msg.get("content", [])
                    msg_list.append(msg)
                    if role == "user":
                        texts = [
                            c.get("text", "")
                            for c in content
                            if c.get("type") == "text"
                        ]
                        if texts and "prompt" not in session:
                            session["prompt"] = texts[0]
                    elif role == "assistant":
                        thinking = " ".join(
                            c.get("thinking", "")
                            for c in content
                            if c.get("type") == "thinking"
                        )
                        if thinking and "answer" not in session:
                            session["answer"] = thinking

        if msg_list:
            session["messages"] = msg_list
        if not session:
            continue

        # Alias prompt -> question for DSPy question->answer signatures
        if "prompt" in session and "question" not in session:
            session["question"] = session["prompt"]

        # Build the DSPy Example
        if fields:
            filtered = {f: session[f] for f in fields if f in session}
            if filtered:
                examples.append(dspy.Example(**filtered))
        else:
            examples.append(dspy.Example(**session))

    if not examples:
        raise click.ClickException(
            f"Failed to parse any examples from {len(files)} files.\n"
            "  The dataset files may have an unexpected structure."
        )

    return examples


@click.group(name="data", cls=LLMGroup)
def data_cmd():
    """Load, inspect, and manage datasets."""


@data_cmd.command(name="load", cls=LLMCommand)
@click.argument("source")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["huggingface", "json", "csv", "auto"]),
    default="auto",
    help="Data format",
)
@click.option("--split", default="train", help="Dataset split (for HuggingFace)")
@click.option("--name", help="HuggingFace config name")
@click.option("--fields", help="Comma-separated field names")
@click.option("--rename", help="Rename fields (format: old=new,old2=new2)")
@click.option("--input-keys", help="Comma-separated input field names")
@click.option("--limit", type=int, help="Limit number of examples")
@click.option("--output", "-o", help="Save to JSON file")
@click.option(
    "--raw",
    "force_raw",
    is_flag=True,
    default=False,
    help="Download raw JSONL files from HuggingFace directly (bypasses datasets library)",
)
def data_load(
    source: str,
    fmt: str,
    split: str,
    name: str | None,
    fields: str | None,
    rename: str | None,
    input_keys: str | None,
    limit: int | None,
    output: str | None,
    force_raw: bool = False,
):
    """Load a dataset from huggingface, JSON, or CSV.

    SOURCE: HuggingFace dataset name (e.g., 'PolyAI/banking77') or file path

    Use --raw for datasets with complex file structures (multi-line JSONL
    sessions, tool-call traces) that the HuggingFace datasets library cannot
    parse into a flat table.
    """
    from dspytools.core._dspy import dspy

    source_path = Path(source)
    file_exists = source_path.exists()
    dataset: list = []

    # Parse rename map (old -> new)
    rename_map: dict[str, str] = {}
    if rename:
        for pair in rename.split(","):
            pair = pair.strip()
            if "=" not in pair:
                raise click.ClickException(
                    f"Invalid rename '{pair}'. Use format: old=new"
                )
            old_k, new_k = pair.split("=", 1)
            rename_map[old_k.strip()] = new_k.strip()

    # Classify source type
    is_local_file = file_exists or source.endswith(".json") or source.endswith(".csv")

    if fmt == "huggingface" or (fmt == "auto" and "/" in source and not is_local_file):
        field_list = fields.split(",") if fields else None
        input_list = input_keys.split(",") if input_keys else None

        if force_raw:
            raw_files = _hf_list_files(source, split, limit)
            dataset = _hf_download_parse(source, raw_files, field_list)
        else:
            from dspy.datasets import DataLoader

            loader = DataLoader()
            hf_fields = tuple(field_list) if field_list else None
            hf_inputs = tuple(input_list) if input_list else ()

            kwargs: dict = {"dataset_name": source, "fields": hf_fields, "split": split}
            if name:
                kwargs["name"] = name
            if hf_inputs:
                kwargs["input_keys"] = hf_inputs

            try:
                loaded = loader.from_huggingface(**kwargs)
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                error(f"HuggingFace load failed for '{source}'")
                raise click.ClickException(
                    f"Could not load dataset '{source}' from HuggingFace.\n"
                    f"  {exc}\n"
                    "  Verify the dataset exists at huggingface.co/datasets/{source}\n"
                    "  Or use a local .json / .csv file instead."
                )

            if isinstance(loaded, list):
                dataset = loaded
            elif hasattr(loaded, split):
                dataset = getattr(loaded, split)
            else:
                dataset = list(loaded)

        if input_list and dataset:
            dataset = [ex.with_inputs(*input_list) for ex in dataset]

        if limit and dataset:
            dataset = dataset[:limit]

        mode = "raw JSONL" if force_raw else "datasets"
        ok(f"Loaded {len(dataset)} examples from '{source}' [{split}] ({mode})")

    elif fmt in ("json", "csv") or (
        fmt == "auto"
        and file_exists
        and (source.endswith(".json") or source.endswith(".csv"))
    ):
        if source.endswith(".csv"):
            with open(source) as f:
                reader = _csv.DictReader(f)
                rows = list(reader)
        else:
            rows = read_json(Path(source))

        if isinstance(rows, list):
            dataset = (
                [dspy.Example(**r) for r in rows[:limit]]
                if limit
                else [dspy.Example(**r) for r in rows]
            )
            if input_keys:
                dataset = [
                    ex.with_inputs(*[k.strip() for k in input_keys.split(",")])
                    for ex in dataset
                ]
            ok(f"Loaded {len(dataset)} examples from '{source}'")

    elif fmt == "auto" and not is_local_file and "/" not in source:
        # Bare dataset name without org prefix (e.g. "squad", "trivia_qa")
        error(f"unknown dataset source: {source}")
        raise click.ClickException(
            f"Dataset source '{source}' not found.\n"
            "  HuggingFace datasets require the format: org/name (e.g., rajpurkar/squad)\n"
            "  Provide a full HuggingFace dataset name (e.g., PolyAI/banking77)\n"
            "  Or a path to a local .json/.csv file"
        )

    else:
        error(f"unknown dataset source: {source}")
        raise click.ClickException(
            f"Dataset source '{source}' not found.\n"
            "  Provide a HuggingFace dataset name (e.g., PolyAI/banking77)\n"
            "  Or a path to a local .json/.csv file"
        )

    # Apply field renaming
    if rename_map and dataset:
        renamed: list = []
        for ex in dataset:
            d = ex.toDict() if hasattr(ex, "toDict") else dict(ex)
            for old_k, new_k in rename_map.items():
                if old_k in d:
                    d[new_k] = d.pop(old_k)
            if hasattr(ex, "with_inputs"):
                renamed.append(dspy.Example(**d))
            else:
                renamed.append(type(ex)(**d))
        dataset = renamed
        ok(f"Renamed fields: {', '.join(f'{k}->{v}' for k, v in rename_map.items())}")

    # Show preview
    if dataset:
        first = dataset[0].toDict() if hasattr(dataset[0], "toDict") else dataset[0]
        panel(
            "Dataset Preview",
            f"[bold]Size:[/] {len(dataset)} examples\n"
            f"[bold]Fields:[/] {', '.join(first.keys()) if isinstance(first, dict) else 'N/A'}\n"
            f"[bold]First:[/] {str(first)[:200]}",
            border_style="cyan",
        )

    if output and dataset:
        out_data = [ex.toDict() if hasattr(ex, "toDict") else str(ex) for ex in dataset]
        write_json(Path(output), out_data)
        ok(f"Saved to {output}")


@data_cmd.command(name="preview", cls=LLMCommand)
@click.argument("path")
@click.option("--n", default=5, type=int, help="Number of examples to show")
def data_preview(path: str, n: int):
    """Preview a dataset JSON file."""
    p = Path(path)
    if not p.exists():
        error(f"file not found: {path}")
        raise click.Abort()
    data = read_json(p)
    if isinstance(data, list):
        console.print(f"[bold]{len(data)}[/] examples")
        for i, ex in enumerate(data[:n]):
            console.print(f"\n[cyan]Example {i + 1}:[/]")
            if isinstance(ex, dict):
                for k, v in ex.items():
                    val = str(v)[:100]
                    console.print(f"  [bold]{k}:[/] {val}")
            else:
                console.print(f"  {str(ex)[:200]}")
    else:
        error(f"Expected a list, got {type(data).__name__}")


@data_cmd.command(name="list", cls=LLMCommand)
def data_list():
    """List available datasets in the data/ directory."""

    data_dir = Path("data")
    if not data_dir.exists():
        info("No data/ directory found")
        return

    files = sorted(data_dir.glob("*.json")) + sorted(data_dir.glob("*.csv"))
    if not files:
        info("No datasets found")
        return

    rows = []
    for f in files:
        size_bytes = getsize(f)
        if size_bytes < 1024:
            size_str = f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f}KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f}MB"
        fmt = "json" if f.suffix == ".json" else "csv"
        rows.append([f.name, size_str, fmt])

    header("Available Datasets")
    table("Datasets", ["Name", "Size", "Format"], rows)
