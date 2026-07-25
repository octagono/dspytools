"""Output directory management for compiled programs.

SSOT: Uses dspytools.core._io.read_json/write_json for all file operations.
No other module serializes run artifacts — this module is the single gateway.
"""

import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from dspytools.config.settings import compiled_dir
from dspytools.core._io import read_json, write_json


def create_run_dir(
    optimizer: str,
    label: str | None = None,
    metadata: dict | None = None,
) -> tuple[str, Path]:
    """Create a new run directory under compiled/.

    Returns (run_id, run_path).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_part = f"_{label}" if label else ""
    run_id = f"{timestamp}_{optimizer}{label_part}"
    run_path = compiled_dir() / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": run_id,
        "optimizer": optimizer,
        "created": datetime.now().isoformat(),
        **(metadata or {}),
    }
    write_json(run_path / "metadata.json", meta)

    return run_id, run_path


def save_program(
    run_path: Path,
    program,
    signature_data: dict,
    metrics: dict | None = None,
    module_type: str = "predict",
) -> None:
    """Save compiled program + signature + metrics + module_type to run directory.

    Fail-fast: if save fails, program can't be loaded.
    """
    program.save(str(run_path / "program.json"))

    sig_data = dict(signature_data)
    sig_data["module_type"] = module_type
    write_json(run_path / "signature.json", sig_data)

    meta_path = run_path / "metadata.json"
    if meta_path.exists():
        meta = read_json(meta_path)
        meta["module_type"] = module_type
        write_json(meta_path, meta)

    if metrics:
        write_json(run_path / "metrics.json", metrics)


def clean_old_runs(max_age_days: int = 30) -> int:
    """Remove run directories older than max_age_days. Returns count."""
    now = datetime.now()
    count = 0
    for entry in compiled_dir().iterdir():
        if not entry.is_dir():
            continue
        meta_path = entry / "metadata.json"
        if not meta_path.exists():
            continue
        meta = read_json(meta_path)
        created = datetime.fromisoformat(meta["created"])
        if (now - created).days > max_age_days:
            shutil.rmtree(entry)
            count += 1
    return count


@contextmanager
def compile_session(
    optimizer: str,
    label: str | None = None,
    metadata: dict | None = None,
) -> Iterator[tuple[str, Path]]:
    """Context manager for the compile run lifecycle.

    Creates a run directory on entry, cleans up on failure.
    Caller must call save_program() inside the block.
    """
    run_id, run_path = create_run_dir(optimizer, label, metadata or {})
    yield run_id, run_path


def finalize_run(
    run_id: str,
    run_path: Path,
    program: Any,
    signature_data: dict,
    optimizer: str,
    module_name: str,
    score: float,
    label: str | None = None,
    module_type: str = "predict",
) -> None:
    """Save program and register run in one call.

    Consolidates the save_program + register_run_with_graph pattern
    duplicated across 18+ command files.
    """
    save_program(run_path, program, signature_data, {"score": score}, module_type)

    from dspytools.core.registry import register_run_with_graph

    register_run_with_graph(
        run_id,
        {
            "optimizer": optimizer,
            "module": module_name,
            "score": score,
            "label": label or "",
        },
    )
