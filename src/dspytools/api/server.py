"""FastAPI server for dspytools hot-swap inference API.

Optimization 1: LRU program cache via HotSwapManager.
Optimization 2: Lazy-loading — index loaded on first access, programs on swap/infer.
Optimization 10: Batch config endpoint for atomic model updates.
"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dspytools.config.settings import load_config, save_user_config
from dspytools.core.hotswap import HotSwapManager
from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)

app = FastAPI(title="DSPyTools Hot-Swap API", version="0.1.0")
_hotswap: HotSwapManager | None = None


def get_hotswap() -> HotSwapManager:
    global _hotswap
    if _hotswap is None:
        # Optimization 2: Don't call load_all() — lazy index + on-demand loading
        _hotswap = HotSwapManager()
    return _hotswap


class InferRequest(BaseModel):
    inputs: dict[str, Any]


class SwapResponse(BaseModel):
    status: str
    active: str | None = None
    previous: str | None = None
    message: str | None = None


class ProgramInfo(BaseModel):
    id: str
    active: bool
    optimizer: str = ""
    created: str = ""
    score: float | None = None


class ModelConfig(BaseModel):
    model: str
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None


class ModelsConfigRequest(BaseModel):
    """Optimization 10: Atomic batch model config."""

    student: ModelConfig | None = None
    teacher: ModelConfig | None = None


# ── Program Management ─────────────────────────────────────────────


@app.get("/programs", response_model=list[ProgramInfo])
async def list_programs():
    mgr = get_hotswap()
    mgr._ensure_index()  # Optimization 2: ensure metadata loaded for list
    return [ProgramInfo(**p) for p in mgr.list()]


@app.get("/programs/{program_id}", response_model=dict)
async def get_program(program_id: str):
    mgr = get_hotswap()
    meta = mgr.get_metadata(program_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Program '{program_id}' not found")
    return meta


@app.post("/swap/{program_id}", response_model=SwapResponse)
async def swap_program(program_id: str, warm: bool = False):
    mgr = get_hotswap()
    try:
        if warm:
            prev = mgr.warm_swap(program_id)
            return SwapResponse(
                status="ok",
                active=program_id,
                previous=prev,
                message="Warm swap — verified with test inference",
            )
        prev = mgr.swap(program_id)
        return SwapResponse(status="ok", active=program_id, previous=prev)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/infer")
async def infer(request: InferRequest):
    mgr = get_hotswap()
    try:
        result = mgr.infer(**request.inputs)
        return {"status": "ok", "result": result}
    except (RuntimeError, ValueError, KeyError, TypeError, OSError) as e:
        _log.error("infer_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/active")
async def get_active():
    mgr = get_hotswap()
    return {"active": mgr.active_id}


# ── Model Configuration ──────────────────────────────────────────


@app.get("/config/models")
async def get_model_config():
    """Get current student and teacher model configuration."""
    cfg = load_config()
    return {
        "student": cfg.get("lm", {}).get("student"),
        "teacher": cfg.get("lm", {}).get("teacher"),
        "default": cfg.get("lm", {}).get("default"),
    }


@app.post("/config/models/student")
async def set_student_model(config: ModelConfig):
    """Configure the student (inference) model."""
    cfg = load_config()
    cfg.setdefault("lm", {})
    cfg["lm"]["student"] = config.model_dump(exclude_none=True)
    save_user_config(cfg)
    return {"status": "ok", "student": config.model}


@app.post("/config/models/teacher")
async def set_teacher_model(config: ModelConfig):
    """Configure the teacher (optimization/reflection) model."""
    cfg = load_config()
    cfg.setdefault("lm", {})
    cfg["lm"]["teacher"] = config.model_dump(exclude_none=True)
    save_user_config(cfg)
    return {"status": "ok", "teacher": config.model}


@app.put("/config/models")
async def set_models_batch(config: ModelsConfigRequest):
    """Optimization 10: Atomic batch update of student + teacher models."""
    cfg = load_config()
    cfg.setdefault("lm", {})
    if config.student:
        cfg["lm"]["student"] = config.student.model_dump(exclude_none=True)
    if config.teacher:
        cfg["lm"]["teacher"] = config.teacher.model_dump(exclude_none=True)
    save_user_config(cfg)
    return {
        "status": "ok",
        "student": config.student.model if config.student else None,
        "teacher": config.teacher.model if config.teacher else None,
    }


# ── Health ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "active_program": get_hotswap().active_id}


def run_api(host: str = "0.0.0.0", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port)
