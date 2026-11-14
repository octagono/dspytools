"""FastAPI server for dspytools hot-swap inference API.

Optimization 1: LRU program cache via HotSwapManager.
Optimization 2: Lazy-loading — index loaded on first access, programs on swap/infer.
Optimization 10: Batch config endpoint for atomic model updates.

DSPy configuration contract:
- ``lifespan`` configures the process LM once at startup via ``setup_dspy()``.
  ``get_hotswap()`` repeats that guard lazily so any non-ASGI entry point (a
  direct ``TestClient`` call, a forked worker) cannot serve requests with
  ``dspy.settings.lm is None`` — which made every ``/infer`` a 500.
- Model config writes clear ``LMRegistry`` and re-apply configuration in the
  running process, so ``/config/models/*`` takes effect without a restart.
- Blocking DSPy work runs on a worker thread via ``dspy.asyncify`` (anyio
  thread pool, limited by the ``async_max_workers`` setting), never on the
  event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from starlette.responses import JSONResponse

from dspytools.config.settings import load_config, save_user_config
from dspytools.core._dspy import dspy
from dspytools.core.hotswap import HotSwapManager
from dspytools.core.logging_config import get_logger
from dspytools.core.setup import LMRegistry, setup_dspy

_log = get_logger(__name__)


class LMConfigurationError(RuntimeError):
    """The API process has no usable DSPy LM configured."""


_lm_configured = False
# The LM this process serves with. Held explicitly because a running process
# cannot call ``dspy.configure`` from a second thread/async task (DSPy's
# single-owner rule) — requests pin it with ``dspy.context`` instead.
_serving_lm = None


def _ensure_lm_configured(force: bool = False) -> None:
    """Configure the process LM via ``setup_dspy()``; fail loudly if unusable.

    Idempotent. ``force=True`` drops the cached LM instances and rebuilds from
    the persisted model config — the path used after ``/config/models/*``.
    """
    global _lm_configured, _serving_lm
    if force:
        # Drop cached dspy.LM instances so the newly persisted config is read.
        LMRegistry.clear()
    elif _lm_configured and _serving_lm is not None:
        return

    if force:
        lm = LMRegistry.get_or_default()
        try:
            dspy.configure(lm=lm)
        except RuntimeError as e:
            # DSPy allows configure() from one thread and one async task only.
            # A config change arriving inside a request task cannot publish
            # globally; serving already pins the LM with dspy.context().
            _log.info("api_lm_configure_deferred", error=str(e))
    else:
        lm = dspy.settings.lm
        if lm is None:
            lm = setup_dspy()

    if lm is None:
        raise LMConfigurationError(
            "DSPy has no LM configured — inference would fail with "
            "'No LM is loaded'. Configure one with "
            "`dspytools configure lm set <model>` (or set DSPYTOOLS_LLM_MODEL) "
            "and restart the API server."
        )

    _serving_lm = lm
    _lm_configured = True


def _current_lm():
    """The LM serving this process: explicit serving LM, else DSPy's setting."""
    return _serving_lm if _serving_lm is not None else dspy.settings.lm


def _active_lm_name() -> str | None:
    """Model name of the currently serving LM, or None."""
    lm = _current_lm()
    return getattr(lm, "model", None) if lm is not None else None


def _reload_lm_configuration() -> str | None:
    """Re-apply persisted model config inside the running process."""
    _ensure_lm_configured(force=True)
    model = _active_lm_name()
    _log.info("api_lm_reconfigured", model=model)
    return model


def _with_serving_lm(fn, *args, **kwargs):
    """Run ``fn`` with the serving LM pinned for this thread.

    ``dspy.context()`` is allowed from any thread or async task, unlike
    ``dspy.configure()``. Pinning per call is what makes a ``/config/models``
    change take effect without restarting the process.
    """
    with dspy.context(lm=_current_lm()):
        return fn(*args, **kwargs)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup hook: configure the LM before serving any request."""
    _ensure_lm_configured()
    _log.info("api_startup", lm=_active_lm_name())
    yield


app = FastAPI(title="DSPyTools Hot-Swap API", version="0.1.0", lifespan=lifespan)
_hotswap: HotSwapManager | None = None


@app.exception_handler(LMConfigurationError)
async def _lm_configuration_error_handler(
    _request: Request, exc: LMConfigurationError
) -> JSONResponse:
    """Report a missing LM as 503 (service not ready), not an opaque 500."""
    _log.error("api_lm_unavailable", error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def get_hotswap() -> HotSwapManager:
    global _hotswap
    # Configure the process LM before any program can be served. Idempotent —
    # every request path goes through here, so a manager created elsewhere (or
    # by an earlier request) can never leave ``dspy.settings.lm`` unset.
    _ensure_lm_configured()
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
            # warm_swap() runs a verification inference — blocking, so off-loop.
            prev = await dspy.asyncify(_with_serving_lm)(mgr.warm_swap, program_id)
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
        # dspy.asyncify runs the LM-bound, synchronous infer() on an anyio worker
        # thread (bounded by the `async_max_workers` setting, default 8) and
        # re-applies the caller thread's DSPy overrides inside it.
        result = await dspy.asyncify(_with_serving_lm)(mgr.infer, **request.inputs)
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
        "active_lm": _active_lm_name(),
    }


@app.post("/config/models/student")
async def set_student_model(config: ModelConfig):
    """Configure the student (inference) model and apply it in-process."""
    cfg = load_config()
    cfg.setdefault("lm", {})
    cfg["lm"]["student"] = config.model_dump(exclude_none=True)
    save_user_config(cfg)
    active_lm = _reload_lm_configuration()
    return {"status": "ok", "student": config.model, "active_lm": active_lm}


@app.post("/config/models/teacher")
async def set_teacher_model(config: ModelConfig):
    """Configure the teacher (optimization/reflection) model."""
    cfg = load_config()
    cfg.setdefault("lm", {})
    cfg["lm"]["teacher"] = config.model_dump(exclude_none=True)
    save_user_config(cfg)
    # The teacher LM is never the inference LM, but a config change must clear
    # the registry cache so optimization paths pick up the new teacher; the
    # student configuration is unchanged, so the re-applied LM stays the same.
    active_lm = _reload_lm_configuration()
    return {"status": "ok", "teacher": config.model, "active_lm": active_lm}


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
    active_lm = _reload_lm_configuration()
    return {
        "status": "ok",
        "student": config.student.model if config.student else None,
        "teacher": config.teacher.model if config.teacher else None,
        "active_lm": active_lm,
    }


# ── Health ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Liveness/readiness probe — no LM calls, no program loading."""
    lm = _current_lm()
    return {
        "status": "ok" if lm is not None else "degraded",
        "active_program": _hotswap.active_id if _hotswap is not None else None,
        "lm_configured": lm is not None,
        "lm": getattr(lm, "model", None) if lm is not None else None,
    }


def run_api(host: str = "0.0.0.0", port: int = 8080) -> None:
    uvicorn.run(app, host=host, port=port)
