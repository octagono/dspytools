"""Auto-compile engine — builds and optimizes the self-help module."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

from dspytools.config.settings import DEFAULT_SEED, help_cache_path, help_meta_path
from dspytools.core._io import write_json
from dspytools.core.setup import LMRegistry
from dspytools.help.context import build_trainset_from_cli
from dspytools.help.module import HelpModule


class AutoCompiler:
    """Compiles and caches the self-help DSPy module.

    On first --help: auto-compile with LabeledFewShot.
    On self-optimize: re-compile with GEPA + deepseek teacher.
    """

    CACHE_PATH = help_cache_path()
    META_PATH = help_meta_path()

    @classmethod
    def compile_if_needed(cls, cli: Any) -> dspy.Module | None:
        """Load from cache if exists, else return None (no auto-compile)."""
        if cls.CACHE_PATH.exists():
            module = HelpModule()
            module.load(str(cls.CACHE_PATH))
            return module
        return None

    @classmethod
    def force_compile(cls, cli: Any, use_teacher: bool = False) -> dspy.Module:
        """Force re-compile the help module."""
        return cls._compile(cli, force=True, use_teacher=use_teacher)

    @classmethod
    def _compile(
        cls, cli: Any, force: bool = False, use_teacher: bool = False
    ) -> dspy.Module:
        """Internal: build trainset, run optimizer, save, return compiled."""
        trainset = build_trainset_from_cli(cli)

        if not trainset:
            return HelpModule()

        student = HelpModule()
        student_lm = LMRegistry.get_or_default()
        dspy.configure(lm=student_lm)

        teacher = LMRegistry.get_teacher() if use_teacher else None

        if teacher:
            # GEPA with deepseek teacher for rich reflection
            def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                return 1.0 if getattr(pred, "answer", "") else 0.0

            # Split valset to suppress GEPA "no valset" warning
            valset_size = max(1, len(trainset) // 5)
            random.seed(DEFAULT_SEED)
            shuffled = list(trainset)
            random.shuffle(shuffled)
            gfl_trainset = shuffled[valset_size:]
            gfl_valset = shuffled[:valset_size]

            optimizer = dspy.GEPA(
                metric=metric,
                reflection_lm=teacher,
                num_threads=1,
                max_metric_calls=50,
            )
            compiled = optimizer.compile(
                student=student, trainset=gfl_trainset, valset=gfl_valset
            )
        else:
            # LabeledFewShot for quick compilation
            optimizer = dspy.LabeledFewShot(k=min(4, len(trainset)))
            compiled = optimizer.compile(student=student, trainset=trainset)

        # Save
        cls.CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(cls.CACHE_PATH))

        # Save metadata
        write_json(
            cls.META_PATH,
            {
                "compiled_at": time.time(),
                "teacher": bool(teacher),
                "trainset_size": len(trainset),
            },
        )

        return compiled

    @classmethod
    def is_compiled(cls) -> bool:
        return cls.CACHE_PATH.exists()

    @classmethod
    def clear(cls) -> None:
        cls.CACHE_PATH.unlink(missing_ok=True)
        cls.META_PATH.unlink(missing_ok=True)
