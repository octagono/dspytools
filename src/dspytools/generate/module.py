"""DSPy signatures and module for llms.txt generation.

Signatures:
  AnalyzeRepository       → purpose, concepts, architecture
  AnalyzeCodeStructure    → directories, entry points, dev info
  GenerateLLMsTxt         → final llms.txt content

Module: RepositoryAnalyzer — multi-stage ChainOfThought pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy
else:
    from dspytools.core._dspy import dspy

import hashlib
import subprocess
import threading

from dspytools.core.logging_config import get_logger
from dspytools.generate.cache import get_analysis_cache

_log = get_logger(__name__)


# ── Sandbox Pool ────────────────────────────────────────────────────────────


class SandboxPool:
    """Persistent pool of sandbox workers for sandboxed code execution.

    Instead of cold-starting a process for each sandboxed execution,
    keep 2-4 workers warm and reuse them across evaluations.

    Falls back to direct subprocess if pool is exhausted.
    """

    def __init__(
        self,
        pool_size: int = 2,
        command: str = "python3",
        timeout: float = 30.0,
        max_reuse: int = 50,
        max_output_size: int = 1_000_000,  # 1M chars — guard against runaway output
    ):
        self.pool_size = pool_size
        self.command = command
        self.timeout = timeout
        self.max_reuse = max_reuse
        self.max_output_size = max_output_size
        self._workers: list[subprocess.Popen] = []
        self._lock = threading.Lock()
        self._in_use: set[int] = set()
        self._reuse_count: dict[int, int] = {}  # idx -> times reused
        self._total_recycled: int = 0  # cumulative recycle events
        self._warm_up()

    def _warm_up(self):
        """Pre-spawn workers to eliminate cold starts."""
        for _i in range(self.pool_size):
            proc = subprocess.Popen(
                [
                    self.command,
                    "-c",
                    (
                        "import sys\n"
                        "while True:\n"
                        "    try:\n"
                        "        exec(sys.stdin.readline())\n"
                        "    except Exception as e:\n"
                        "        print(e)\n"
                        "    finally:\n"
                        "        print('__SANDBOX_DONE__', flush=True)\n"
                    ),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self._workers.append(proc)

    def _acquire(self) -> tuple[int | None, subprocess.Popen | None]:
        """Acquire a free worker from the pool. Recycles workers that exceed max_reuse."""
        with self._lock:
            for idx, proc in enumerate(self._workers):
                if idx not in self._in_use and proc.poll() is None:
                    # Check reuse limit
                    if self._reuse_count.get(idx, 0) >= self.max_reuse:
                        # Recycle this worker
                        proc.terminate()
                        proc.wait(timeout=2)
                        # Spawn replacement
                        try:
                            new_proc = subprocess.Popen(  # noqa: S603
                                [
                                    self.command,
                                    "-c",
                                    (
                                        "import sys\n"
                                        "while True:\n"
                                        "    try:\n"
                                        "        exec(sys.stdin.readline())\n"
                                        "    except Exception as e:\n"
                                        "        print(e)\n"
                                        "    finally:\n"
                                        "        print('__SANDBOX_DONE__', flush=True)\n"
                                    ),
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                            )
                        except OSError as e:
                            _log.warning("sandbox_spawn_failed", error=str(e))
                            continue
                        self._workers[idx] = new_proc
                        self._reuse_count[idx] = 0
                        self._total_recycled += 1
                        self._in_use.add(idx)
                        return idx, new_proc

                    self._in_use.add(idx)
                    self._reuse_count[idx] = self._reuse_count.get(idx, 0) + 1
                    return idx, proc
        return None, None

    def _release(self, idx: int):
        """Release a worker back to the pool."""
        with self._lock:
            self._in_use.discard(idx)

    def execute(self, code: str) -> dict:
        """Execute code on a warm worker.

        Returns {success, output, error, worker_reused}.

        Falls back to one-shot subprocess if pool exhausted.
        """
        idx, proc = self._acquire()

        if proc is None or idx is None:
            # Pool exhausted — fall back to one-shot subprocess
            return self._execute_fallback(code)

        assert proc.stdin is not None
        assert proc.stdout is not None

        try:
            # Wrap multi-line code in a single exec() call
            # (worker reads one line per execute() cycle)
            safe_code = f"exec({code!r})"
            proc.stdin.write(safe_code + "\n")
            proc.stdin.flush()

            # Read output until delimiter or size limit
            output_lines = []
            total_chars = 0
            while True:
                line = proc.stdout.readline()
                if line.strip() == "__SANDBOX_DONE__":
                    break
                total_chars += len(line)
                if total_chars > self.max_output_size:
                    # Kill the worker — it's runaway
                    proc.kill()
                    self._release(idx)
                    return {
                        "success": False,
                        "output": "\n".join(output_lines),
                        "error": f"Output exceeded {self.max_output_size} chars — worker killed",
                        "worker_reused": False,
                    }
                output_lines.append(line.rstrip())

            self._release(idx)
            return {
                "success": True,
                "output": "\n".join(output_lines),
                "error": None,
                "worker_reused": True,
            }
        except (OSError, ValueError, TimeoutError) as e:
            _log.warning("sandbox_execution_failed", error=str(e))
            self._release(idx)
            # Worker may be dead — fall back
            return self._execute_fallback(code)

    def _execute_fallback(self, code: str) -> dict:
        """One-shot subprocess execution (fallback)."""
        try:
            result = subprocess.run(
                [self.command, "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            # Guard against runaway output
            if len(result.stdout) > self.max_output_size:
                return {
                    "success": False,
                    "output": result.stdout[: self.max_output_size],
                    "error": f"Output exceeded {self.max_output_size} chars — truncated",
                    "worker_reused": False,
                }
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "worker_reused": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "Timeout",
                "worker_reused": False,
            }
        except (OSError, ValueError, TimeoutError) as e:
            _log.error("sandbox_fallback_failed", error=str(e))
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "worker_reused": False,
            }

    def shutdown(self):
        """Kill all workers."""
        with self._lock:
            for proc in self._workers:
                proc.terminate()
            self._workers.clear()
            self._in_use.clear()

    @property
    def stats(self) -> dict:
        """Pool statistics."""
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "workers_alive": sum(1 for p in self._workers if p.poll() is None),
                "workers_in_use": len(self._in_use),
                "workers_available": sum(
                    1
                    for i, p in enumerate(self._workers)
                    if i not in self._in_use and p.poll() is None
                ),
                "reuse_counts": dict(self._reuse_count),
                "max_reuse": self.max_reuse,
                "recycled": sum(
                    1 for c in self._reuse_count.values() if c >= self.max_reuse
                ),
                "total_recycled": self._total_recycled,
            }

    def __del__(self):
        self.shutdown()


# Module-level singleton
_sandbox_pool: SandboxPool | None = None


def get_sandbox_pool(pool_size: int = 2) -> SandboxPool:
    """Get or create the global sandbox pool."""
    global _sandbox_pool
    if _sandbox_pool is None:
        _sandbox_pool = SandboxPool(pool_size=pool_size)
    return _sandbox_pool


# ── Signatures ──────────────────────────────────────────────────────────────


class AnalyzeRepository(dspy.Signature):
    """Analyze a repository structure and identify key components."""

    repo_url: str = dspy.InputField(desc="GitHub repository URL")
    file_tree: str = dspy.InputField(desc="Repository file structure")
    readme_content: str = dspy.InputField(desc="README.md content")

    project_purpose: str = dspy.OutputField(desc="Main purpose and goals")
    key_concepts: list[str] = dspy.OutputField(
        desc="Important concepts and terminology"
    )
    architecture_overview: str = dspy.OutputField(desc="High-level architecture")


# Signature manipulation variants
AnalyzeRepositoryV2 = AnalyzeRepository.with_instructions(
    "Focus deeply on the architectural patterns, design decisions, "
    "and how components interact. Be thorough but concise."
)

AnalyzeRepositoryWithConfidence = AnalyzeRepository.append(
    "confidence",
    dspy.OutputField(desc="Confidence level 0-1 for analysis"),
    type_=float,
)

AnalyzeRepositoryWithContext = AnalyzeRepository.prepend(
    "prior_context", dspy.InputField(desc="Previous analysis context"), type_=str
)

AnalyzeRepositoryTyped = AnalyzeRepository.with_updated_fields(
    "key_concepts",
    desc="List of key concepts, each prefixed with domain category",
).with_updated_fields(
    "project_purpose",
    type_=str,
    desc="One-sentence elevator pitch of the project",
)


class AnalyzeCodeStructure(dspy.Signature):
    """Analyze code structure to identify important directories and files."""

    file_tree: str = dspy.InputField(desc="Repository file structure")
    package_files: str = dspy.InputField(desc="Key package and configuration files")

    important_directories: list[str] = dspy.OutputField(
        desc="Key directories and purposes"
    )
    entry_points: list[str] = dspy.OutputField(
        desc="Main entry points and important files"
    )
    development_info: str = dspy.OutputField(desc="Setup and workflow information")


AnalyzeCodeStructureWithSize = AnalyzeCodeStructure.insert(
    0,
    "repo_size_hint",
    dspy.InputField(desc="Lines of code or repo size estimate"),
    type_=str,
)

AnalyzeCodeStructureMinimal = AnalyzeCodeStructure.delete("development_info")


class GenerateLLMsTxt(dspy.Signature):
    """Generate a comprehensive llms.txt from analyzed information."""

    project_purpose: str = dspy.InputField()
    key_concepts: list[str] = dspy.InputField()
    architecture_overview: str = dspy.InputField()
    important_directories: list[str] = dspy.InputField()
    entry_points: list[str] = dspy.InputField()
    development_info: str = dspy.InputField()
    usage_examples: str = dspy.InputField(desc="Common usage patterns")

    llms_txt_content: str = dspy.OutputField(desc="Complete llms.txt content")


# ── Module ───────────────────────────────────────────────────────────────────


class RepositoryAnalyzer(dspy.Module):
    """Multi-stage llms.txt generator.

    Pipeline:
      1. AnalyzeRepository (ChainOfThought) → purpose, concepts, architecture
      2. AnalyzeCodeStructure (ChainOfThought) → directories, entries, dev info
      3. ChainOfThought → usage_examples from file_tree
      4. ChainOfThought → structured summary of concepts
      5. GenerateLLMsTxt (ChainOfThought) → final llms.txt
    """

    def __init__(self):
        super().__init__()
        self.analyze_repo = dspy.ChainOfThought(AnalyzeRepositoryV2)
        self.analyze_structure = dspy.ChainOfThought(AnalyzeCodeStructure)

        # Stages 3+4 run on the student LM (Qwen local), which cannot execute
        # sandboxed code. DSPy's CodeAct/ProgramOfThought were deprecated in
        # 3.3.1 (removal in 3.5) and always fell back to these ChainOfThought
        # predictors on the local model anyway — so ChainOfThought is the
        # primary, non-deprecated structure.
        self.code_analyzer = dspy.ChainOfThought(
            "file_tree, package_files -> usage_examples: str, edge_cases: str"
        )
        self.pot_summarizer = dspy.ChainOfThought(
            "bullet_points -> structured_summary: str"
        )

        self.generate_llms_txt = dspy.ChainOfThought(GenerateLLMsTxt)

    def forward(
        self,
        repo_url: str,
        file_tree: str,
        readme_content: str,
        package_files: str,
    ) -> dspy.Prediction:
        # ── AST-based caching ────────────────────────────────────────────

        cache = get_analysis_cache()
        key_material = f"{repo_url}|{file_tree}|{readme_content}|{package_files}"
        cache_key = hashlib.sha256(key_material.encode()).hexdigest()[:24]

        cached = cache.get(cache_key)
        if cached:
            return dspy.Prediction(
                llms_txt_content=cached.get("llms_txt_content", ""),
                analysis=cached.get("analysis"),
                structure=cached.get("structure"),
                code_analysis=cached.get("code_analysis"),
                pot_summary=cached.get("pot_summary"),
            )
        # ── End caching ──────────────────────────────────────────────────

        # Stage 1: Analyze repository purpose & concepts
        repo_analysis = self.analyze_repo(
            repo_url=repo_url,
            file_tree=file_tree,
            readme_content=readme_content,
        )

        # Stage 2: Analyze code structure
        structure_analysis = self.analyze_structure(
            file_tree=file_tree,
            package_files=package_files,
        )

        # Stage 3: Generate usage examples from the repository file tree
        code_result = self.code_analyzer(
            file_tree=file_tree,
            package_files=package_files,
        )
        usage_examples = code_result.usage_examples

        # Stage 4: Summarize the analyzed concepts
        concept_bullets = "\n".join(f"- {c}" for c in repo_analysis.key_concepts)
        pot_summary = self.pot_summarizer(bullet_points=concept_bullets)

        # Stage 5: Generate final llms.txt
        llms_txt = self.generate_llms_txt(
            project_purpose=repo_analysis.project_purpose,
            key_concepts=repo_analysis.key_concepts,
            architecture_overview=repo_analysis.architecture_overview,
            important_directories=structure_analysis.important_directories,
            entry_points=structure_analysis.entry_points,
            development_info=structure_analysis.development_info,
            usage_examples=usage_examples,
        )

        # Cache the result for future runs
        cache.set(
            cache_key,
            {
                "llms_txt_content": llms_txt.llms_txt_content,
                "analysis": str(repo_analysis)
                if hasattr(repo_analysis, "__dict__")
                else repo_analysis,
                "structure": str(structure_analysis)
                if hasattr(structure_analysis, "__dict__")
                else structure_analysis,
            },
        )

        return dspy.Prediction(
            llms_txt_content=llms_txt.llms_txt_content,
            analysis=repo_analysis,
            structure=structure_analysis,
            code_analysis=code_result,
            pot_summary=pot_summary,
        )

    def inspect_predictors(self) -> None:
        """Show all named sub-predictors in this module."""
        for name, predictor in self.named_predictors():
            _log.info("predictor_loaded", name=name, type=type(predictor).__name__)

    def show_lm(self) -> None:
        """Show the current LM bound to this module."""
        current_lm = self.get_lm()
        _log.info("module_lm", lm=str(current_lm))
