"""Skill Manager — lifecycle: create, compile, optimize, integrate.

Skills follow the harness-so pattern:
  1. CREATE: Generate SKILL.md from task description
  2. COMPILE: Generate DSPy program, run optimizer, save compiled
  3. INTEGRATE: Load into agent pipeline, BM25-index
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from dspytools.config.settings import compiled_dir
from dspytools.config.settings import skills_dir as get_skills_dir
from dspytools.core._dspy import dspy
from dspytools.core._io import read_json, write_json
from dspytools.core.logging_config import get_logger
from dspytools.core.metrics import answer_exact_match
from dspytools.core.programs import is_current_dspy_version, saved_dspy_version
from dspytools.skills.loader import Skill, SkillLoader

_log = get_logger(__name__)


def _yaml_scalar(value: str) -> str:
    """Encode a frontmatter scalar as a JSON string (a valid YAML subset).

    Unquoted values broke every consumer that parses the frontmatter as YAML:
    a description containing ``": "`` (or ``" #"``) produced invalid YAML and
    GEPA's ``InstructionProposer`` raised ``ValueError`` before any reflection.
    JSON string encoding needs no extra dependency and round-trips exactly.
    """
    return json.dumps(value, ensure_ascii=False)


def _count_demos(program: Any) -> int:
    """Number of labeled examples stored on a compiled program's predictors."""
    try:
        predictors = list(program.named_predictors())
    except Exception:  # not a dspy.Module (e.g. a bare Predict)
        demos = getattr(program, "demos", None)
        return len(demos or [])
    return sum(
        len(getattr(predictor, "demos", None) or []) for _, predictor in predictors
    )


def _module_for_type(module_type: str, signature: str) -> Any:
    """Build the module shape recorded in ``metadata.module_type``.

    Mirrors ``core.hotswap._load_program_from_run``: a CoT/ReAct artifact loaded
    into a bare ``dspy.Predict`` raised because the saved state has more
    predictors than the fresh program.
    """
    kind = (module_type or "predict").lower()
    if kind in {"cot", "chain_of_thought"}:
        return dspy.ChainOfThought(signature)
    if kind == "react":
        return dspy.ReAct(signature, tools=[])
    if kind == "flex":
        return dspy.Flex(signature)
    return dspy.Predict(signature)


def _signature_string(skill_path: Path) -> str:
    sig_path = skill_path / "signature.json"
    if not sig_path.exists():
        return "question -> answer"
    sig_data = read_json(sig_path)
    inputs = ", ".join(sig_data.get("inputs", ["input"]))
    outputs = ", ".join(sig_data.get("outputs", ["output"]))
    return f"{inputs} -> {outputs}"


def _module_type(skill_path: Path) -> str:
    meta_path = skill_path / "metadata.json"
    if meta_path.exists():
        module_type = read_json(meta_path).get("module_type")
        if module_type:
            return str(module_type)
    sig_path = skill_path / "signature.json"
    if sig_path.exists():
        module_type = read_json(sig_path).get("module_type")
        if module_type:
            return str(module_type)
    return "predict"


def load_skill_program(skill_path: str | Path) -> Any | None:
    """Load a skill's saved program in the shape it was compiled in.

    Returns ``None`` when there is no program or when it was saved by a
    different DSPy (stale artifacts are treated as absent, matching
    ``SkillLoader``).
    """
    path = Path(skill_path)
    prog_path = path / "program.json"
    if not prog_path.exists():
        return None
    if not is_current_dspy_version(prog_path):
        _log.warning(
            "skill_program_dspy_version_mismatch",
            skill=path.name,
            saved_version=saved_dspy_version(prog_path),
            hint="recompile with `dspytools skills compile`",
        )
        return None

    program = _module_for_type(_module_type(path), _signature_string(path))

    saved = json.loads(prog_path.read_text())
    try:
        program.load(str(prog_path))
    except (KeyError, ValueError):
        # Predict format: program.json wraps state in {predictor: {...}} but
        # load_state() needs the flat {signature, demos, ...} mapping.
        if "predictor" in saved:
            program.load_state(saved["predictor"])
    return program


class SkillManager:
    """Manages the skill lifecycle: generate → compile → integrate."""

    def __init__(self, skills_dir: str | None = None):
        # Default to user-wide skills dir for SSOT consistency with SkillConsolidator
        self.dir = Path(skills_dir) if skills_dir else get_skills_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.loader = SkillLoader(str(self.dir))

    def create_skill(
        self, name: str, description: str, signature: str, body: str = ""
    ) -> Skill:
        """Create a new skill directory with SKILL.md."""
        skill_dir = self.dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        md_content = f"""---
name: {_yaml_scalar(name)}
description: {_yaml_scalar(description)}
signature: {_yaml_scalar(signature)}
---
{body or f"# {name}\n\n{description}"}
"""
        (skill_dir / "SKILL.md").write_text(md_content)

        return Skill(
            name=name,
            description=description,
            path=skill_dir,
            frontmatter={
                "name": name,
                "description": description,
                "signature": signature,
            },
            body=body or description,
        )

    def compile_skill(
        self,
        name: str,
        trainset: list | None = None,
        optimizer: str = "labeled_few_shot",
    ) -> dict:
        """Compile a skill's DSPy program and save to the skill directory."""
        skill = self.loader.get(name)
        if not skill:
            return {"error": f"Skill '{name}' not found"}

        skill_path = skill.path
        if not skill_path:
            return {"error": f"Skill '{name}' has no path"}

        sig = skill.frontmatter.get("signature", "question -> answer")
        student = dspy.Predict(sig)

        # Metric: normalized exact match against the example's expected answer
        # (a truthiness check accepted every non-empty output, so bootstrap
        # never discriminated good demos from bad ones).
        def metric(e, p, t=None):
            gold = getattr(e, "output", getattr(e, "answer", ""))
            pred = getattr(p, "output", getattr(p, "answer", ""))
            return 1.0 if answer_exact_match(gold, pred) else 0.0

        optimizers = {
            # DSPy's own default demo cap (16). Pinning k=4 contradicted the
            # documented "demo caps stay at DSPy defaults" contract.
            "labeled_few_shot": lambda: dspy.LabeledFewShot().compile(
                student=student, trainset=trainset or []
            ),
            # Demo caps stay at DSPy's defaults (labeled 16 / bootstrapped 4).
            "bootstrap_few_shot": lambda: dspy.BootstrapFewShot(
                metric=metric,
                # None keeps metric truthiness; 0 would accept every score >= 0.
                metric_threshold=None,
            ).compile(student=student, trainset=trainset or []),
        }

        if trainset:
            compiled = optimizers.get(optimizer, optimizers["labeled_few_shot"])()
            applied_optimizer = (
                optimizer if optimizer in optimizers else "labeled_few_shot"
            )
        else:
            compiled = student  # No trainset, just save the base program
            applied_optimizer = "none"

        compiled.save(str(skill_path / "program.json"))

        inputs, arrow, outputs = sig.rpartition("->")
        if not arrow:
            inputs, outputs = sig, "answer"
        signature_data = {
            "inputs": inputs.strip().split(", "),
            "outputs": outputs.strip().split(", "),
            "module_type": "predict",
        }
        write_json(skill_path / "signature.json", signature_data)
        write_json(
            skill_path / "metadata.json",
            {
                "skill": name,
                "module_type": "predict",
                "optimizer": applied_optimizer,
                "demos": _count_demos(compiled),
                "trainset_size": len(trainset or []),
            },
        )

        skill.program_path = str(skill_path / "program.json")

        # Best-effort graph write: the artifact is already on disk, so an
        # unreachable FalkorDB must not turn a finished compile into a failure.
        # No score is recorded — nothing was evaluated here.
        try:
            from dspytools.graph.skill_graph import FalkorDBSkillGraph

            FalkorDBSkillGraph().record_program(
                run_id=f"skill_{name}",
                optimizer=applied_optimizer,
                score=None,
                skill_name=name,
                program_path=str(skill_path / "program.json"),
            )
        except Exception as exc:
            _log.warning("skill_graph_record_failed", skill=name, error=str(exc))

        return {
            "status": "compiled",
            "skill": name,
            "optimizer": applied_optimizer,
            "demos": _count_demos(compiled),
            "trainset_size": len(trainset or []),
            "score": None,
            "has_program": True,
        }

    def generate_from_program(
        self, run_id: str, skill_name: str, description: str
    ) -> Skill | None:
        """Generate a skill from an existing compiled program."""

        run_path = compiled_dir() / run_id
        prog_path = run_path / "program.json"
        sig_path = run_path / "signature.json"

        if not prog_path.exists():
            return None

        # Copy program to skill directory
        skill_dir = self.dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(prog_path, skill_dir / "program.json")
        sig_str = "question -> answer"
        if sig_path.exists():
            shutil.copy(sig_path, skill_dir / "signature.json")
            sig_data = read_json(sig_path)
            sig_str = f"{', '.join(sig_data.get('inputs', ['input']))} -> {', '.join(sig_data.get('outputs', ['output']))}"
        # metadata.json carries module_type; without it the loader falls back
        # to a bare Predict and cannot rebuild a CoT/ReAct artifact.
        meta_path = run_path / "metadata.json"
        if meta_path.exists():
            shutil.copy(meta_path, skill_dir / "metadata.json")

        skill = self.create_skill(skill_name, description, sig_str)
        skill.program_path = str(skill_dir / "program.json")
        return skill

    def auto_optimize_skill(self, name: str, trainset: list | None = None) -> dict:
        """Auto-optimize a skill using the GFL pipeline.

        Args:
            name: Skill name to optimize.
            trainset: Optional training examples. When None and no data file
                is found, falls back to labeled_few_shot on a minimal synthetic set.
        """

        skill = self.loader.get(name)
        if not skill or not skill.has_program:
            return {"error": f"Skill '{name}' has no compiled program"}
        assert skill.path is not None, f"Skill '{name}' has no path"

        # Try to auto-discover a trainset alongside the skill directory
        if trainset is None:
            for candidate in ("trainset.json", "train.json", "examples.json"):
                p = skill.path / candidate
                if p.exists():
                    from dspytools.core.loaders import load_trainset

                    trainset = load_trainset(str(p))
                    break

        if not trainset:
            return {
                "error": (
                    f"No training data for skill '{name}'. "
                    "Pass a trainset or place trainset.json/train.json/examples.json "
                    "in the skill directory."
                ),
            }

        from dspytools.core.setup import LMRegistry, setup_dspy
        from dspytools.gfl.pipeline import GFLPipeline  # lazy: breaks skills↔gfl cycle

        setup_dspy()

        student = load_skill_program(skill.path)
        if student is None:
            return {
                "error": (
                    f"Skill '{name}' has no loadable program "
                    "(missing, or compiled by another DSPy — recompile it)"
                )
            }
        student.set_lm(LMRegistry.get_or_default())

        pipeline = GFLPipeline()
        result = pipeline.run(student, trainset)

        if result.get("best_optimizer"):
            best_prog, _ = pipeline.results[result["best_optimizer"]]
            best_prog.save(str(skill.path / "program.json"))
            return {
                "status": "optimized",
                "skill": name,
                "best": result["best_optimizer"],
                "score": result["best_score"],
            }

        return {"error": "Optimization produced no results"}

    def get(self, name: str) -> Skill | None:
        """Resolve a skill by exact identity (never by BM25 rank)."""
        return self.loader.get(name)

    def list_skills(self) -> list[Skill]:
        return self.loader.load_all()

    def search(self, query: str, k: int = 5) -> list[Skill]:
        self.loader.load_all()
        return self.loader.search(query, k)

    def search_hybrid(self, query: str, k: int = 5) -> list[Skill]:
        """Opt-in embedding + BM25 search (falls back to BM25 offline)."""
        self.loader.load_all()
        return self.loader.search_embeddings(query, k)
