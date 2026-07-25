"""Skill Manager — lifecycle: create, compile, optimize, integrate.

Skills follow the harness-so pattern:
  1. CREATE: Generate SKILL.md from task description
  2. COMPILE: Generate DSPy program, run optimizer, save compiled
  3. INTEGRATE: Load into agent pipeline, BM25-index
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dspytools.config.settings import compiled_dir, skills_dir as get_skills_dir
from dspytools.core._dspy import dspy
from dspytools.core._io import read_json
from dspytools.graph.skill_graph import FalkorDBSkillGraph
from dspytools.skills.loader import Skill, SkillLoader


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
name: {name}
description: {description}
signature: {signature}
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
        # Search all paths via loader (project-local, user-wide, ~/.agents)
        loader = SkillLoader(str(self.dir))
        skills = loader.load_all()
        skill = next((s for s in skills if s.name == name), None)
        if not skill:
            return {"error": f"Skill '{name}' not found"}

        skill_path = skill.path
        if not skill_path:
            return {"error": f"Skill '{name}' has no path"}

        sig = skill.frontmatter.get("signature", "question -> answer")
        student = dspy.Predict(sig)

        # Simple metric: output has content
        def metric(e, p, t=None):
            return 1.0 if getattr(p, "output", getattr(p, "answer", "")) else 0.0

        optimizers = {
            "labeled_few_shot": lambda: dspy.LabeledFewShot(
                k=min(4, len(trainset or []))
            ).compile(student=student, trainset=trainset or []),
            "bootstrap_few_shot": lambda: dspy.BootstrapFewShot(
                metric=metric, max_labeled_demos=4, max_bootstrapped_demos=4
            ).compile(student=student, trainset=trainset or []),
        }

        if trainset:
            compiled = optimizers.get(optimizer, optimizers["labeled_few_shot"])()
        else:
            compiled = student  # No trainset, just save the base program

        from dspytools.core._io import write_json

        compiled.save(str(skill_path / "program.json"))

        write_json(
            skill_path / "signature.json",
            {
                "inputs": sig.split("->")[0].strip().split(", "),
                "outputs": sig.split("->")[1].strip().split(", "),
            },
        )

        skill.program_path = str(skill_path / "program.json")

        # Record in FalkorDB graph
        graph = FalkorDBSkillGraph()
        graph.record_program(
            run_id=f"skill_{name}",
            optimizer=optimizer,
            score=1.0 if trainset else 0.0,
        )

        return {
            "status": "compiled",
            "skill": name,
            "optimizer": optimizer,
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
        if sig_path.exists():
            shutil.copy(sig_path, skill_dir / "signature.json")
            sig_data = read_json(sig_path)
            sig_str = f"{', '.join(sig_data.get('inputs', ['input']))} -> {', '.join(sig_data.get('outputs', ['output']))}"
        else:
            sig_str = "question -> answer"

        return self.create_skill(skill_name, description, sig_str)

    def auto_optimize_skill(self, name: str, trainset: list | None = None) -> dict:
        """Auto-optimize a skill using the GFL pipeline.

        Args:
            name: Skill name to optimize.
            trainset: Optional training examples. When None and no data file
                is found, falls back to labeled_few_shot on a minimal synthetic set.
        """

        skill = self.loader.load_all()  # populate index
        skill = next((s for s in skill if s.name == name), None)
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

        student = dspy.Predict(skill.frontmatter.get("signature", "question -> answer"))
        student.load(str(skill.path / "program.json"))
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

    def list_skills(self) -> list[Skill]:
        return self.loader.load_all()

    def search(self, query: str, k: int = 5) -> list[Skill]:
        self.loader.load_all()
        return self.loader.search(query, k)
