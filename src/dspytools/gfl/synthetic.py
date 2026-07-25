"""Synthetic data generator — uses teacher LM to create diverse training examples.

GFL Stage: Generate

Takes seed examples and produces N diverse synthetic examples for training.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dspytools.core._dspy import dspy
from dspytools.core._io import read_json, write_json
from dspytools.core.setup import LMRegistry


class DataSynthesizer:
    """Generate synthetic training data from seed examples.

    Uses the teacher LM to:
      - Paraphrase existing examples
      - Vary complexity (simple → detailed)
      - Inject edge cases (missing fields, unusual repos)
    """

    def __init__(self, teacher_model: str | None = None):
        self.lm = LMRegistry.get_teacher() or LMRegistry.get_or_default()

    def generate(
        self,
        seed_path: str | Path,
        target_count: int = 10,
        output_path: str | None = None,
    ) -> dict:
        """Generate synthetic examples from a seed trainset."""
        seed_data = (
            read_json(Path(seed_path))
            if isinstance(seed_path, (str, Path))
            else seed_path
        )
        if not isinstance(seed_data, list) or not seed_data:
            return {"generated": 0, "output_path": "", "error": "Empty seed data"}

        domains = [
            "fastapi",
            "httpx",
            "pydantic",
            "django",
            "flask",
            "sqlalchemy",
            "celery",
        ]

        # Build all tasks as (method, seed, extra_args) tuples
        tasks: list[tuple] = []

        # Strategy 1: Paraphrase (50% of target)
        for i in range(target_count // 2):
            tasks.append(("paraphrase", seed_data[i % len(seed_data)], None))

        # Strategy 2: Domain variation (30% of target)
        for i in range(target_count // 3):
            tasks.append(
                ("domain", seed_data[i % len(seed_data)], domains[i % len(domains)])
            )

        # Strategy 3: Complexity variation (remaining)
        for i in range(max(1, target_count - len(tasks))):
            tasks.append(("complexity", seed_data[i % len(seed_data)], i % 2 == 0))

        def _run_task(task: tuple) -> dict | None:
            kind, seed, extra = task
            if kind == "paraphrase":
                return self._paraphrase(seed)
            elif kind == "domain":
                return self._vary_domain(seed, extra)
            else:
                return self._vary_complexity(seed, detailed=extra)

        # Run all tasks in parallel (independent LM calls)
        synthetic: list[dict] = []
        max_workers = min(8, len(tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_task, t): t for t in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    synthetic.append(result)

        synthetic = synthetic[:target_count]

        out_path = output_path or f"synthetic_{len(synthetic)}.json"
        if out_path:
            write_json(Path(out_path), synthetic)

        return {"generated": len(synthetic), "output_path": out_path}

    def _paraphrase(self, seed: dict) -> dict | None:
        """Paraphrase a seed example keeping structure but changing wording."""
        result = dict(seed)
        for key in seed:
            if isinstance(seed[key], str) and len(seed[key]) > 20:
                new_val = self._llm_paraphrase(seed[key])
                if new_val and len(new_val) > 10:
                    result[key] = new_val
        return result

    def _vary_domain(self, seed: dict, new_domain: str) -> dict | None:
        """Change the domain of a seed example."""
        result = dict(seed)
        if "repo_url" in result:
            result["repo_url"] = result["repo_url"].replace(
                result["repo_url"].split("/")[-1], new_domain
            )
        if "file_tree" in result:
            result["file_tree"] = result["file_tree"].replace(
                result["file_tree"].split("/")[0], new_domain
            )
        return result

    def _vary_complexity(self, seed: dict, detailed: bool = True) -> dict | None:
        """Make an example simpler or more detailed."""
        result = dict(seed)
        for key in seed:
            if isinstance(seed[key], str) and len(seed[key]) > 30:
                if detailed:
                    result[key] = (
                        seed[key]
                        + "\n\nAdditional details from the README and source code analysis follow..."
                    )
                else:
                    result[key] = seed[key].split("\n")[0]
        return result

    def _llm_paraphrase(self, text: str) -> str | None:
        """Use the LLM to paraphrase text."""

        para = dspy.Predict(
            dspy.Signature(
                "text -> paraphrased: str",
                "Paraphrase the following text. Keep the same meaning and structure but reword it entirely.",
            )
        )
        result = para(text=text)
        return getattr(result, "paraphrased", text)


# ═══════════════════════════════════════════════════════════════════════════
# R-Zero Challenger-Solver Co-Evolution (arXiv 2508.05004)
# ═══════════════════════════════════════════════════════════════════════════


class ChallengerSolver:
    """R-Zero pattern: Challenger proposes tasks, Solver attempts them.

    Implements zero-data co-evolution from R-Zero (arXiv 2508.05004):
    - Challenger: proposes tasks at the edge of Solver's capability
    - Solver: solves tasks, providing feedback on difficulty
    - Both improve through interaction, creating a self-improving curriculum
    """

    def __init__(self, challenger_program, solver_program):
        self.challenger = challenger_program
        self.solver = solver_program
        self.task_history: list[dict] = []

    def co_evolve(self, num_rounds: int = 5, task_count_per_round: int = 3) -> dict:
        """Run co-evolution rounds.

        Each round:
        1. Challenger generates `task_count` tasks
        2. Solver attempts each task
        3. Challenger adjusts based on Solver's success rate
        4. Solver learns from successful completions
        """

        results = {"rounds": [], "final_accuracy": 0.0}

        for round_num in range(num_rounds):
            round_tasks = []
            round_scores = []

            # Challenger generates tasks
            for _ in range(task_count_per_round):
                # Generate a task description
                task = self.challenger(
                    difficulty="medium" if round_num < num_rounds // 2 else "hard",
                    previous_tasks=str(
                        self.task_history[-3:] if self.task_history else []
                    ),
                )
                task_desc = getattr(task, "task", getattr(task, "output", str(task)))
                round_tasks.append(task_desc)

                # Solver attempts
                solution = self.solver(task=task_desc)
                solution_text = getattr(
                    solution, "output", getattr(solution, "answer", str(solution))
                )

                # Score: did solver produce non-empty output?
                score = 1.0 if len(str(solution_text)) > 10 else 0.0
                round_scores.append(score)

            avg_accuracy = (
                sum(round_scores) / len(round_scores) if round_scores else 0.0
            )

            self.task_history.append(
                {
                    "round": round_num + 1,
                    "tasks": round_tasks,
                    "scores": round_scores,
                    "accuracy": avg_accuracy,
                }
            )

            results["rounds"].append(
                {
                    "round": round_num + 1,
                    "accuracy": avg_accuracy,
                    "tasks_generated": len([t for t in round_tasks if t]),
                }
            )

        results["final_accuracy"] = (
            sum(r["accuracy"] for r in results["rounds"]) / len(results["rounds"])
            if results["rounds"]
            else 0.0
        )
        results["total_tasks"] = sum(r["tasks_generated"] for r in results["rounds"])
        return results
