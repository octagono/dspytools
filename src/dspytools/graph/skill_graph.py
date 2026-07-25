"""FalkorDB-backed SkillGraph — replaces JSON persistence.

Provides graph-based skill dependency tracking with transitive traversal.
"""

from __future__ import annotations

from dspytools.graph.client import get_graph_client


class FalkorDBSkillGraph:
    """Graph-backed skill dependency tracker.

    Replaces evolve/self_evolve.py SkillGraph with FalkorDB.
    Supports transitive dependency queries and task profile tracking.
    """

    def __init__(self) -> None:
        self.client = get_graph_client()
        self.graph = self.client.graph("skills")

    @staticmethod
    def _rows_to_dicts(result) -> list[dict]:
        """Convert FalkorDB QueryResult to list of dicts using header metadata."""
        if not result.result_set or not result.header:
            return []
        col_names = [h[1] for h in result.header]
        return [dict(zip(col_names, row)) for row in result.result_set]

    @staticmethod
    def _col_values(result, col_index: int = 0) -> list:
        """Extract a single column from FalkorDB QueryResult as a flat list."""
        return [row[col_index] for row in result.result_set]

    def add_dependency(self, skill: str, depends_on: str) -> None:
        """Add edge: skill depends_on depends_on."""
        self.graph.query(
            """
            MERGE (a:Skill {name: $skill})
            MERGE (b:Skill {name: $depends_on})
            MERGE (a)-[:DEPENDS_ON]->(b)
            """,
            {"skill": skill, "depends_on": depends_on},
        )

    def remove_dependency(self, skill: str, depends_on: str) -> None:
        """Remove dependency edge."""
        self.graph.query(
            """
            MATCH (a:Skill {name: $skill})-[r:DEPENDS_ON]->(b:Skill {name: $depends_on})
            DELETE r
            """,
            {"skill": skill, "depends_on": depends_on},
        )

    def get_dependents(self, skill: str) -> list[str]:
        """Get direct dependents of a skill."""
        result = self.graph.query(
            """
            MATCH (a:Skill {name: $skill})<-[:DEPENDS_ON]-(b:Skill)
            RETURN b.name AS name
            """,
            {"skill": skill},
        )
        return self._col_values(result, 0)

    def get_dependencies(self, skill: str) -> list[str]:
        """Get direct dependencies of a skill."""
        result = self.graph.query(
            """
            MATCH (a:Skill {name: $skill})-[:DEPENDS_ON]->(b:Skill)
            RETURN b.name AS name
            """,
            {"skill": skill},
        )
        return self._col_values(result, 0)

    def transitive_dependents(self, skill: str) -> list[str]:
        """BFS traversal for all indirect dependents."""
        result = self.graph.query(
            """
            MATCH (a:Skill {name: $skill})<-[:DEPENDS_ON*1..10]-(b:Skill)
            RETURN DISTINCT b.name AS name
            ORDER BY name
            """,
            {"skill": skill},
        )
        return self._col_values(result, 0)

    def on_improvement(self, skill: str) -> list[str]:
        """Return all transitive dependents needing re-evaluation."""
        return self.transitive_dependents(skill)

    def list_skills(self) -> list[dict]:
        """List all skills in the graph."""
        result = self.graph.query(
            """
            MATCH (s:Skill)
            RETURN s.name AS name, s.description AS description,
                   s.version AS version, s.score AS score
            ORDER BY s.name
            """
        )
        return self._rows_to_dicts(result)

    def skill_stats(self, skill: str) -> dict:
        """Get statistics for a skill."""
        result = self.graph.query(
            """
            MATCH (s:Skill {name: $skill})
            OPTIONAL MATCH (s)<-[:DEPENDS_ON]-(dependents)
            OPTIONAL MATCH (s)-[:DEPENDS_ON]->(dependencies)
            RETURN s.name AS name,
                   count(DISTINCT dependents) AS dependent_count,
                   count(DISTINCT dependencies) AS dependency_count
            """,
            {"skill": skill},
        )
        rows = self._rows_to_dicts(result)
        return (
            rows[0]
            if rows
            else {"name": skill, "dependent_count": 0, "dependency_count": 0}
        )

    def record_task_profile(
        self, profile: str, pattern_type: str, success: bool
    ) -> None:
        """Record a task execution result."""
        self.graph.query(
            """
            MERGE (p:TaskProfile {profile: $profile})
            MERGE (t:Pattern {type: $pattern_type})
            MERGE (p)-[r:TRIED_PATTERN]->(t)
            SET r.count = COALESCE(r.count, 0) + 1,
                r.success_count = COALESCE(r.success_count, 0) + $success_int,
                r.updated_at = timestamp()
            WITH p, t, r
            SET r.success_rate = toFloat(r.success_count) / toFloat(r.count)
            """,
            {
                "profile": profile,
                "pattern_type": pattern_type,
                "success_int": 1 if success else 0,
            },
        )

    def best_pattern(self, profile: str) -> str | None:
        """Get best pattern for a task profile (count >= 3)."""
        result = self.graph.query(
            """
            MATCH (p:TaskProfile {profile: $profile})-[r:TRIED_PATTERN]->(t:Pattern)
            WHERE r.count >= 3
            RETURN t.type AS type, r.success_rate AS rate
            ORDER BY r.rate DESC
            LIMIT 1
            """,
            {"profile": profile},
        )
        rows = self._rows_to_dicts(result)
        return rows[0]["type"] if rows else None

    def get_task_profiles(self) -> list[dict]:
        """Get all task profiles with their patterns."""
        result = self.graph.query(
            """
            MATCH (p:TaskProfile)-[r:TRIED_PATTERN]->(t:Pattern)
            RETURN p.profile AS profile, t.type AS pattern,
                   r.count AS count, r.success_rate AS success_rate
            ORDER BY p.profile, r.success_rate DESC
            """
        )
        return self._rows_to_dicts(result)

    def record_program(
        self,
        run_id: str,
        optimizer: str,
        score: float,
        parent_id: str | None = None,
        dataset_hash: str | None = None,
    ) -> None:
        """Record a compiled program in the graph."""
        self.graph.query(
            """
            MERGE (r:Run {id: $run_id})
            SET r.optimizer = $optimizer,
                r.score = $score,
                r.created_at = timestamp()
            """,
            {"run_id": run_id, "optimizer": optimizer, "score": score},
        )

        if parent_id:
            self.graph.query(
                """
                MATCH (child:Run {id: $run_id})
                MERGE (parent:Run {id: $parent_id})
                MERGE (child)-[:CHILD_OF]->(parent)
                """,
                {"run_id": run_id, "parent_id": parent_id},
            )

        if dataset_hash:
            self.graph.query(
                """
                MATCH (r:Run {id: $run_id})
                MERGE (d:Dataset {hash: $hash})
                MERGE (r)-[:TRAINED_ON]->(d)
                """,
                {"run_id": run_id, "hash": dataset_hash},
            )

    def program_lineage(self, run_id: str) -> list[dict]:
        """Get full ancestry chain for a program."""
        result = self.graph.query(
            """
            MATCH (r:Run {id: $run_id})-[:CHILD_OF*0..10]->(ancestor:Run)
            RETURN ancestor.id AS id, ancestor.optimizer AS optimizer,
                   ancestor.score AS score, ancestor.created_at AS created_at
            ORDER BY ancestor.created_at
            """,
            {"run_id": run_id},
        )
        return self._rows_to_dicts(result)
