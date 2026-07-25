"""Tests for FalkorDBSkillGraph — FalkorDB-backed skill dependency graph.

Uses mocked FalkorDB — no actual database connection needed.
Mock query results use FalkorDB's result_set format (list-of-lists).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def skill_graph():
    """Create a FalkorDBSkillGraph with mocked backend.

    Returns (graph, mock_graph_obj) where mock_graph_obj.query can be
    configured to return desired query results.
    """
    with patch("dspytools.graph.skill_graph.get_graph_client") as mock_client:
        mock_redis = MagicMock()
        mock_graph = MagicMock()
        mock_client.return_value.redis.return_value = mock_redis
        mock_client.return_value.graph.return_value = mock_graph

        from dspytools.graph.skill_graph import FalkorDBSkillGraph

        g = FalkorDBSkillGraph()
        yield g, mock_graph


def _make_result(result_set: list[list], col_names: list[str] | None = None):
    """Create a mock FalkorDB query result with result_set and header."""
    mock = MagicMock()
    mock.result_set = result_set
    if col_names is None and result_set:
        col_names = [f"col{i}" for i in range(len(result_set[0]))]
    mock.header = [["SCALAR", name] for name in (col_names or [])] if col_names else []
    return mock


# ═══════════════════════════════════════════════════════════════════════════


class TestAddRemoveDependency:
    def test_add_dependency(self, skill_graph):
        g, mock_g = skill_graph
        g.add_dependency("skill_a", "skill_b")
        mock_g.query.assert_called_once()
        cypher = mock_g.query.call_args[0][0]
        assert "MERGE" in cypher
        assert "DEPENDS_ON" in cypher

    def test_remove_dependency(self, skill_graph):
        g, mock_g = skill_graph
        g.remove_dependency("skill_a", "skill_b")
        mock_g.query.assert_called_once()
        cypher = mock_g.query.call_args[0][0]
        assert "DELETE r" in cypher

    def test_add_dependency_params(self, skill_graph):
        g, mock_g = skill_graph
        g.add_dependency("skill_a", "skill_b")
        params = mock_g.query.call_args[0][1]
        assert params["skill"] == "skill_a"
        assert params["depends_on"] == "skill_b"


class TestGetDependents:
    def test_get_dependents(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([["skill_c"], ["skill_d"]], ["name"])
        deps = g.get_dependents("skill_a")
        assert deps == ["skill_c", "skill_d"]

    def test_get_dependents_empty(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        deps = g.get_dependents("orphan")
        assert deps == []

    def test_get_dependencies(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([["base_a"], ["base_b"]], ["name"])
        deps = g.get_dependencies("skill_a")
        assert deps == ["base_a", "base_b"]


class TestTransitiveDependents:
    def test_transitive_dependents(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [["skill_b"], ["skill_c"], ["skill_d"]], ["name"]
        )
        deps = g.transitive_dependents("skill_a")
        assert len(deps) == 3
        assert "skill_b" in deps

    def test_transitive_empty(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        deps = g.transitive_dependents("leaf")
        assert deps == []

    def test_transitive_cypher_pattern(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        g.transitive_dependents("root")
        cypher = mock_g.query.call_args[0][0]
        assert "[:DEPENDS_ON*1..10]" in cypher


class TestListSkills:
    def test_list_skills(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [["skill_a", "Does A", "1.0", 0.85], ["skill_b", "Does B", "2.0", 0.92]],
            ["name", "description", "version", "score"],
        )
        skills = g.list_skills()
        assert len(skills) == 2
        assert skills[0]["name"] == "skill_a"
        assert skills[1]["score"] == 0.92

    def test_list_skills_empty(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        assert g.list_skills() == []


class TestSkillStats:
    def test_skill_stats(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [["skill_a", 3, 2]],
            ["name", "dependent_count", "dependency_count"],
        )
        stats = g.skill_stats("skill_a")
        assert stats["name"] == "skill_a"
        assert stats["dependent_count"] == 3
        assert stats["dependency_count"] == 2

    def test_skill_stats_missing(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        stats = g.skill_stats("nobody")
        assert stats["name"] == "nobody"
        assert stats["dependent_count"] == 0


class TestRecordProgram:
    def test_record_program(self, skill_graph):
        g, mock_g = skill_graph
        g.record_program("run_001", "gepa", 0.85)
        assert mock_g.query.call_count >= 1

    def test_record_program_with_parent(self, skill_graph):
        g, mock_g = skill_graph
        g.record_program("run_002", "mipro", 0.75, parent_id="run_001")
        assert mock_g.query.call_count >= 2

    def test_record_program_with_dataset(self, skill_graph):
        g, mock_g = skill_graph
        g.record_program("run_003", "copro", 0.65, dataset_hash="abc123")
        assert mock_g.query.call_count >= 2


class TestProgramLineage:
    def test_lineage(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [
                ["run_001", "mipro", 0.75, "2024-01-01"],
                ["run_002", "gepa", 0.85, "2024-01-02"],
            ],
            ["id", "optimizer", "score", "created_at"],
        )
        lineage = g.program_lineage("run_003")
        assert len(lineage) == 2
        assert lineage[0]["optimizer"] == "mipro"

    def test_lineage_no_ancestors(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        assert g.program_lineage("orphan") == []


class TestRecordTaskProfile:
    def test_record_success(self, skill_graph):
        g, mock_g = skill_graph
        g.record_task_profile("qa_classify", "labeled_few_shot", success=True)
        cypher = mock_g.query.call_args[0][0]
        assert "TRIED_PATTERN" in cypher
        params = mock_g.query.call_args[0][1]
        assert params["profile"] == "qa_classify"
        assert params["success_int"] == 1

    def test_record_failure(self, skill_graph):
        g, mock_g = skill_graph
        g.record_task_profile("qa_classify", "gepa", success=False)
        params = mock_g.query.call_args[0][1]
        assert params["success_int"] == 0


class TestBestPattern:
    def test_best_pattern_found(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([["mipro", 0.92]], ["type", "rate"])
        best = g.best_pattern("qa_classify")
        assert best == "mipro"

    def test_best_pattern_none(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result([])
        best = g.best_pattern("unknown")
        assert best is None

    def test_best_pattern_ordered(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [["gepa", 0.95], ["mipro", 0.85]], ["type", "rate"]
        )
        best = g.best_pattern("qa_classify")
        assert best == "gepa"


class TestTaskProfiles:
    def test_get_task_profiles(self, skill_graph):
        g, mock_g = skill_graph
        mock_g.query.return_value = _make_result(
            [["qa", "mipro", 5, 0.8]],
            ["profile", "pattern", "count", "success_rate"],
        )
        profiles = g.get_task_profiles()
        assert len(profiles) == 1
        assert profiles[0]["profile"] == "qa"
        assert profiles[0]["success_rate"] == 0.8
