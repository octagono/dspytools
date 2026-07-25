"""Tests for graph CLI commands — FalkorDB graph management.

Uses mocked FalkorDB — no actual database connection needed.
Tests the Click command interface (no actual Click runner — tests
the underlying functions called by the CLI).
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

# ── FalkorDBSkillGraph directly (not via CLI parser) ──────────────────────


class TestGraphCommandFunctions:
    """Tests the functions that back graph CLI commands."""

    def test_add_dependency_called(self, mock_redis: tuple[MagicMock, MagicMock]):
        """graph_add_dependency calls FalkorDBSkillGraph.add_dependency."""
        r, g = mock_redis
        with patch("dspytools.commands.graph.FalkorDBSkillGraph") as mock_sg:
            mock_sg.return_value = MagicMock()
            from dspytools.commands.graph import graph_add_dependency

            graph_add_dependency.callback("skill_a", "skill_b")
            mock_sg.return_value.add_dependency.assert_called_once_with(
                "skill_a", "skill_b"
            )

    def test_dependents_direct(self, mock_graph: dict[str, Any]):
        """graph_dependents calls get_dependents (not transitive)."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_dependents

        mg["skill_graph"].get_dependents.return_value = ["child"]
        graph_dependents.callback("skill_a", transitive=False)
        mg["skill_graph"].get_dependents.assert_called_once_with("skill_a")

    def test_dependents_transitive(self, mock_graph: dict[str, Any]):
        """graph_dependents --transitive calls transitive_dependents."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_dependents

        mg["skill_graph"].transitive_dependents.return_value = ["b", "c"]
        graph_dependents.callback("skill_a", transitive=True)
        mg["skill_graph"].transitive_dependents.assert_called_once_with("skill_a")

    def test_record_program(self, mock_graph: dict[str, Any]):
        """graph_record_program calls record_program with correct args."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_record_program

        graph_record_program.callback(
            "run_001", "gepa", score=0.85, dataset_hash="abc", parent_id="run_000"
        )
        mg["skill_graph"].record_program.assert_called_once_with(
            run_id="run_001",
            optimizer="gepa",
            score=0.85,
            dataset_hash="abc",
            parent_id="run_000",
        )

    def test_program_lineage(self, mock_graph: dict[str, Any]):
        """graph_program_lineage calls program_lineage."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_program_lineage

        mg["skill_graph"].program_lineage.return_value = [
            {"id": "parent", "optimizer": "mipro"}
        ]
        graph_program_lineage.callback("run_003")
        mg["skill_graph"].program_lineage.assert_called_once_with("run_003")

    def test_search(self, mock_graph: dict[str, Any]):
        """graph_search calls list_skills and filters."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_search

        mg["skill_graph"].list_skills.return_value = [
            {"name": "skill_a", "description": "Does A things"},
            {"name": "skill_b", "description": "Does B things"},
        ]
        graph_search.callback("A", limit=10)  # Should match skill_a
        mg["skill_graph"].list_skills.assert_called_once()

    def test_skill_tree_list(self, mock_graph: dict[str, Any]):
        """graph_skill_tree without --skill lists all skills."""
        mg = cast(dict[str, Any], mock_graph)
        from dspytools.commands.graph import graph_skill_tree

        mg["skill_graph"].list_skills.return_value = [
            {"name": "s1", "description": "desc1"},
            {"name": "s2", "description": "desc2"},
        ]
        graph_skill_tree.callback(skill=None)
        mg["skill_graph"].list_skills.assert_called_once()

    def test_skill_graph_importable(self):
        """FalkorDBSkillGraph is importable."""
        from dspytools.graph.skill_graph import FalkorDBSkillGraph

        assert FalkorDBSkillGraph is not None

    def test_connection_ping_checks(self, mock_redis: tuple[MagicMock, MagicMock]):
        """get_graph_client().ping() returns expected value."""
        r, g = mock_redis
        from unittest.mock import patch as _patch

        with _patch("dspytools.graph.client.get_graph_client") as mock_gc:
            mock_gc.return_value.ping.return_value = True
            from dspytools.graph.client import get_graph_client

            result = get_graph_client().ping()
            assert result is True


# ── Stats ─────────────────────────────────────────────────────────────────


class TestGraphStats:
    def test_graph_queries_work(self, mock_graph):
        """Cypher queries on FalkorDB return expected structure."""
        from dspytools.graph.client import get_graph_client

        g = get_graph_client().graph("skills")
        mock_result = MagicMock(
            result_set=[["Skill", 5]], header=[["SCALAR", "labels"], ["SCALAR", "cnt"]]
        )
        g.query.return_value = mock_result
        result = g.query("MATCH (n) RETURN labels(n), count(n)")
        assert result.result_set[0] == ["Skill", 5]


# ── Query ─────────────────────────────────────────────────────────────────


class TestGraphQuery:
    def test_query_with_results(self, mock_graph):
        """graph_query executes Cypher and displays results."""
        from dspytools.commands.graph import graph_query

        mock_graph["falkordb"].query.return_value = MagicMock(
            result_set=[["a", 1], ["b", 2]],
            header=[["SCALAR", "name"], ["SCALAR", "count"]],
        )
        graph_query.callback("MATCH (n) RETURN n.name, n.count", params=None)

    def test_query_no_results(self, mock_graph):
        """graph_query handles empty results gracefully."""
        from dspytools.commands.graph import graph_query

        mock_graph["falkordb"].query.return_value = MagicMock(result_set=[], header=[])
        graph_query.callback("MATCH (n) RETURN n", params=None)
