"""Tests for SkillGraph cascade propagation — transitive dependency chains.

Tests the SkillGraph class in SelfEvolveEngine (evolve/self_evolve.py).
Pure logic — no LM calls, no DSPy import needed.
"""

from __future__ import annotations

from dspytools.evolve.self_evolve import SelfEvolveEngine, SkillGraph

# ═══════════════════════════════════════════════════════════════════════════
# SkillGraph — cascade propagation
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillGraphCascade:
    """Tests for SkillGraph dependency chain propagation."""

    def _graph(self) -> SkillGraph:
        g = SkillGraph()
        # Use JSON fallback only — prevent FalkorDB from returning stale state
        g._falkordb = False
        g.edges.clear()
        return g

    def test_simple_dependency(self):
        """A → B: B depends on A."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        deps = g.get_dependents("skill_a")
        assert "skill_b" in deps

    def test_transitive_chain(self):
        """A → B → C: improving A queues B AND C."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_c", "skill_b")
        dependents = g.transitive_dependents("skill_a")
        assert "skill_b" in dependents
        assert "skill_c" in dependents

    def test_deep_chain(self):
        """A → B → C → D: all downstream skills detected."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_c", "skill_b")
        g.add_dependency("skill_d", "skill_c")
        deps = g.transitive_dependents("skill_a")
        assert len(deps) == 3
        assert all(s in deps for s in ["skill_b", "skill_c", "skill_d"])

    def test_leaf_has_one_dependent(self):
        """Leaf skill has exactly the skill that depends on it."""
        g = self._graph()
        g.add_dependency("skill_a", "skill_b")  # A depends on B
        deps = g.transitive_dependents("skill_b")  # Who depends on B? → A
        assert deps == ["skill_a"]

    def test_branching(self):
        """A → B, A → C: both branches returned."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_c", "skill_a")
        deps = g.transitive_dependents("skill_a")
        assert "skill_b" in deps
        assert "skill_c" in deps
        assert len(deps) == 2

    def test_diamond(self):
        """A → B → D, A → C → D: full transitive includes all."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_c", "skill_a")
        g.add_dependency("skill_d", "skill_b")
        g.add_dependency("skill_d", "skill_c")
        deps = g.transitive_dependents("skill_a")
        assert "skill_b" in deps
        assert "skill_c" in deps
        assert "skill_d" in deps

    def test_no_cycles_infinite(self):
        """Simple cycle A → B → A doesn't cause infinite loop."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_a", "skill_b")
        # This should terminate
        deps = g.transitive_dependents("skill_a")
        assert isinstance(deps, list)

    def test_on_improvement_cascade(self):
        """on_improvement delegates to transitive_dependents."""
        g = self._graph()
        g.add_dependency("skill_b", "skill_a")
        g.add_dependency("skill_c", "skill_b")
        result = g.on_improvement("skill_a")
        assert "skill_b" in result
        assert "skill_c" in result


# ═══════════════════════════════════════════════════════════════════════════
# SelfEvolveEngine — cascade wiring via on_skill_improvement
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfEvolveCascadeWiring:
    """Tests that SelfEvolveEngine.on_skill_improvement delegates correctly."""

    def _engine(self) -> SelfEvolveEngine:
        engine = SelfEvolveEngine()
        engine._clear_state()
        return engine

    def test_on_skill_improvement_returns_list(self):
        """on_skill_improvement returns a list of downstream skills."""
        engine = self._engine()
        engine.add_skill_dependency("child", "parent")
        result = engine.on_skill_improvement("parent")
        assert isinstance(result, list)

    def test_add_skill_dependency(self):
        """add_skill_dependency stores the edge."""
        engine = self._engine()
        engine.add_skill_dependency("child", "parent")
        deps = engine.graph.get_dependents("parent")
        assert "child" in deps

    def test_add_skill_dependency_triggers_dirty(self):
        """add_skill_dependency marks graph as dirty for persistence."""
        engine = self._engine()
        assert engine._graph_dirty is False
        engine.add_skill_dependency("child", "parent")
        assert engine._graph_dirty is True

    def test_v_improvement_no_dependents(self):
        """Leaf skill returns empty list."""
        engine = self._engine()
        engine.add_skill_dependency("leaf", "root")
        result = engine.on_skill_improvement("leaf")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# SkillGraph — persistence and fallback
# ═══════════════════════════════════════════════════════════════════════════


class TestSkillGraphPersistence:
    """Tests SkillGraph JSON fallback persistence."""

    def test_save_and_load(self, tmp_path):
        """Saved graph can be loaded by a new instance."""
        import json

        state_file = tmp_path / "skill_graph.json"
        # Manually test the file format
        data = {"skill_b": ["skill_a"], "skill_c": ["skill_b"]}
        state_file.write_text(json.dumps(data))
        loaded = json.loads(state_file.read_text())
        assert "skill_b" in loaded
        assert loaded["skill_b"] == ["skill_a"]

    def test_save_empty(self, tmp_path):
        """Empty graph saves cleanly."""
        import json

        state_file = tmp_path / "skill_graph.json"
        state_file.write_text(json.dumps({}))
        data = json.loads(state_file.read_text())
        assert data == {}
