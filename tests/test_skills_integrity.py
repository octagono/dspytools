"""Hermetic integrity tests for the `skills` subsystem.

No live LM, no Redis, no FalkorDB, no network: skills live in `tmp_path`
directories, the graph and the embedder are monkeypatched, and DSPy runs on
saved artifacts only.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from dspytools.core._dspy import dspy
from dspytools.skills.loader import SkillLoader
from dspytools.skills.manager import SkillManager, load_skill_program


def _write_skill(
    root: Path,
    name: str,
    description: str,
    signature: str = "question -> answer",
    program: dict | None = None,
) -> Path:
    """Write a minimal skill directory (JSON-quoted frontmatter, like the real one)."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {json.dumps(name)}\n"
        f"description: {json.dumps(description)}\n"
        f"signature: {json.dumps(signature)}\n"
        "---\n"
        f"# {name}\n\n{description}\n"
    )
    if program is not None:
        (skill_dir / "program.json").write_text(json.dumps(program))
    return skill_dir


@pytest.fixture
def skills_root(tmp_path, monkeypatch):
    """Isolated skills root: manager + loader both point at tmp_path only."""
    root = tmp_path / "skills"
    root.mkdir()
    isolated_user = tmp_path / "isolated-user-skills"
    isolated_user.mkdir()
    monkeypatch.setattr("dspytools.skills.manager.get_skills_dir", lambda: root)
    monkeypatch.setattr(
        "dspytools.skills.loader._user_skills_dir", lambda: isolated_user
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return root


# ── 1. --gepa-skills feeds InstructionProposer one source per skill ──────────


def test_instruction_proposer_passes_each_skill_directory(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    _write_skill(root, "beta-skill", "second")
    _write_skill(root, "alpha-skill", "first")
    monkeypatch.setattr("dspytools.config.settings.skills_dir", lambda: root)

    from dspytools.commands.compile import _instruction_proposer

    proposer = _instruction_proposer()

    assert [skill.name for skill in proposer.skills] == ["alpha-skill", "beta-skill"]


def test_instruction_proposer_empty_library_raises_actionable_error(
    tmp_path, monkeypatch
):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "not-a-skill").mkdir()
    monkeypatch.setattr("dspytools.config.settings.skills_dir", lambda: root)

    import click

    from dspytools.commands.compile import _instruction_proposer

    with pytest.raises(click.ClickException, match="no skills found"):
        _instruction_proposer()


def test_instruction_proposer_names_invalid_legacy_frontmatter(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    legacy = root / "legacy-skill"
    legacy.mkdir(parents=True)
    (legacy / "SKILL.md").write_text(
        "---\nname: legacy-skill\ndescription: Totals: unquoted colon\n---\nbody\n"
    )
    monkeypatch.setattr("dspytools.config.settings.skills_dir", lambda: root)

    import click

    from dspytools.commands.compile import _instruction_proposer

    with pytest.raises(click.ClickException, match="invalid YAML frontmatter"):
        _instruction_proposer()


# ── 2. create_skill emits valid YAML ────────────────────────────────────────


@pytest.mark.parametrize(
    "description",
    [
        "Totals: report the sum of the rows",
        "Tags #1 and #2 are hashes",
        'Quotes "inside" the description',
        "Multi: colon, then # hash",
    ],
)
def test_create_skill_frontmatter_round_trips_through_yaml(skills_root, description):
    manager = SkillManager()
    manager.create_skill("yaml-skill", description, "question -> answer")

    text = (skills_root / "yaml-skill" / "SKILL.md").read_text()
    frontmatter = yaml.safe_load(text.split("---")[1])
    assert frontmatter["description"] == description
    assert frontmatter["name"] == "yaml-skill"

    # The exact consumer that aborted: GEPA's proposer parses the same file.
    from dspy.teleprompt.gepa.instruction_proposal import InstructionProposer

    proposer = InstructionProposer(skills=[str(skills_root / "yaml-skill")])
    assert len(proposer.skills) == 1
    assert proposer.skills[0].description == description

    # The loader's own parser round-trips it too.
    loaded = SkillLoader(str(skills_root)).get("yaml-skill")
    assert loaded is not None
    assert loaded.description == description


# ── 3./4. compile --trainset reports the real optimizer and demo count ──────


def _trainset_file(tmp_path: Path, size: int = 6) -> Path:
    path = tmp_path / "train.json"
    path.write_text(
        json.dumps([{"question": f"q{i}", "answer": f"a{i}"} for i in range(size)])
    )
    return path


def test_skills_compile_with_trainset_reports_optimizer_and_demos(
    skills_root, tmp_path, monkeypatch
):
    manager = SkillManager()
    manager.create_skill("sum-skill", "add numbers", "question -> answer")
    trainset = _trainset_file(tmp_path)

    monkeypatch.setattr("dspytools.core.setup.setup_dspy", lambda *a, **k: None)

    from dspytools.commands.skills import skills_compile

    result = CliRunner().invoke(
        skills_compile, ["sum-skill", "--trainset", str(trainset)]
    )

    assert result.exit_code == 0, result.output
    assert "labeled_few_shot" in result.output
    assert "6 demos" in result.output
    assert (skills_root / "sum-skill" / "program.json").exists()


def test_skills_compile_without_trainset_says_unoptimized(skills_root):
    manager = SkillManager()
    manager.create_skill("plain-skill", "no data", "question -> answer")

    from dspytools.commands.skills import skills_compile

    result = CliRunner().invoke(skills_compile, ["plain-skill"])

    assert result.exit_code == 0, result.output
    assert "unoptimized" in result.output
    assert "--trainset" in result.output


def test_labeled_few_shot_keeps_dspy_default_k(skills_root, tmp_path):
    manager = SkillManager()
    manager.create_skill("many-skill", "many demos", "question -> answer")

    from dspytools.core.loaders import load_trainset

    examples = load_trainset(str(_trainset_file(tmp_path, size=6)))
    compiled = manager.compile_skill("many-skill", trainset=examples)

    # k=16 upstream: all 6 examples become demos (a k=4 pin would cap at 4).
    assert dspy.LabeledFewShot().k == 16
    assert compiled["demos"] == 6
    assert compiled["optimizer"] == "labeled_few_shot"


# ── 5./6. program load respects module_type and DSPy version ────────────────


def test_program_load_rebuilds_chain_of_thought_shape(skills_root):
    skill_dir = _write_skill(skills_root, "cot-skill", "reasoning skill")
    dspy.ChainOfThought("question -> answer").save(str(skill_dir / "program.json"))
    (skill_dir / "signature.json").write_text(
        json.dumps(
            {"inputs": ["question"], "outputs": ["answer"], "module_type": "cot"}
        )
    )
    (skill_dir / "metadata.json").write_text(json.dumps({"module_type": "cot"}))

    program = load_skill_program(skill_dir)

    assert isinstance(program, dspy.ChainOfThought)
    assert isinstance(load_skill_program(skill_dir), dspy.Module)

    skill = SkillLoader(str(skills_root)).get("cot-skill")
    assert skill is not None
    assert skill.has_program


def test_generate_from_program_sets_program_path_and_carries_module_type(
    skills_root, tmp_path, monkeypatch
):
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "program.json").write_text("{}")
    (run_dir / "signature.json").write_text(
        json.dumps({"inputs": ["question"], "outputs": ["answer"]})
    )
    (run_dir / "metadata.json").write_text(json.dumps({"module_type": "cot"}))
    monkeypatch.setattr(
        "dspytools.skills.manager.compiled_dir", lambda: tmp_path / "runs"
    )

    skill = SkillManager().generate_from_program("run-1", "gen-skill", "generated")

    assert skill is not None
    assert skill.program_path == str(skills_root / "gen-skill" / "program.json")
    assert (skills_root / "gen-skill" / "metadata.json").exists()


def test_stale_dspy_program_is_treated_as_absent(skills_root):
    skill_dir = _write_skill(
        skills_root,
        "stale-skill",
        "compiled by another dspy",
        program={"metadata": {"dspy_version": "3.3.0"}},
    )
    (skill_dir / "signature.json").write_text(
        json.dumps({"inputs": ["question"], "outputs": ["answer"]})
    )

    assert load_skill_program(skill_dir) is None

    skill = SkillLoader(str(skills_root)).get("stale-skill")
    assert skill is not None
    assert skill.program_path is None
    assert skill.has_program is False


# ── 7. index cache tracks deletions and creations ──────────────────────────


def test_index_cache_invalidates_on_delete_and_create(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(
        "dspytools.skills.loader._user_skills_dir", lambda: tmp_path / "none"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_skill(root, "keep-skill", "keep me")
    _write_skill(root, "gone-skill", "gone skill")
    loader = SkillLoader(str(root))

    assert {s.name for s in loader.load_all()} == {"keep-skill", "gone-skill"}

    shutil.rmtree(root / "gone-skill")
    assert {s.name for s in loader.load_all()} == {"keep-skill"}
    assert loader.search("gone") == []

    # A just-created skill is visible immediately (no 5s TTL window).
    _write_skill(root, "fresh-skill", "fresh skill")
    assert {s.name for s in loader.load_all()} == {"keep-skill", "fresh-skill"}


# ── 8. hybrid search is opt-in and cached per model/dimension ──────────────


class _FakeEmbedder:
    """Two-axis embedder: 'alpha' texts point at [1, 0], everything else [0, 1]."""

    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []

    def __call__(self, inputs, batch_size=None, caching=None, **kwargs):
        text = inputs if isinstance(inputs, str) else " ".join(inputs)
        self.calls.append((text, kwargs.get("dimensions")))
        if "alpha" in text:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    @property
    def skill_calls(self) -> int:
        return sum(1 for text, _ in self.calls if not text.startswith("query:"))


def _patch_embedding(monkeypatch, embedder, dimension=8):
    monkeypatch.setattr("dspytools.core._embedder.get_embedder", lambda: embedder)
    monkeypatch.setattr(
        "dspytools.config.settings.embedder_kwargs", lambda: {"model": "fake-embed"}
    )
    monkeypatch.setattr(
        "dspytools.config.settings.embedder_dimension", lambda: dimension
    )


def test_hybrid_search_caches_by_model_and_dimension(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(
        "dspytools.skills.loader._user_skills_dir", lambda: tmp_path / "none"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_skill(root, "alpha-skill", "alpha alpha")
    _write_skill(root, "beta-skill", "beta only")

    embedder = _FakeEmbedder()
    _patch_embedding(monkeypatch, embedder, dimension=8)
    loader = SkillLoader(str(root))

    ranked = loader.search_embeddings("query: alpha", k=2)
    assert [s.name for s in ranked] == ["alpha-skill", "beta-skill"]
    assert len(embedder.calls) == 3  # query + 2 skills
    assert all(dim == 8 for _, dim in embedder.calls)

    # Second call reuses every skill vector; only the query is embedded again.
    loader.search_embeddings("query: alpha", k=2)
    assert len(embedder.calls) == 4

    # A dimension change must invalidate the cached vectors.
    _patch_embedding(monkeypatch, embedder, dimension=16)
    loader.search_embeddings("query: alpha", k=2)
    assert len(embedder.calls) == 7
    assert all(dim == 16 for _, dim in embedder.calls[4:])


def test_bm25_search_never_touches_the_embedder(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    root.mkdir()
    monkeypatch.setattr(
        "dspytools.skills.loader._user_skills_dir", lambda: tmp_path / "none"
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_skill(root, "alpha-skill", "alpha alpha")
    monkeypatch.setattr(
        "dspytools.core._embedder.get_embedder",
        lambda: (_ for _ in ()).throw(AssertionError("embedder must stay opt-in")),
    )

    loader = SkillLoader(str(root))
    assert [s.name for s in loader.search("alpha")] == ["alpha-skill"]

    import click

    from dspytools.commands.skills import skills_search

    hybrid = next(p for p in skills_search.params if p.name == "hybrid")
    assert isinstance(hybrid, click.Option)
    assert hybrid.is_flag is True
    assert hybrid.default is False


# ── 9. graph write is best-effort and records a :Skill node ────────────────


def test_compile_records_skill_node_without_fabricating_score(skills_root, monkeypatch):
    recorded: dict = {}

    class _RecordingGraph:
        def record_program(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(
        "dspytools.graph.skill_graph.FalkorDBSkillGraph", _RecordingGraph
    )

    manager = SkillManager()
    manager.create_skill("graph-skill", "graph", "question -> answer")
    result = manager.compile_skill("graph-skill")

    assert result["status"] == "compiled"
    assert recorded["skill_name"] == "graph-skill"
    assert recorded["score"] is None
    assert recorded["program_path"] == str(skills_root / "graph-skill" / "program.json")


def test_graph_failure_never_fails_the_compile(skills_root, monkeypatch):
    class _BrokenGraph:
        def record_program(self, **kwargs):
            raise RuntimeError("falkordb unreachable")

    monkeypatch.setattr("dspytools.graph.skill_graph.FalkorDBSkillGraph", _BrokenGraph)

    manager = SkillManager()
    manager.create_skill("offline-skill", "offline", "question -> answer")
    result = manager.compile_skill("offline-skill")

    assert result["status"] == "compiled"
    assert (skills_root / "offline-skill" / "program.json").exists()


# ── 10. `skills show` resolves by identity, not BM25 rank ──────────────────


def test_skills_show_resolves_by_exact_name(skills_root):
    _write_skill(skills_root, "alpha", "unrelated words here")
    _write_skill(skills_root, "alphabet", "alpha alpha alpha")

    manager = SkillManager()
    assert manager.get("alpha") is not None
    assert manager.get("alpha").name == "alpha"
    assert manager.get("alph") is None

    from dspytools.commands.skills import skills_show

    result = CliRunner().invoke(skills_show, ["alpha"])
    assert result.exit_code == 0, result.output
    assert "unrelated words here" in result.output


# ── 11. importing the package loads no heavy driver ────────────────────────


def test_importing_skills_pulls_no_numpy_redis_or_falkordb():
    code = (
        "import sys\n"
        "import dspytools.skills\n"
        "heavy = [m for m in ('numpy', 'redis', 'falkordb') if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == ""
