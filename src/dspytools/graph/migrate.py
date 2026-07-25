"""Migration utilities for FalkorDB graph integration.

Imports existing JSON-based data into FalkorDB.
"""

from __future__ import annotations

from dspytools.config.settings import compiled_dir, config_dir
from dspytools.core._io import read_json


def migrate_skill_graph() -> dict:
    """Import existing skill_graph.json into FalkorDB.

    Returns migration statistics.
    """
    from dspytools.config.settings import skill_graph_path as _skill_graph_path
    from dspytools.graph.skill_graph import FalkorDBSkillGraph

    graph = FalkorDBSkillGraph()
    skill_graph_path = _skill_graph_path()

    if not skill_graph_path.exists():
        return {"migrated": 0, "message": "No skill_graph.json found"}

    data = read_json(skill_graph_path)
    edges = data.get("edges", {})
    migrated = 0

    for skill, dependents in edges.items():
        for dep in dependents:
            graph.add_dependency(skill, dep)
            migrated += 1

    return {"migrated": migrated, "skills": len(edges)}


def migrate_morphology() -> dict:
    """Import existing morphology.json into FalkorDB.

    Returns migration statistics.
    """
    from dspytools.graph.skill_graph import FalkorDBSkillGraph

    graph = FalkorDBSkillGraph()
    morph_path = config_dir() / "morphology.json"

    if not morph_path.exists():
        return {"migrated": 0, "message": "No morphology.json found"}

    data = read_json(morph_path)
    migrated = 0

    for profile, patterns in data.get("patterns", {}).items():
        for pattern_type, stats in patterns.items():
            count = stats.get("count", 0)
            success_count = stats.get("success_count", 0)
            for _ in range(count):
                graph.record_task_profile(profile, pattern_type, success_count > 0)
                migrated += 1

    return {"migrated": migrated, "profiles": len(data.get("patterns", {}))}


def migrate_program_registry() -> dict:
    """Import existing compiled/index.json into FalkorDB.

    Returns migration statistics.
    """
    from dspytools.graph.skill_graph import FalkorDBSkillGraph

    graph = FalkorDBSkillGraph()
    registry_path = compiled_dir() / "index.json"

    if not registry_path.exists():
        return {"migrated": 0, "message": "No compiled/index.json found"}

    data = read_json(registry_path)
    migrated = 0

    for run_id, metadata in data.get("runs", {}).items():
        optimizer = metadata.get("optimizer", "unknown")
        score = metadata.get("score", 0.0)
        parent_id = metadata.get("lineage", {}).get("parent_run")
        dataset_hash = metadata.get("lineage", {}).get("dataset_hash")

        graph.record_program(
            run_id=run_id,
            optimizer=optimizer,
            score=score,
            parent_id=parent_id,
            dataset_hash=dataset_hash,
        )
        migrated += 1

    return {"migrated": migrated, "programs": len(data.get("runs", {}))}


def migrate_all() -> dict:
    """Run all migrations.

    Returns combined migration statistics.
    """
    results = {
        "skill_graph": migrate_skill_graph(),
        "morphology": migrate_morphology(),
        "program_registry": migrate_program_registry(),
    }

    total = sum(r.get("migrated", 0) for r in results.values())
    results["total_migrated"] = total

    return results
