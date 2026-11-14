"""Skills system — BM25-indexed skill library with auto-generation.

Usage:
    from dspytools.skills import SkillManager
    mgr = SkillManager()
    mgr.create_skill("my-skill", "description", "question -> answer")
    mgr.compile_skill("my-skill", trainset)
    mgr.auto_optimize_skill("my-skill")
"""

from dspytools.skills.discovery import (
    ExternalSkill,
    list_categories,
    popular_skills,
    search_external,
)
from dspytools.skills.loader import Skill, SkillLoader
from dspytools.skills.manager import SkillManager, load_skill_program

__all__ = [
    "ExternalSkill",
    "Skill",
    "SkillLoader",
    "SkillManager",
    "list_categories",
    "load_skill_program",
    "popular_skills",
    "search_external",
]
