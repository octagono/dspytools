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
from dspytools.skills.manager import SkillManager

__all__ = [
    "Skill",
    "SkillLoader",
    "SkillManager",
    "ExternalSkill",
    "search_external",
    "popular_skills",
    "list_categories",
]
