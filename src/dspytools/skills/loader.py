"""Skill loader — BM25-indexed skill library.

Follows harness-so pattern: skills are directories with SKILL.md files.
Supports both file-based skills and programmatic skills from evolution.
"""

from __future__ import annotations

import math
import re
import time as _time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dspytools.config.settings import embedder_kwargs
from dspytools.config.settings import skills_dir as _user_skills_dir
from dspytools.core._dspy import dspy
from dspytools.skills.bm25_mojo_bridge import score_documents as _bm25_score

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class Skill:
    """A reusable DSPy skill — documentation + compiled program."""

    name: str
    description: str = ""
    path: Path | None = None
    frontmatter: dict = field(default_factory=dict)
    body: str = ""
    program_path: str | None = None

    @property
    def has_program(self) -> bool:
        return self.program_path is not None and Path(self.program_path).exists()


class SkillLoader:
    """BM25-indexed skill library with directory loading.

    Optimization 11: Caches loaded skills + BM25 index with mtime check.
    Only re-scans directories when SKILL.md files change.

    Loads skills from:
      - skills/ directory (project-local)
      - <config_dir>/skills/ (user-wide, default ~/.config/dspytools/skills)
      - Programmatic skills from evolution
    """

    def __init__(self, skills_dir: str | Path | None = None):
        self._paths = [
            Path(skills_dir) if skills_dir else Path("skills"),
            _user_skills_dir(),
            Path.home() / ".agents" / "skills",
        ]
        self.skills: dict[str, Skill] = {}
        self._bm25_index: dict[str, dict[str, Any]] = {}
        # Optimization 11: mtime-based cache
        self._loaded: bool = False
        self._load_time: float = 0
        self._skills_mtime: float = 0  # latest mtime across all SKILL.md files
        # Optimization: cached skill embeddings (invalidated when skills reload)
        self._embeddings_cache: dict[str, np.ndarray] = {}
        self._embeddings_mtime: float = -1

    def _get_skills_mtime(self) -> float:
        """Get the latest mtime across all SKILL.md files."""
        latest = 0.0
        for base in self._paths:
            if base.exists() and base.is_dir():
                for skill_dir in base.iterdir():
                    if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                        md = skill_dir / "SKILL.md"
                        if md.exists():
                            latest = max(latest, md.stat().st_mtime)
        return latest

    def load_all(self) -> list[Skill]:
        """Load all skills from configured directories (cached with mtime check).

        Optimization 25: Skips mtime scan entirely within 5s TTL window.
        Only scans directories when the TTL has expired.
        """
        now = _time.time()

        # Optimization 25: Fast path — skip mtime scan if within TTL
        if self._loaded and (now - self._load_time) < 5.0:
            return list(self.skills.values())

        skills_mtime = self._get_skills_mtime()

        # Return cached if no SKILL.md files changed
        if self._loaded and skills_mtime == self._skills_mtime:
            self._load_time = now  # Reset TTL
            return list(self.skills.values())

        self.skills = {}
        for base in self._paths:
            if base.exists() and base.is_dir():
                for skill_dir in sorted(base.iterdir()):
                    if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                        skill = self._load_skill_dir(skill_dir)
                        # First-loaded (project-local) takes priority over user-wide
                        if skill and skill.name not in self.skills:
                            self.skills[skill.name] = skill

        self._build_index()
        self._loaded = True
        self._load_time = now
        self._skills_mtime = skills_mtime
        return list(self.skills.values())

    def _load_skill_dir(self, skill_dir: Path) -> Skill | None:
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            return None

        text = md_path.read_text()
        fm, body = _parse_frontmatter(text)

        prog_path = skill_dir / "program.json"
        return Skill(
            name=fm.get("name", skill_dir.name),
            description=fm.get("description", ""),
            path=skill_dir,
            frontmatter=fm,
            body=body,
            program_path=str(prog_path) if prog_path.exists() else None,
        )

    def _build_index(self) -> None:
        """Build BM25 inverted index over skill names + descriptions + bodies."""
        self._bm25_index = {}
        for name, skill in self.skills.items():
            text = f"{skill.name} {skill.description} {skill.body}"
            tokens = _tokenize(text)
            self._bm25_index[name] = {
                "tokens": tokens,
                "tf": Counter(tokens),
                "len": len(tokens),
            }

    def search(self, query: str, k: int = 5) -> list[Skill]:
        """BM25 search over skills. Returns top-k matching skills.

        Uses Mojo-accelerated BM25 scoring when available (Phase 3).
        Falls back to pure Python BM25 otherwise.
        """
        query_tokens = _tokenize(query)
        if not self._bm25_index or not query_tokens:
            return list(self.skills.values())[:k]

        n_docs = len(self._bm25_index)
        avg_len = sum(v["len"] for v in self._bm25_index.values()) / max(n_docs, 1)
        k1, b = 1.2, 0.75

        # Precompute IDF for each query token
        idf_values = []
        for token in query_tokens:
            df = _count_docs_with_token(self._bm25_index, token)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            idf_values.append(idf)

        # Build flattened TF matrix: shape (n_docs, n_terms)
        # doc_lens array: shape (n_docs,)
        doc_names = list(self._bm25_index.keys())
        tf_matrix = [
            [float(self._bm25_index[name]["tf"].get(tok, 0)) for tok in query_tokens]
            for name in doc_names
        ]
        doc_lengths = [float(self._bm25_index[name]["len"]) for name in doc_names]

        # Delegate score computation to Mojo bridge (or pure Python fallback)

        score_arr = _bm25_score(tf_matrix, idf_values, doc_lengths, avg_len, k1=k1, b=b)

        # Build scores dict from array — only positive scores matter
        scores = {}
        for i, name in enumerate(doc_names):
            s = float(score_arr[i])
            if s > 0:
                scores[name] = s

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.skills[name] for name, _ in ranked[:k]]

    def search_embeddings(self, query: str, k: int = 5) -> list[Skill]:
        """Embedding-based semantic search (fallback to BM25)."""

        embedder = dspy.Embedder(**embedder_kwargs())
        self.load_all()
        if not self.skills:
            return []

        # Invalidate embedding cache if skills changed
        if self._skills_mtime != self._embeddings_mtime:
            self._embeddings_cache.clear()
            self._embeddings_mtime = self._skills_mtime

        # Get query embedding
        query_vec = np.array(embedder(query))

        # Score each skill by cosine similarity (cache skill embeddings)
        scores = {}
        for name, skill in self.skills.items():
            if name not in self._embeddings_cache:
                text = f"{skill.name}: {skill.description}. {skill.body[:500]}"
                self._embeddings_cache[name] = np.array(embedder(text))
            text_vec = self._embeddings_cache[name]
            similarity = np.dot(query_vec, text_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(text_vec) + 1e-8
            )
            scores[name] = float(similarity)

        # Hybrid: combine embedding similarity + BM25
        bm25_results = {
            s.name: i for i, s in enumerate(self.search(query, k=len(self.skills)))
        }
        for name in scores:
            if name in bm25_results:
                # BM25 rank bonus (higher rank = lower index = higher bonus)
                scores[name] += 0.3 / (bm25_results[name] + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.skills[name] for name, _ in ranked[:k]]

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw = match.group(1)
    body = text[match.end() :]
    fm: dict[str, Any] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.startswith("[") and value.endswith("]"):
                fm[key] = [
                    v.strip().strip('"').strip("'") for v in value[1:-1].split(",")
                ]
            else:
                fm[key] = value
    return fm, body


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


def _count_docs_with_token(index: dict, token: str) -> int:
    return sum(1 for v in index.values() if v["tf"].get(token, 0) > 0)
