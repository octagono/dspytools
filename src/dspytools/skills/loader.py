"""Skill loader — BM25-indexed skill library.

Follows harness-so pattern: skills are directories with SKILL.md files.
Supports both file-based skills and programmatic skills from evolution.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dspytools.config.settings import skills_dir as _user_skills_dir
from dspytools.core.logging_config import get_logger

_log = get_logger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Reciprocal rank fusion constant (Cormack et al.); keeps embedding and BM25
# ranks comparable without normalizing two unlike score scales.
_RRF_K = 60


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

    The loaded skills, the BM25 index and the embedding cache live behind one
    ``threading.RLock``. MCP can issue concurrent tool calls, so a reload builds
    fresh structures and swaps them in rather than mutating live state.

    The cache key is a hash of every skill directory's ``(name, SKILL.md
    mtime_ns)`` plus its ``program.json`` presence/mtime. An mtime watermark on
    ``SKILL.md`` alone could not see a deletion or a rename, so a removed skill
    stayed searchable forever.

    Loads skills from:
      - skills/ directory (project-local)
      - <config_dir>/skills/ (user-wide, default ~/.config/dspytools/skills)
      - ~/.agents/skills/ (agent-ecosystem skills)
    """

    def __init__(self, skills_dir: str | Path | None = None):
        self._paths = [
            Path(skills_dir) if skills_dir else Path("skills"),
            _user_skills_dir(),
            Path.home() / ".agents" / "skills",
        ]
        self.skills: dict[str, Skill] = {}
        self._bm25_index: dict[str, dict[str, Any]] = {}
        self._loaded: bool = False
        self._cache_signature: str | None = None
        # Cached skill embeddings, keyed by (model, dimension, skill name) so a
        # changed embedding model or dimension never reuses stale vectors.
        self._embeddings_cache: dict[tuple[str, int, str], Any] = {}
        self._embeddings_signature: str | None = None
        self._lock = threading.RLock()

    def _index_signature(self) -> str:
        """Hash the skill-directory set, not just the newest mtime.

        Covers additions, deletions, renames, ``SKILL.md`` edits and
        ``program.json`` appearing or being recompiled.
        """
        entries: list[tuple[str, ...]] = []
        for base in self._paths:
            try:
                if not (base.exists() and base.is_dir()):
                    continue
                skill_dirs = sorted(base.iterdir())
            except OSError:
                continue
            for skill_dir in skill_dirs:
                if not skill_dir.is_dir() or skill_dir.name.startswith("."):
                    continue
                md = skill_dir / "SKILL.md"
                try:
                    if not md.exists():
                        continue
                    md_mtime = md.stat().st_mtime_ns
                except OSError:
                    continue
                prog = skill_dir / "program.json"
                try:
                    prog_key = f"{prog.stat().st_mtime_ns}" if prog.exists() else "-"
                except OSError:
                    prog_key = "-"
                entries.append((str(base), skill_dir.name, str(md_mtime), prog_key))
        payload = "\n".join("|".join(entry) for entry in sorted(entries))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load_all(self) -> list[Skill]:
        """Load all skills from configured directories (signature-cached)."""
        with self._lock:
            signature = self._index_signature()
            if self._loaded and signature == self._cache_signature:
                return list(self.skills.values())

            skills: dict[str, Skill] = {}
            for base in self._paths:
                if not (base.exists() and base.is_dir()):
                    continue
                for skill_dir in sorted(base.iterdir()):
                    if skill_dir.is_dir() and not skill_dir.name.startswith("."):
                        skill = self._load_skill_dir(skill_dir)
                        # First-loaded (project-local) wins over user-wide
                        if skill and skill.name not in skills:
                            skills[skill.name] = skill

            index = _build_bm25_index(skills)
            # Swap both structures together under the lock.
            self.skills = skills
            self._bm25_index = index
            # A changed index invalidates every cached skill embedding.
            self._embeddings_cache.clear()
            self._embeddings_signature = None
            self._loaded = True
            self._cache_signature = signature
            return list(skills.values())

    def _load_skill_dir(self, skill_dir: Path) -> Skill | None:
        md_path = skill_dir / "SKILL.md"
        if not md_path.exists():
            return None

        text = md_path.read_text()
        fm, body = _parse_frontmatter(text)

        prog_path = skill_dir / "program.json"
        program_path: str | None = None
        if prog_path.exists():
            # A program saved by another DSPy silently degrades under 3.5's
            # prompts/adapters — treat it as absent so callers recompile.
            from dspytools.core.programs import (
                is_current_dspy_version,
                saved_dspy_version,
            )

            if is_current_dspy_version(prog_path):
                program_path = str(prog_path)
            else:
                _log.warning(
                    "skill_program_dspy_version_mismatch",
                    skill=skill_dir.name,
                    saved_version=saved_dspy_version(prog_path),
                    hint="recompile the skill program with `dspytools skills compile`",
                )
        return Skill(
            name=fm.get("name", skill_dir.name),
            description=fm.get("description", ""),
            path=skill_dir,
            frontmatter=fm,
            body=body,
            program_path=program_path,
        )

    def search(self, query: str, k: int = 5) -> list[Skill]:
        """BM25 search over skills. Returns top-k matching skills.

        Uses Mojo-accelerated BM25 scoring when available (Phase 3).
        Falls back to pure Python BM25 otherwise.
        """
        self.load_all()
        with self._lock:
            query_tokens = _tokenize(query)
            if not self._bm25_index or not query_tokens:
                return list(self.skills.values())[:k]

            ranked = _bm25_ranked(self._bm25_index, query_tokens)
            return [self.skills[name] for name, _ in ranked[:k]]

    def search_embeddings(self, query: str, k: int = 5) -> list[Skill]:
        """Hybrid semantic search: embedding cosine + BM25 via reciprocal rank fusion.

        Opt-in (`skills search --hybrid`); BM25 remains the default. Falls back
        to plain BM25 when the embedding server is unreachable, so a hybrid
        search never fails a caller over an optional backend.
        """
        self.load_all()
        with self._lock:
            skills = dict(self.skills)
            index = dict(self._bm25_index)
        if not skills:
            return []

        try:
            # Lazy: numpy and the embedder must stay off the CLI start-up path.
            import numpy as np

            from dspytools.config.settings import embedder_dimension, embedder_kwargs
            from dspytools.core._embedder import get_embedder

            kwargs = embedder_kwargs()
            model = str(kwargs.get("model", ""))
            dimension = embedder_dimension()
            embedder = get_embedder()

            def embed(text: str):
                return np.asarray(embedder(text, dimensions=dimension), dtype=float)

            query_vec = embed(query)
            query_norm = np.linalg.norm(query_vec)

            keys = {name: (model, dimension, name) for name in skills}
            with self._lock:
                if self._cache_signature != self._embeddings_signature:
                    self._embeddings_cache.clear()
                    self._embeddings_signature = self._cache_signature
                vectors = {
                    name: self._embeddings_cache[key]
                    for name, key in keys.items()
                    if key in self._embeddings_cache
                }

            # Embed outside the lock: a slow embedding server must not block
            # concurrent BM25 readers.
            for name, skill in skills.items():
                if name in vectors:
                    continue
                text = f"{skill.name}: {skill.description}. {skill.body[:500]}"
                vectors[name] = embed(text)
                with self._lock:
                    self._embeddings_cache[keys[name]] = vectors[name]

            similarities = {
                name: float(
                    np.dot(query_vec, vector)
                    / (query_norm * np.linalg.norm(vector) + 1e-8)
                )
                for name, vector in vectors.items()
            }
        except Exception as exc:  # optional backend — degrade, never fail
            _log.warning("skill_embeddings_unavailable", error=str(exc))
            return self.search(query, k)

        emb_rank = {
            name: rank
            for rank, (name, _) in enumerate(
                sorted(similarities.items(), key=lambda kv: kv[1], reverse=True)
            )
        }
        bm25_rank = {
            name: rank
            for rank, (name, _) in enumerate(_bm25_ranked(index, _tokenize(query)))
        }

        # Reciprocal rank fusion: comparable regardless of each scorer's scale.
        fused: dict[str, float] = {}
        for name in skills:
            score = 0.0
            if name in emb_rank:
                score += 1.0 / (_RRF_K + emb_rank[name])
            if name in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[name])
            fused[name] = score

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        return [skills[name] for name, _ in ranked[:k]]

    def get(self, name: str) -> Skill | None:
        """Exact-identity lookup — never a ranked match."""
        self.load_all()
        with self._lock:
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
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                fm[key] = [_unquote_scalar(v.strip()) for v in value[1:-1].split(",")]
            else:
                fm[key] = _unquote_scalar(value)
    return fm, body


def _unquote_scalar(value: str) -> str:
    """Decode a JSON-quoted frontmatter scalar, falling back to quote-stripping.

    ``create_skill`` JSON-encodes every scalar because that is a valid YAML
    subset and quotes any ``": "`` / ``" #"`` that would otherwise change the
    document's meaning. Decoding with ``json`` keeps those round-trips exact.
    """
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            decoded = json.loads(value)
        except ValueError:
            return value[1:-1]
        return decoded if isinstance(decoded, str) else value[1:-1]
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1].replace("''", "'")
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_-]+", text.lower())


def _count_docs_with_token(index: dict, token: str) -> int:
    return sum(1 for v in index.values() if v["tf"].get(token, 0) > 0)


def _build_bm25_index(skills: dict[str, Skill]) -> dict[str, dict[str, Any]]:
    """Build the BM25 inverted index over skill names + descriptions + bodies."""
    index: dict[str, dict[str, Any]] = {}
    for name, skill in skills.items():
        text = f"{skill.name} {skill.description} {skill.body}"
        tokens = _tokenize(text)
        index[name] = {
            "tokens": tokens,
            "tf": Counter(tokens),
            "len": len(tokens),
        }
    return index


def _bm25_ranked(
    index: dict[str, dict[str, Any]], query_tokens: list[str]
) -> list[tuple[str, float]]:
    """Score every document and return ``(name, score)`` for positive scores."""
    if not index or not query_tokens:
        return []

    n_docs = len(index)
    avg_len = sum(v["len"] for v in index.values()) / max(n_docs, 1)
    k1, b = 1.2, 0.75

    idf_values = []
    for token in query_tokens:
        df = _count_docs_with_token(index, token)
        idf_values.append(math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0))

    # Flattened TF matrix: shape (n_docs, n_terms), plus per-doc lengths.
    doc_names = list(index.keys())
    tf_matrix = [
        [float(index[name]["tf"].get(tok, 0)) for tok in query_tokens]
        for name in doc_names
    ]
    doc_lengths = [float(index[name]["len"]) for name in doc_names]

    # Delegate score computation to the Mojo bridge (or its Python fallback).
    # Imported here: the bridge pulls numpy, which must stay off the CLI
    # start-up path (`import dspytools.skills` loads no numpy/redis/falkordb).
    from dspytools.skills.bm25_mojo_bridge import score_documents as bm25_score

    score_arr = bm25_score(tf_matrix, idf_values, doc_lengths, avg_len, k1=k1, b=b)

    scored = [(name, float(score_arr[i])) for i, name in enumerate(doc_names)]
    scored = [(name, score) for name, score in scored if score > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored
