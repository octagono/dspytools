"""FalkorDB-backed memory manager for dspytools.

Provides persistent agent memory with entity extraction, deduplication,
and semantic search via FalkorDB native vector index (db.idx.vector.queryNodes)
+ graph entity/tag traversal.

No mem0 dependency (incompatible with v2). No Redis FT.SEARCH dependency
(FalkorDB container lacks RediSearch module).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from functools import lru_cache

import numpy as np

from dspytools.config.settings import embedder_dimension
from dspytools.core._embedder import get_embedder
from dspytools.core.errors import ServiceUnavailableError
from dspytools.graph.client import get_graph_client


class MemoryManager:
    """Singleton memory manager using FalkorDB graph + native vector index.

    Graph schema:
      (:Memory {id, content, hash, user_id, agent_id, run_id, embedding, created_at, updated_at})
      (:Entity {name, type}) -[:MENTIONED_IN]-> (:Memory)
      (:Memory)-[:TAGGED]-> (:Tag {name})

    Vector index on Memory.embedding provides O(log N) semantic search
    via db.idx.vector.queryNodes. Score = distance (lower = more similar).

    Provides: add, add_batch, search (semantic + graph), get, update, delete, dedup.
    """

    _instance: MemoryManager | None = None

    def __new__(cls) -> MemoryManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._graph = None
        self._embedder = None

    def _ensure_init(self) -> None:
        """Lazy-init graph and embedder clients. Creates vector index."""
        if self._graph is None:
            client = get_graph_client()
            self._graph = client.graph("memories")
            self._embedder = get_embedder()
            # Create vector index on Memory nodes (idempotent)
            dim = embedder_dimension()
            client.ensure_vector_index("memories", "Memory", "embedding", dim)

    def _embed(self, text: str) -> list[float] | None:
        """Get embedding vector via shared embedder. Fail-fast on embedder failure."""
        if self._embedder is None:
            raise ServiceUnavailableError("memory", "Embedder not initialized")
        vec = self._embedder(text)
        return np.array(vec).tolist()

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(content.strip().lower().encode()).hexdigest()[:16]

    @staticmethod
    def _extract_entities(content: str) -> list[tuple[str, str]]:
        words = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
        entities = []
        seen = set()
        for w in words:
            key = w.lower()
            if key not in seen:
                seen.add(key)
                etype = "concept"
                if w.isupper():
                    etype = "acronym"
                elif w.endswith(("tion", "ment", "ity", "ness")):
                    etype = "abstract"
                entities.append((w, etype))
        return entities

    @staticmethod
    def _extract_tags(content: str) -> list[str]:
        words = re.findall(r"\b[a-z]{4,}\b", content.lower())
        stops = {
            "this",
            "that",
            "with",
            "from",
            "have",
            "been",
            "were",
            "they",
            "their",
            "which",
            "about",
            "would",
            "could",
            "should",
            "there",
            "what",
            "when",
            "where",
            "will",
            "does",
            "also",
            "than",
            "into",
            "some",
            "more",
            "other",
            "each",
            "very",
            "just",
        }
        return list(dict.fromkeys(w for w in words if w not in stops))[:10]

    def add(
        self,
        content: str,
        user_id: str = "dspytools",
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Add a memory with deduplication, entity extraction, and embedding."""
        return self._add_single(content, user_id, agent_id, run_id, metadata)

    def add_batch(
        self,
        entries: list[dict],
        user_id: str = "dspytools",
    ) -> list[dict]:
        """Add multiple memories in batch.

        Each entry supports: content, agent_id, run_id, metadata.
        Uses shared embedder once per unique content hash.
        """
        results = []
        for entry in entries:
            result = self._add_single(
                content=entry["content"],
                user_id=user_id,
                agent_id=entry.get("agent_id"),
                run_id=entry.get("run_id"),
                metadata=entry.get("metadata"),
            )
            results.append(result)
        return results

    def _add_single(
        self,
        content: str,
        user_id: str,
        agent_id: str | None,
        run_id: str | None,
        metadata: dict | None,
    ) -> dict:
        """Internal single-memory add with dedup + entity + vector."""
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        content_hash = self._content_hash(content)

        # Dedup
        dedup = self._graph.query(
            "MATCH (m:Memory {hash: $hash, user_id: $user_id}) RETURN m.id AS id LIMIT 1",
            {"hash": content_hash, "user_id": user_id},
        )
        if dedup.result_set:
            return {
                "id": dedup.result_set[0][0],
                "deduplicated": True,
                "content": content,
            }

        memory_id = str(uuid.uuid4())[:12]
        now = time.time()

        # Create Memory node with embedding via vecf32()
        vec = self._embed(content)
        if vec is not None:
            self._graph.query(
                """CREATE (m:Memory {
                    id: $id, content: $content, hash: $hash,
                    user_id: $user_id, agent_id: $agent_id, run_id: $run_id,
                    metadata: $metadata, created_at: $created_at, updated_at: $updated_at,
                    embedding: vecf32($vec)
                })""",
                {
                    "id": memory_id,
                    "content": content,
                    "hash": content_hash,
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "run_id": run_id or "",
                    "metadata": json.dumps(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                    "vec": vec,
                },
            )
        else:
            self._graph.query(
                """CREATE (m:Memory {
                    id: $id, content: $content, hash: $hash,
                    user_id: $user_id, agent_id: $agent_id, run_id: $run_id,
                    metadata: $metadata, created_at: $created_at, updated_at: $updated_at
                })""",
                {
                    "id": memory_id,
                    "content": content,
                    "hash": content_hash,
                    "user_id": user_id,
                    "agent_id": agent_id or "",
                    "run_id": run_id or "",
                    "metadata": json.dumps(metadata or {}),
                    "created_at": now,
                    "updated_at": now,
                },
            )

        # Extract entities and tags once (avoid double extraction)
        entities = self._extract_entities(content)
        tags = self._extract_tags(content)

        # Batch-link entities via UNWIND (1 query instead of N)
        if entities:
            self._graph.query(
                """UNWIND $entities AS ent
                   MERGE (e:Entity {name: ent.name, type: ent.type})
                   WITH e MATCH (m:Memory {id: $memory_id})
                   MERGE (e)-[:MENTIONED_IN]->(m)""",
                {
                    "entities": [{"name": n, "type": t} for n, t in entities],
                    "memory_id": memory_id,
                },
            )

        # Batch-link tags via UNWIND (1 query instead of N)
        if tags:
            self._graph.query(
                """UNWIND $tags AS tag_name
                   MERGE (t:Tag {name: tag_name})
                   WITH t MATCH (m:Memory {id: $memory_id})
                   MERGE (m)-[:TAGGED]->(t)""",
                {"tags": tags, "memory_id": memory_id},
            )

        return {
            "id": memory_id,
            "deduplicated": False,
            "content": content,
            "entities": [e[0] for e in entities],
            "tags": tags,
        }

    def search(
        self,
        query: str,
        user_id: str = "dspytools",
        agent_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search memories by semantic similarity + graph traversal.

        Tier 1: FalkorDB vector index — O(log N) top-k by cosine distance.
        Tier 2: Graph entity/tag traversal — fills remaining slots.
        """
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        results: list[dict] = []
        seen_ids: set[str] = set()

        # Tier 1: FalkorDB native vector similarity
        query_vec = self._embed(query)
        if query_vec is not None:
            vec_results = self._graph.query(
                """CALL db.idx.vector.queryNodes('Memory', 'embedding', $k, vecf32($vec))
                   YIELD node, score
                   WHERE node.user_id = $user_id
                   RETURN node.id, node.content, node.created_at, node.metadata, score
                   ORDER BY score ASC""",
                {"k": limit, "vec": query_vec, "user_id": user_id},
            )
            for row in vec_results.result_set:
                mid = row[0]
                if mid not in seen_ids:
                    seen_ids.add(mid)
                    distance = row[4]
                    results.append(
                        {
                            "id": mid,
                            "memory": row[1],
                            "user_id": user_id,
                            "created_at": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {},
                            "score": round(1.0 - distance, 4),
                            "source": "semantic",
                        }
                    )

        # Tier 2: Graph entity/tag traversal for related memories
        if len(results) < limit:
            graph_results = self._graph.query(
                """MATCH (m:Memory {user_id: $user_id})
                   WHERE NOT m.id IN $seen_ids
                   OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(m)
                   OPTIONAL MATCH (m)-[:TAGGED]->(t:Tag)
                   RETURN m.id AS id, m.content AS content,
                          m.created_at AS created_at, m.metadata AS metadata,
                          collect(DISTINCT e.name) AS entities,
                          collect(DISTINCT t.name) AS tags
                   ORDER BY m.created_at DESC
                   LIMIT $limit""",
                {
                    "user_id": user_id,
                    "seen_ids": list(seen_ids),
                    "limit": limit - len(results),
                },
            )
            for row in graph_results.result_set:
                if row[0] not in seen_ids:
                    seen_ids.add(row[0])
                    results.append(
                        {
                            "id": row[0],
                            "memory": row[1],
                            "user_id": user_id,
                            "created_at": row[2],
                            "metadata": json.loads(row[3]) if row[3] else {},
                            "entities": row[4],
                            "tags": row[5],
                            "score": 0.5,
                            "source": "graph",
                        }
                    )

        return results[:limit]

    def get_all(
        self,
        user_id: str = "dspytools",
        agent_id: str | None = None,
    ) -> list[dict]:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        result = self._graph.query(
            "MATCH (m:Memory {user_id: $user_id}) RETURN m.id, m.content, m.created_at, m.metadata ORDER BY m.created_at DESC",
            {"user_id": user_id},
        )
        return [
            {
                "id": r[0],
                "memory": r[1],
                "user_id": user_id,
                "created_at": r[2],
                "metadata": json.loads(r[3]) if r[3] else {},
            }
            for r in result.result_set
        ]

    def get(self, memory_id: str) -> dict | None:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        result = self._graph.query(
            "MATCH (m:Memory {id: $id}) RETURN m.id, m.content, m.user_id, m.created_at, m.metadata",
            {"id": memory_id},
        )
        if result.result_set:
            r = result.result_set[0]
            return {
                "id": r[0],
                "memory": r[1],
                "user_id": r[2],
                "created_at": r[3],
                "metadata": json.loads(r[4]) if r[4] else {},
            }
        return None

    def update(self, memory_id: str, content: str) -> dict:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        now = time.time()
        new_hash = self._content_hash(content)

        self._graph.query(
            "MATCH (m:Memory {id: $id}) SET m.content = $content, m.hash = $hash, m.updated_at = $updated_at",
            {"id": memory_id, "content": content, "hash": new_hash, "updated_at": now},
        )

        # Update embedding on the node
        vec = self._embed(content)
        if vec is not None:
            self._graph.query(
                "MATCH (m:Memory {id: $id}) SET m.embedding = vecf32($vec)",
                {"id": memory_id, "vec": vec},
            )

        # Re-extract entities and tags — delete old edges, link new ones
        self._graph.query(
            "MATCH (m:Memory {id: $id})<-[r:MENTIONED_IN]-(e:Entity) DELETE r",
            {"id": memory_id},
        )
        entities = self._extract_entities(content)
        if entities:
            self._graph.query(
                """UNWIND $entities AS ent
                   MERGE (e:Entity {name: ent.name, type: ent.type})
                   WITH e MATCH (m:Memory {id: $memory_id})
                   MERGE (e)-[:MENTIONED_IN]->(m)""",
                {
                    "entities": [{"name": n, "type": t} for n, t in entities],
                    "memory_id": memory_id,
                },
            )

        # Re-link tags
        self._graph.query(
            "MATCH (m:Memory {id: $id})-[r:TAGGED]->(t:Tag) DELETE r",
            {"id": memory_id},
        )
        tags = self._extract_tags(content)
        if tags:
            self._graph.query(
                """UNWIND $tags AS tag
                   MERGE (t:Tag {name: tag})
                   WITH t MATCH (m:Memory {id: $memory_id})
                   MERGE (m)-[:TAGGED]->(t)""",
                {"tags": tags, "memory_id": memory_id},
            )

        return {"id": memory_id, "updated": True}

    def delete(self, memory_id: str) -> dict:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        self._graph.query(
            "MATCH (m:Memory {id: $id}) DETACH DELETE m", {"id": memory_id}
        )
        return {"id": memory_id, "deleted": True}

    def delete_all(
        self, user_id: str = "dspytools", agent_id: str | None = None
    ) -> dict:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        result = self._graph.query(
            "MATCH (m:Memory {user_id: $user_id}) RETURN count(m)", {"user_id": user_id}
        )
        count = result.result_set[0][0] if result.result_set else 0
        self._graph.query(
            "MATCH (m:Memory {user_id: $user_id}) DETACH DELETE m", {"user_id": user_id}
        )
        return {"deleted": count}

    def history(self, memory_id: str) -> list[dict]:
        mem = self.get(memory_id)
        return [mem] if mem else []

    def reset(self) -> None:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        self._graph.query("MATCH (m:Memory) DETACH DELETE m")
        self._graph.query("MATCH (e:Entity) DETACH DELETE e")
        self._graph.query("MATCH (t:Tag) DETACH DELETE t")

    def stats(self, user_id: str = "dspytools") -> dict:
        self._ensure_init()
        if self._graph is None:
            raise ServiceUnavailableError("memory", "FalkorDB not initialized")
        # Single query for all counts (was 3 separate queries)
        result = self._graph.query(
            """MATCH (m:Memory {user_id: $user_id})
               OPTIONAL MATCH (e:Entity)-[:MENTIONED_IN]->(m)
               OPTIONAL MATCH (m)-[:TAGGED]->(t:Tag)
               RETURN count(DISTINCT m) AS mem, count(DISTINCT e) AS ent, count(DISTINCT t) AS tag""",
            {"user_id": user_id},
        )
        row = result.result_set[0] if result.result_set else [0, 0, 0]
        return {
            "user_id": user_id,
            "total_memories": row[0],
            "total_entities": row[1],
            "total_tags": row[2],
        }


@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """Get cached MemoryManager singleton."""
    return MemoryManager()
