"""External skill discovery — search skills.sh ecosystem.

Integrates with the open agent skills ecosystem:
  - skills.sh leaderboard: https://skills.sh/
  - npx skills find <query>: CLI package manager
  - GitHub source: vercel-labs/agent-skills, anthropics/skills, etc.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.error as _urlerror
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class ExternalSkill:
    """A skill from the open agent skills ecosystem."""

    name: str
    source: str  # e.g., "vercel-labs/agent-skills"
    description: str = ""
    installs: int = 0
    category: str = ""
    url: str = ""

    @property
    def install_command(self) -> str:
        return f"npx skills add {self.source}@{self.name}"

    @property
    def browse_url(self) -> str:
        return self.url or f"https://skills.sh/{self.source}/{self.name}"


# ── Curated popular skills dataset ──────────────────────────────────────

_POPULAR_SKILLS: list[dict] = [
    {
        "name": "react-best-practices",
        "source": "vercel-labs/agent-skills",
        "description": "React and Next.js performance optimization guidelines",
        "installs": 185000,
        "category": "web-development",
        "url": "https://skills.sh/vercel-labs/agent-skills/react-best-practices",
    },
    {
        "name": "next-best-practices",
        "source": "next-skills",
        "description": "Next.js best practices — file conventions, RSC boundaries",
        "installs": 120000,
        "category": "web-development",
    },
    {
        "name": "next-cache-components",
        "source": "next-skills",
        "description": "Next.js 16 Cache Components — PPR, use cache directive",
        "installs": 95000,
        "category": "web-development",
    },
    {
        "name": "next-upgrade",
        "source": "next-skills",
        "description": "Upgrade Next.js following official migration guides and codemods",
        "installs": 85000,
        "category": "web-development",
    },
    {
        "name": "frontend-design",
        "source": "anthropics/skills",
        "description": "Create distinctive, production-grade frontend interfaces",
        "installs": 140000,
        "category": "design",
    },
    {
        "name": "e2e-setup",
        "source": "opencode",
        "description": "Set up e2e test suite — real flows, layered assertions, video+trace",
        "installs": 45000,
        "category": "testing",
    },
    {
        "name": "webapp-testing",
        "source": "opencode",
        "description": "Test local web applications with Playwright — verify, debug UI",
        "installs": 52000,
        "category": "testing",
    },
    {
        "name": "crabbox-setup",
        "source": "opencode",
        "description": "Scaffold isolated cloud dev boxes per agent for parallel-safe testing",
        "installs": 15000,
        "category": "devops",
    },
    {
        "name": "dev-local-setup",
        "source": "opencode",
        "description": "Scaffold one-command dev-local launcher for any codebase",
        "installs": 18000,
        "category": "devops",
    },
    {
        "name": "setup-codebase-harness",
        "source": "opencode",
        "description": "Set up full agent harness — legible, executable, verifiable repo",
        "installs": 22000,
        "category": "devops",
    },
    {
        "name": "dspy",
        "source": "opencode",
        "description": "Build and optimize LM programs using DSPy 3.2",
        "installs": 35000,
        "category": "ai-ml",
    },
    {
        "name": "vllm",
        "source": "opencode",
        "description": "vLLM high-throughput LLM serving engine — OpenAI-compatible API",
        "installs": 28000,
        "category": "ai-ml",
    },
    {
        "name": "mlflow",
        "source": "opencode",
        "description": "MLflow AI engineering — experiment tracking, model registry",
        "installs": 19000,
        "category": "ai-ml",
    },
    {
        "name": "dapr-agents",
        "source": "opencode",
        "description": "Build production-grade resilient AI agent systems with Dapr",
        "installs": 12000,
        "category": "ai-ml",
    },
    {
        "name": "livekit-agents",
        "source": "opencode",
        "description": "Build production-grade realtime voice AI agents with LiveKit",
        "installs": 14000,
        "category": "ai-ml",
    },
    {
        "name": "mem0",
        "source": "opencode",
        "description": "Mem0 memory layer for AI — long-term, conversational, agent memory",
        "installs": 16000,
        "category": "ai-ml",
    },
    {
        "name": "mcp-builder",
        "source": "opencode",
        "description": "Guide for creating high-quality MCP servers for LLM interaction",
        "installs": 31000,
        "category": "tools",
    },
    {
        "name": "pr",
        "source": "opencode",
        "description": "Prove the feature you just built works — verify then open a PR",
        "installs": 24000,
        "category": "productivity",
    },
    {
        "name": "skill-creator",
        "source": "opencode",
        "description": "Create new skills, modify and improve existing skills",
        "installs": 17000,
        "category": "productivity",
    },
    {
        "name": "find-skills",
        "source": "opencode",
        "description": "Discover and install agent skills from the open ecosystem",
        "installs": 13000,
        "category": "productivity",
    },
    {
        "name": "esp-claw",
        "source": "opencode",
        "description": "Build AI-powered edge agents on ESP32 microcontrollers",
        "installs": 8000,
        "category": "iot",
    },
    {
        "name": "walter",
        "source": "opencode",
        "description": "Walter IoT module for ESP32-S3 with cellular connectivity",
        "installs": 5000,
        "category": "iot",
    },
    {
        "name": "livekit-python",
        "source": "opencode",
        "description": "Build realtime audio/video apps in Python with LiveKit",
        "installs": 11000,
        "category": "ai-ml",
    },
    {
        "name": "livekit-typescript",
        "source": "opencode",
        "description": "LiveKit TypeScript SDK — real-time audio/video",
        "installs": 10000,
        "category": "ai-ml",
    },
    {
        "name": "livekit-esp32",
        "source": "opencode",
        "description": "Build realtime audio/video and AI agent apps on ESP32",
        "installs": 6000,
        "category": "iot",
    },
    {
        "name": "a2ui",
        "source": "opencode",
        "description": "Build agent-generated user interfaces with A2UI standard",
        "installs": 9000,
        "category": "design",
    },
    {
        "name": "copilotkit",
        "source": "opencode",
        "description": "Build full-stack agentic apps with CopilotKit — AG-UI protocol",
        "installs": 14000,
        "category": "ai-ml",
    },
    {
        "name": "falkordb",
        "source": "opencode",
        "description": "FalkorDB — ultra-fast property graph database with OpenCypher",
        "installs": 7000,
        "category": "tools",
    },
    {
        "name": "new-loop",
        "source": "opencode",
        "description": "Spin up a new loop (domain) in a file-based knowledge base",
        "installs": 4000,
        "category": "productivity",
    },
    {
        "name": "customize-opencode",
        "source": "opencode",
        "description": "Edit opencode's configuration — agents, subagents, skills, MCP",
        "installs": 8000,
        "category": "tools",
    },
]


# ── Discovery API ───────────────────────────────────────────────────────


def _make_skill(data: dict) -> ExternalSkill:
    return ExternalSkill(
        name=data["name"],
        source=data["source"],
        description=data.get("description", ""),
        installs=data.get("installs", 0),
        category=data.get("category", ""),
        url=data.get("url", ""),
    )


def search_external(query: str, k: int = 10) -> list[ExternalSkill]:
    """Search external skills ecosystem by query.

    Searches the curated popular skills dataset using keyword matching.
    Future: integrate with skills.sh API for live results.
    """
    query_lower = query.lower()
    query_terms = query_lower.split()

    scored: list[tuple[int, ExternalSkill]] = []

    for data in _POPULAR_SKILLS:
        text = (
            f"{data['name']} {data.get('description', '')} "
            f"{data.get('category', '')} {data['source']}"
        ).lower()
        score = 0

        if query_lower in data["name"].lower():
            score += 10
        if any(term in data.get("description", "").lower() for term in query_terms):
            score += 5
        if query_lower in data.get("category", ""):
            score += 8
        if query_lower in data["source"].lower():
            score += 3
        for term in query_terms:
            if term in text:
                score += 1

        if score > 0:
            scored.append((score, _make_skill(data)))

    scored.sort(key=lambda x: (x[0], x[1].installs), reverse=True)
    return [skill for _, skill in scored[:k]]


def popular_skills(k: int = 10, category: str | None = None) -> list[ExternalSkill]:
    """Get most popular skills, optionally filtered by category."""
    skills = [_make_skill(d) for d in _POPULAR_SKILLS]
    if category:
        skills = [s for s in skills if s.category == category]
    skills.sort(key=lambda s: s.installs, reverse=True)
    return skills[:k]


def list_categories() -> list[str]:
    """List all available skill categories."""
    cats: set[str] = set()
    for d in _POPULAR_SKILLS:
        cats.add(d.get("category", "uncategorized"))
    return sorted(cats)


def try_skills_sh_api(query: str) -> list[dict] | None:
    """Fetch live results from skills.sh API.

    Returns None if the API is unreachable.
    """
    url = f"https://skills.sh/api/search?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if isinstance(data, list):
                return data
    except (_urlerror.URLError, json.JSONDecodeError):
        return None
    return None
