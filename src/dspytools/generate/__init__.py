"""llms.txt generation pipeline — DSPy module + signatures + quality + data.

CLI: dspytools generate llms-txt|batch|stream
Compile: dspytools compile mipro generate.RepositoryAnalyzer trainset.json
"""

from __future__ import annotations

from dspytools.core.metrics import llms_txt_metric, llms_txt_quality
from dspytools.generate.data import build_ground_truth_examples
from dspytools.generate.explorer import (
    GitRepoExplorer,
    gather_repository_info,
    load_mcp_tools_sync,
)
from dspytools.generate.module import (
    AnalyzeCodeStructure,
    AnalyzeRepository,
    GenerateLLMsTxt,
    RepositoryAnalyzer,
    SandboxPool,
    get_sandbox_pool,
)

__all__ = [
    "AnalyzeCodeStructure",
    "AnalyzeRepository",
    "GenerateLLMsTxt",
    "GitRepoExplorer",
    "RepositoryAnalyzer",
    "SandboxPool",
    "build_ground_truth_examples",
    "gather_repository_info",
    "get_sandbox_pool",
    "llms_txt_metric",
    "llms_txt_quality",
    "load_mcp_tools_sync",
]
