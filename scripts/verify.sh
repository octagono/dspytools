#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src"

PASS=0
FAIL=0

check() {
	local name="$1"
	shift
	echo -n "  $name ... "
	if "$@" >/dev/null 2>&1; then
		echo "✅"
		PASS=$((PASS + 1))
	else
		echo "❌"
		FAIL=$((FAIL + 1))
	fi
}

echo "=== dspytools verify ==="
echo ""

# ── Lint ──────────────────────────────────────────
echo "Lint:"
check "ruff" ruff check --fix --unsafe-fixes
echo ""

# ── Tests ─────────────────────────────────────────
echo "Tests:"
check "pytest (360)" "$PYTHON_BIN" -m pytest tests/ -q
echo ""

# ── Imports ───────────────────────────────────────
echo "Imports:"
check "core" "$PYTHON_BIN" -c "from dspytools.core._dspy import dspy; from dspytools.core.setup import LMRegistry; from dspytools.core.hotswap import HotSwapManager; from dspytools.core.registry import register_run, list_compiled_runs; from dspytools.core.mlflow_tracker import get_tracker; from dspytools.core.errors import DspyToolsError, ServiceUnavailableError, CompileError"
check "gfl" "$PYTHON_BIN" -c "from dspytools.gfl.pipeline import GFLPipeline; from dspytools.gfl.paper_optimizers import SPINOptimizer, MetaPromptOptimizer, GEPAParetoFrontier, LSETreeExplorer, GRAOMetaOptimizer, PurifiedOPSDOptimizer; from dspytools.gfl.synthetic import ChallengerSolver; from dspytools.gfl.consolidation import SkillConsolidator"
check "generate" "$PYTHON_BIN" -c "from dspytools.generate import RepositoryAnalyzer, llms_txt_quality, build_ground_truth_examples, GitRepoExplorer, get_sandbox_pool"
check "evolve" "$PYTHON_BIN" -c "from dspytools.evolve.self_evolve import SelfEvolveEngine; from dspytools.evolve.router import RouterAgent"
check "commands" "$PYTHON_BIN" -c "from dspytools.commands.compile import compile_cmd, _OPTIMIZER_SPECS; from dspytools.commands.generate import generate_cmd; from dspytools.core.metrics import exact_match_metric"
check "mcp" "$PYTHON_BIN" -c "from dspytools.mcp.server import create_mcp_server; from dspytools.mcp.tools import BUILTIN_TOOLS"
check "skills" "$PYTHON_BIN" -c "from dspytools.skills import SkillManager"
check "config" "$PYTHON_BIN" -c "from dspytools.config.settings import compiled_dir, mlflow_tracking_uri; from dspytools.config.env import load_env"
echo ""

# ── Summary ───────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo "=== Result: $PASS/$TOTAL passed ==="
if [ "$FAIL" -gt 0 ]; then
	echo "❌ $FAIL checks failed"
	exit 1
else
	echo "✅ All checks passed"
fi
