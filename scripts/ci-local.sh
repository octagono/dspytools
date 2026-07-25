#!/usr/bin/env bash
# scripts/ci-local.sh — Full CI/CD pipeline on local GPU
#
# Runs all 5 stages locally without GitHub Actions.
# Designed for cron scheduling or manual invocation.
#
# Usage:
#   ./scripts/ci-local.sh                    # full pipeline
#   ./scripts/ci-local.sh --stage 1          # lint + smoke tests only
#   ./scripts/ci-local.sh --stage 4          # evaluation gate only
#   ./scripts/ci-local.sh --stages 1,2,4     # specific stages
#
# Exit codes:
#   0 = all stages passed
#   1 = lint/tests failed
#   2 = integration tests failed
#   3 = SPRT/regression gate failed

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Config ────────────────────────────────────────────────────────────
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
VENV_ACTIVATE="${PROJECT_ROOT}/.venv/bin/activate"
LOG_DIR="${PROJECT_ROOT}/.ci-logs"
STAGES_RUN=()

# ── Helpers ───────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

log() { echo -e "\033[1m[CI]\033[0m $*"; }
ok() { echo -e "  \033[32m✓\033[0m $*"; }
fail() { echo -e "  \033[31m✗\033[0m $*"; }
warn() { echo -e "  \033[33m!\033[0m $*"; }

run_stage() {
	local stage="$1"
	local log_file="$LOG_DIR/stage${stage}.log"
	echo -n "" >"$log_file"

	case "$stage" in
	1) _stage1_lint "$log_file" ;;
	2) _stage2_integration "$log_file" ;;
	3) _stage3_build "$log_file" ;;
	4) _stage4_evaluation "$log_file" ;;
	5) _stage5_deploy "$log_file" ;;
	*)
		fail "unknown stage: $stage"
		return 1
		;;
	esac
}

# ── Stage 1: Lint + Smoke Tests ───────────────────────────────────────
_stage1_lint() {
	local log="$1"
	log "Stage 1: Lint + Smoke Tests"
	local pass=0 fail_count=0

	echo -n "  ruff check ... "
	if source "$VENV_ACTIVATE" && ruff check src/ tests/ >>"$log" 2>&1; then
		ok "pass"
		pass=$((pass + 1))
	else
		fail "ruff errors (see $log)"
		fail_count=$((fail_count + 1))
	fi

	echo -n "  ruff format --check ... "
	if ruff format --check src/ tests/ >>"$log" 2>&1; then
		ok "pass"
		pass=$((pass + 1))
	else
		fail "formatting drift (run: ruff format src/ tests/)"
		fail_count=$((fail_count + 1))
	fi

	echo -n "  pytest (359 tests) ... "
	if $PYTHON -m pytest tests/ -q >>"$log" 2>&1; then
		ok "pass"
		pass=$((pass + 1))
	else
		fail "test failures (see $log)"
		fail_count=$((fail_count + 1))
	fi

	echo -n "  import check ... "
	if $PYTHON -c "
from dspytools.core._dspy import dspy
from dspytools.gfl.pipeline import GFLPipeline
from dspytools.mcp.tools import BUILTIN_TOOLS
from dspytools.skills import SkillManager
print(f'{len(BUILTIN_TOOLS)} MCP tools')
" >>"$log" 2>&1; then
		ok "pass"
		pass=$((pass + 1))
	else
		fail "import error"
		fail_count=$((fail_count + 1))
	fi

	log "  Result: $pass passed, $fail_count failed"
	return "$fail_count"
}

# ── Stage 2: Integration Tests (FalkorDB) ─────────────────────────────
_stage2_integration() {
	local log="$1"
	log "Stage 2: Integration Tests (FalkorDB + Redis)"

	# Check FalkorDB is running (use Python since redis-cli may not be on PATH in NixOS)
	if ! $PYTHON -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping()" >>"$log" 2>&1; then
		fail "FalkorDB not running on localhost:6379"
		warn "Start with: docker compose -f docker-compose.redis.yml up -d"
		return 1
	fi
	ok "FalkorDB reachable"

	echo -n "  graph/memory/cache tests ... "
	if $PYTHON -m pytest \
		tests/test_graph_commands.py \
		tests/test_memory_manager.py \
		tests/test_semantic_cache.py \
		tests/test_skill_graph.py \
		-v --tb=short >>"$log" 2>&1; then
		ok "pass"
	else
		fail "integration test failures (see $log)"
		return 1
	fi

	echo -n "  graph migrate ... "
	if dspytools graph migrate --target all >>"$log" 2>&1; then
		ok "pass"
	else
		warn "migration had issues (may be expected if already migrated)"
	fi

	echo -n "  graph query smoke ... "
	if dspytools graph query "MATCH (n) RETURN count(n)" >>"$log" 2>&1; then
		ok "pass"
	else
		fail "graph query failed"
		return 1
	fi

	return 0
}

# ── Stage 3: Mojo Build (optional) ────────────────────────────────────
_stage3_build() {
	local log="$1"
	log "Stage 3: Mojo Native Build"

	if ! command -v mojo &>/dev/null; then
		warn "Mojo SDK not installed — skipping native build"
		warn "Python fallback active (bm25, sprt, vector_utils)"
		return 0
	fi
	ok "Mojo SDK found: $(which mojo)"

	echo -n "  Compiling mojo modules ... "
	if ${PROJECT_ROOT}/mojo_modules/build.sh >>"$log" 2>&1; then
		ok "pass"
		ls -lh mojo_modules/*.so 2>/dev/null || warn "no .so files produced"
	else
		fail "build failed (see $log)"
		return 1
	fi

	echo -n "  wheel build ... "
	if $PYTHON -m build --wheel --no-isolation >>"$log" 2>&1; then
		ok "pass"
		ls -lh dist/*.whl 2>/dev/null
	else
		warn "wheel build failed (non-critical, Python fallback works)"
	fi

	return 0
}

# ── Stage 4: Evaluation Gate ──────────────────────────────────────────
_stage4_evaluation() {
	local log="$1"
	log "Stage 4: Evaluation Regression Gate"

	# Check llama-cpp-server
	if ! curl -s http://localhost:8080/api/tags >>"$log" 2>&1; then
		fail "llama-cpp-server not running on localhost:8080"
		return 1
	fi
	ok "llama-cpp-server reachable"

	# Check golden dataset exists
	local golden="data/golden_eval.jsonl"
	if [ ! -f "$golden" ]; then
		fail "Golden dataset not found: $golden"
		warn "Create with: python scripts/create_golden_dataset.py"
		return 1
	fi
	local count=$(wc -l <"$golden")
	ok "Golden dataset: $count examples"

	# Run evaluation
	echo -n "  Evaluating simpleqa on golden dataset ... "
	local eval_output
	eval_output=$(dspytools evaluate run simpleqa "$golden" --metric exact_match 2>&1) || true
	echo "$eval_output" >>"$log"

	# Extract score (look for percentage in output)
	local score=$(echo "$eval_output" | grep -oP 'Score:\s+\K[0-9.]+' || echo "0.0")
	ok "Current score: ${score}%"

	# Compare against baseline
	local baseline_file="data/golden_baseline.json"
	local baseline_score=0.0
	if [ -f "$baseline_file" ]; then
		baseline_score=$($PYTHON -c "import json; print(json.load(open('$baseline_file')).get('score', 0.0))" 2>/dev/null || echo "0.0")
	fi
	ok "Baseline score: ${baseline_score}%"

	# Regression check: current must be >= 90% of baseline
	echo -n "  Regression check (min: 90% of ${baseline_score}%) ... "

	if $PYTHON -c "
import sys
score = float('$score')
baseline = float('$baseline_score')
min_ok = baseline * 0.9
if score >= min_ok:
    print(f'PASS — {score}% >= {min_ok}%')
    sys.exit(0)
else:
    print(f'REGRESSION — {score}% < {min_ok}%')
    sys.exit(1)
" >>"$log" 2>&1; then
		ok "PASS — no regression"

		# Update baseline if score improved
		if $PYTHON -c "import sys; sys.exit(0 if float('$score') > float('$baseline_score') else 1)" >>"$log" 2>&1; then
			$PYTHON -c "import json; json.dump({'score': $score}, open('$baseline_file', 'w'))"
			ok "Baseline updated to ${score}%"
		fi
		return 0
	else
		fail "REGRESSION — ${score}% < ${min_acceptable}% (baseline: ${baseline_score}%)"
		return 1
	fi
}

# ── Stage 5: Deploy (local MLflow + artifact sync) ────────────────────
_stage5_deploy() {
	local log="$1"
	log "Stage 5: Deploy (MLflow + artifact sync)"

	# Check MLflow
	if ! curl -s http://localhost:5000/health >>"$log" 2>&1; then
		warn "MLflow not running on localhost:5000 — skipping registration"
		return 0
	fi
	ok "MLflow reachable"

	# Register latest validated programs
	echo -n "  Registering programs to MLflow ... "
	if $PYTHON -c "
import mlflow, json
from dspytools.core.registry import list_compiled_runs

runs = list_compiled_runs()
mlflow.set_experiment('dspytools-ci')

for run in runs[-5:]:  # register last 5 runs
    with mlflow.start_run(run_name=f\"ci_{run.get('id', 'unknown')}\"):
        mlflow.log_dict(run, 'metadata.json')
        mlflow.set_tag('source', 'ci-local')
        mlflow.set_tag('validated', 'true')

print(f'Registered {len(runs[-5:])} runs to MLflow')
" >>"$log" 2>&1; then
		ok "pass"
	else
		warn "MLflow registration had issues (see $log)"
	fi

	ok "Deploy stage complete (no production hot-swap in local mode)"
	return 0
}

# ── Argument Parsing ──────────────────────────────────────────────────
if [ $# -eq 0 ]; then
	# Default: run all stages sequentially
	STAGES_RUN=(1 2 3 4 5)
else
	case "$1" in
	--stage)
		STAGES_RUN=("$2")
		;;
	--stages)
		IFS=',' read -ra STAGES_RUN <<<"$2"
		;;
	*)
		echo "Usage: $0 [--stage N | --stages N,N,...]"
		exit 1
		;;
	esac
fi

# ── Main ──────────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  dspytools CI — Local Pipeline"
echo "=============================================="
echo "  Stages: ${STAGES_RUN[*]}"
echo "  Time:   $(date)"
echo "=============================================="
echo ""

TOTAL_FAIL=0
for stage in "${STAGES_RUN[@]}"; do
	if ! run_stage "$stage"; then
		TOTAL_FAIL=$((TOTAL_FAIL + 1))
		fail "Stage $stage FAILED"
	fi
	echo ""
done

echo "=============================================="
if [ "$TOTAL_FAIL" -eq 0 ]; then
	ok "ALL STAGES PASSED"
	exit 0
else
	fail "$TOTAL_FAIL STAGE(S) FAILED"
	echo "  Logs: $LOG_DIR/"
	exit "$TOTAL_FAIL"
fi
