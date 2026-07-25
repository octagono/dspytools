#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════
# dev-local.sh — One-command service orchestrator for dspytools
#
# Services managed:
#   1. FalkorDB (Docker)    — Redis + graph DB on port 6379
#   2. llama-cpp-server (external) — LM inference on port 8080
#   3. MCP Server (SSE)      — Agent tool surface on port 8002
#   4. API Server (FastAPI)  — REST inference on port 8080
#
# Usage:
#   ./scripts/dev-local.sh up       Start all services
#   ./scripts/dev-local.sh down     Stop all services
#   ./scripts/dev-local.sh status   Show service health
#   ./scripts/dev-local.sh restart  Restart all services
#   ./scripts/dev-local.sh logs     Tail service logs (Ctrl-C to exit)
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
DSPYTOOLS_BIN="${PROJECT_ROOT}/.venv/bin/dspytools"
TMUX_SESSION="dspytools-dev"

# Service config
FALKORDB_CONTAINER="dspytools-falkordb"
FALKORDB_IMAGE="docker.io/falkordb/falkordb:latest"
FALKORDB_PORT=6379
LLAMA_CPP_URL="http://localhost:8080"
MCP_PORT=8002
API_PORT=8080

# Colors
G='\033[0;32m'
R='\033[0;31m'
Y='\033[1;33m'
C='\033[0;36m'
D='\033[0;90m'
N='\033[0m'

export PYTHONPATH="${PROJECT_ROOT}/src"

# ── Helpers ────────────────────────────────────────────────────────────────

log() { echo -e "  ${G}✓${N} $1"; }
warn() { echo -e "  ${Y}⚠${N} $1"; }
err() { echo -e "  ${R}✗${N} $1"; }
hdr() { echo -e "\n${C}═══ $1 ═══${N}"; }

check_port() {
	local port=$1
	if command -v ss &>/dev/null; then
		ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0 || return 1
	elif command -v lsof &>/dev/null; then
		lsof -i :${port} -P -n 2>/dev/null | grep -q LISTEN && return 0 || return 1
	fi
	return 1
}

wait_for() {
	local url=$1
	local name=$2
	local max=${3:-30}
	for i in $(seq 1 "$max"); do
		if curl -sf "$url" >/dev/null 2>&1; then
			log "$name is up"
			return 0
		fi
		sleep 1
	done
	err "$name did not respond at $url after ${max}s"
	return 1
}

# ── FalkorDB ───────────────────────────────────────────────────────────────

start_falkordb() {
	if rtk docker inspect -f '{{.State.Running}}' "$FALKORDB_CONTAINER" 2>/dev/null | grep -q true; then
		log "FalkorDB already running"
		return 0
	fi
	echo -e "  ${D}starting FalkorDB container...${N}"
	rtk docker run -d --name "$FALKORDB_CONTAINER" \
		-p ${FALKORDB_PORT}:6379 \
		"$FALKORDB_IMAGE" >/dev/null 2>&1
	sleep 2
	if rtk docker exec "$FALKORDB_CONTAINER" redis-cli PING 2>/dev/null | grep -q PONG; then
		log "FalkorDB started (port ${FALKORDB_PORT})"
	else
		err "FalkorDB failed to start"
		return 1
	fi
}

stop_falkordb() {
	if rtk docker inspect "$FALKORDB_CONTAINER" &>/dev/null; then
		rtk docker stop "$FALKORDB_CONTAINER" >/dev/null 2>&1 || true
		rtk docker rm "$FALKORDB_CONTAINER" >/dev/null 2>&1 || true
		log "FalkorDB stopped"
	fi
}

# ── llama-cpp-server ────────────────────────────────────────────────────────

check_llama_cpp() {
	if curl -sf "${LLAMA_CPP_URL}/api/tags" >/dev/null 2>&1; then
		local models
		models=$(curl -sf "${LLAMA_CPP_URL}/api/tags" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))" 2>/dev/null || echo "?")
		log "llama-cpp-server running (${models} models)"
	else
		err "llama-cpp-server not responding at ${LLAMA_CPP_URL}"
		warn "Start your llama-cpp-server"
		return 1
	fi
}

# ── MCP + API Servers (tmux) ──────────────────────────────────────────────

start_servers() {
	# Kill existing session
	tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

	tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT"

	# Window 1: MCP Server
	tmux rename-window -t "$TMUX_SESSION" "mcp"
	tmux send-keys -t "$TMUX_SESSION" "cd $PROJECT_ROOT && PYTHONPATH=src $DSPYTOOLS_BIN mcp serve --transport sse --port $MCP_PORT" C-m

	sleep 2

	# Window 2: API Server
	tmux new-window -t "$TMUX_SESSION" -n "api"
	tmux send-keys -t "$TMUX_SESSION" "cd $PROJECT_ROOT && PYTHONPATH=src $DSPYTOOLS_BIN server start --api --no-mcp --api-port $API_PORT" C-m

	sleep 3
	log "MCP server started (port ${MCP_PORT})"
	log "API server started (port ${API_PORT})"
}

stop_servers() {
	if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
		tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true
		log "MCP + API servers stopped"
	fi
}

# ── Commands ───────────────────────────────────────────────────────────────

cmd_up() {
	hdr "Starting dspytools services"
	echo ""
	check_llama_cpp || true
	start_falkordb || true
	start_servers
	echo ""
	cmd_status
	echo ""
	hdr "All services ready"
	echo -e "  MCP SSE:   ${C}http://localhost:${MCP_PORT}/sse${N}"
	echo -e "  API:       ${C}http://localhost:${API_PORT}/health${N}"
	echo -e "  FalkorDB:  ${C}localhost:${FALKORDB_PORT}${N}"
	echo -e "  llama-cpp: ${C}${LLAMA_CPP_URL}${N}"
	echo ""
	echo -e "  Logs:     ${D}./scripts/dev-local.sh logs${N}"
	echo -e "  Stop:     ${D}./scripts/dev-local.sh down${N}"
}

cmd_down() {
	hdr "Stopping dspytools services"
	echo ""
	stop_servers
	stop_falkordb
	echo ""
	log "All services stopped"
}

cmd_status() {
	hdr "Service Status"
	echo ""

	# llama-cpp-server
	if curl -sf "${LLAMA_CPP_URL}/api/tags" >/dev/null 2>&1; then
		models=$(curl -sf "${LLAMA_CPP_URL}/api/tags" 2>/dev/null | python3 -c "import sys,json; print(len(json.load(sys.stdin)['models']))" 2>/dev/null || echo "?")
		log "llama-cpp   :${D} ${LLAMA_CPP_URL} ${N}(${models} models)"
	else
		err "llama-cpp   :${D} not responding${N}"
	fi

	# FalkorDB
	if rtk docker exec "$FALKORDB_CONTAINER" redis-cli PING 2>/dev/null | grep -q PONG; then
		log "FalkorDB    :${D} localhost:${FALKORDB_PORT} ${N}(Redis + graph)"
	else
		err "FalkorDB    :${D} not running${N}"
	fi

	# MCP Server
	if check_port "$MCP_PORT"; then
		log "MCP Server  :${D} http://localhost:${MCP_PORT} ${N}(SSE)"
	else
		err "MCP Server  :${D} not running${N}"
	fi

	# API Server
	if curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1; then
		active=$(curl -sf "http://localhost:${API_PORT}/health" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('active_program','none'))" 2>/dev/null || echo "?")
		log "API Server  :${D} http://localhost:${API_PORT} ${N}(active: ${active})"
	else
		err "API Server  :${D} not running${N}"
	fi

	# tmux
	if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
		echo ""
		echo -e "  ${D}tmux session: ${TMUX_SESSION} (windows: $(tmux list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null | tr '\n' ' '))${N}"
	fi

	echo ""
	echo -e "  ${D}Python:  $($PYTHON_BIN --version 2>/dev/null || echo 'not found')${N}"
	if command -v nvidia-smi &>/dev/null; then
		echo -e "  ${D}GPU:     $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)${N}"
	fi
}

cmd_restart() {
	cmd_down
	sleep 1
	cmd_up
}

cmd_logs() {
	if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
		err "No running tmux session. Run: ./scripts/dev-local.sh up"
		exit 1
	fi

	hdr "Tailing logs (Ctrl-C to detach)"
	echo -e "  ${D}Windows: $(tmux list-windows -t "$TMUX_SESSION" -F '#W' | tr '\n' ', ' | sed 's/,$//')${N}"
	echo ""

	local win="${1:-}"
	if [ -n "$win" ]; then
		echo -e "  Following window: ${C}${win}${N}"
		tmux attach -t "$TMUX_SESSION" -t "=${win}"
	else
		# Cycle through windows showing last 20 lines each
		for w in $(tmux list-windows -t "$TMUX_SESSION" -F '#W' 2>/dev/null); do
			echo -e "\n${C}── ${w} ──${N}"
			tmux capture-pane -t "${TMUX_SESSION}=${w}" -p -S -20 2>/dev/null | tail -15
		done
		echo ""
		echo -e "  Attach live: ${D}tmux attach -t ${TMUX_SESSION}${N}"
		echo -e "  Or:          ${D}./scripts/dev-local.sh logs <window>${N}"
	fi
}

cmd_test() {
	hdr "Running verification"
	echo ""
	"${PROJECT_ROOT}/scripts/verify.sh"
}

# ── Main ───────────────────────────────────────────────────────────────────

case "${1:-status}" in
up) cmd_up ;;
down) cmd_down ;;
status) cmd_status ;;
restart) cmd_restart ;;
logs) cmd_logs "${2:-}" ;;
test) cmd_test ;;
*)
	echo "Usage: dev-local.sh {up|down|status|restart|logs|test}"
	echo ""
	echo "  up         Start FalkorDB + MCP + API servers"
	echo "  down       Stop all services"
	echo "  status     Show health of all services"
	echo "  restart    Down then up"
	echo "  logs [win] Show recent logs or attach to a window (mcp|api)"
	echo "  test       Run full verification"
	echo ""
	;;
esac
