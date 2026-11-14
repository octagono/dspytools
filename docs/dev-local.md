# Dev Local — Full Stack Launcher

## Services

| # | Service | Port | Purpose |
|---|---------|------|---------|
| 1 | FalkorDB | 6379 | Graph database (skills, memory, lineage) + Redis cache |
| 2 | LLM Server | 8080 | LM inference endpoint (llama-cpp-server, external — checked, not started) |
| 3 | FastAPI | 8080 | Hot-swap inference server |
| 4 | MCP SSE | 8002 | MCP server for agent connections |

`scripts/dev-local.sh` manages FalkorDB (Docker) plus the MCP and API servers (in a `dspytools-dev` tmux session). It does not start the llama-cpp-server, an embedding server, or MLflow — those are external.

## Quick Start

```bash
scripts/dev-local.sh up         # Start FalkorDB + MCP + API servers
scripts/dev-local.sh status     # Check health of all services
scripts/dev-local.sh down       # Stop all services
scripts/dev-local.sh logs mcp   # Attach to a tmux window: mcp|api
scripts/dev-local.sh test       # Run scripts/verify.sh
```

## Commands

| Command | Description |
|---------|-------------|
| `up` | Start FalkorDB (Docker), then MCP + API servers in tmux |
| `down` | Stop the tmux session and the FalkorDB container |
| `status` | Health check llama-cpp-server, FalkorDB, MCP, API |
| `logs [win]` | Attach to a tmux window (`mcp` or `api`); with no window, print the last lines of each |
| `restart` | Down then up (all services) |
| `test` | Run `scripts/verify.sh` |

## Service Details

### FalkorDB (Redis Stack)
Docker container (`rtk docker`) running `docker.io/falkordb/falkordb:latest` on port 6379.
Provides graph queries (Cypher), vector search, and response caching.

### LLM Server
The LLM server runs on llama-cpp-server (default on port 8080) and is **not** managed by this script — `up`/`status` only probe it. Configure the endpoint via `DSPYTOOLS_LLAMA_CPP_URL` environment variable.

### FastAPI
Hot-swap inference server. Loads compiled programs and serves them over HTTP.
Web UI on port 8080.

### MCP Server
SSE transport on port 8002 for agent connections.
