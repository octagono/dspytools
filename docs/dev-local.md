# Dev Local — Full Stack Launcher

## Services

| # | Service | Port | Purpose |
|---|---------|------|---------|
| 1 | FalkorDB | 6379 | Graph database (skills, memory, lineage) + Redis cache |
| 2 | LLM Server | 8000 | LLM inference endpoint (configurable) |
| 3 | Embedding | 11434 | embeddinggemma (768-dim) |
| 4 | MLflow | 5000 | Experiment tracking (file store at `$PROJECT_ROOT/.mlflow`) |
| 5 | FastAPI | 8080 | Hot-swap inference server |
| 6 | MCP SSE | 8002 | MCP server for agent connections |

## Quick Start

```bash
scripts/dev-local.sh up         # Start all services
scripts/dev-local.sh status     # Check health of all services
scripts/dev-local.sh down       # Stop all services
scripts/dev-local.sh logs <svc> # Tail logs: redis|llm|emb|mlflow|api|mcp
```

## Commands

| Command | Description |
|---------|-------------|
| `up` | Start all services (FalkorDB, LLM, embedding, MLflow, API, MCP) |
| `down` | Stop all services |
| `status` | Health check all running services |
| `logs <svc>` | Tail logs for a specific service |
| `restart <svc>` | Restart a specific service |
| `redis-start\|stop` | FalkorDB container only |
| `mlflow-start\|stop` | MLflow tracking server only |
| `api-start\|stop` | FastAPI hot-swap server only |
| `mcp-start\|stop` | MCP SSE server only |

## Service Details

### FalkorDB (Redis Stack)
Podman container running `falkordb/falkordb:latest` on port 6379.
Provides graph queries (Cypher), vector search, and response caching.

### LLM Server
The LLM server runs on port 8000. Configure the endpoint via `DSPYTOOLS_LLM_URL` environment variable.
Models are served by llama-cpp-server (default on port 8080).

### Embedding Server
Runs on port 8001. Configure via `DSPYTOOLS_EMBEDDING_URL`.
Default model: `embeddinggemma` (768-dim).

### MLflow
File-based tracking at `$PROJECT_ROOT/.mlflow`. Web UI on port 5000.

### FastAPI
Hot-swap inference server. Loads compiled programs and serves them over HTTP.
Web UI on port 8080.

### MCP Server
SSE transport on port 8002 for agent connections.
