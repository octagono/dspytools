# MCP Server

The unified MCP (Model Context Protocol) server exposes DSPyTools functionality to AI agents through stdio and SSE transports.

## Transport

```bash
# stdio (default — for local MCP clients)
dspytools mcp serve --transport stdio

# SSE (for remote clients)
dspytools mcp serve --transport sse --port 8002
```

## 65 Tools (all built-in)

| Category | Tools |
|----------|-------|
| **Programs** | `list_programs`, `swap_program`, `infer`, `get_program_metadata`, `stream_infer` |
| **Listing** | `list_signatures`, `list_modules`, `list_compiled_runs`, `list_optimizers`, `compile_stats` |
| **Compile** | `compile_optimizer`, `compile_cost`, `holdout_status` |
| **Skills** | `skills_list`, `skills_search`, `skills_external_search` |
| **Drift** | `drift_status`, `drift_history`, `drift_auto_fix` |
| **Synthetic** | `gfl_synthesize`, `challenger_solver`, `meta_prompt_learn` |
| **Trace2Skill** | `trace2skill_evolve` |
| **Paper Optimizers** | `spin_optimize`, `lse_explore`, `gepa_frontier`, `opsd_purify` |
| **Diagnostics** | `doctor` |
| **Sandbox** | `sandbox_execute`, `sandbox_stats` |
| **Validation** | `validate_deploy`, `archive_search` |
| **Cache** | `cache_stats`, `cache_invalidate` |
| **Agent** | `agent_run` |
| **Self** | `self_status`, `inspect_history` |
| **MLflow** | `mlflow_status` |
| **Evaluate** | `evaluate` |
| **GFL** | `gfl_run_halving` |
| **Generate** | `generate_llms_txt` |
| **LoRA** | `lora_list_adapters`, `lora_load_adapter`, `lora_unload_adapter` |
| **Graph** | `graph_query`, `graph_skill_tree`, `graph_program_lineage`, `graph_stats`, `graph_add_dependency`, `graph_dependents`, `graph_record_program` |
| **Memory** | `memory_add`, `memory_search`, `memory_get_all`, `memory_delete`, `memory_update`, `memory_stats` |
| **Redis** | `redis_get`, `redis_set`, `redis_stats`, `redis_flush` |

Responses cached 5s TTL for read-only tools (`list_programs`, `list_signatures`, `list_modules`, `list_compiled_runs`).

## 9 Resources

All served as `application/json`:

| URI | Description |
|-----|-------------|
| `dspytools://programs` | All compiled programs |
| `dspytools://programs/{id}` | Single program detail |
| `dspytools://config` | DSPyTools configuration |
| `dspytools://mlflow` | MLflow tracking status |
| `dspytools://skills` | All skills in library |
| `dspytools://evolve` | Self-evolve engine state |
| `dspytools://gfl/status` | GFL pipeline status |
| `dspytools://sandbox` | Sandbox worker pool status |
| `dspytools://optimizers` | Available DSPy optimizers |

## 3 Prompts

| Prompt | Content |
|--------|---------|
| `compile` | Guide: list optimizers → compile → verify |
| `gfl` | Guide: run 4-way halving comparison |
| `validate` | Guide: SPRT validation before deployment |

## Response Caching

Read-only tool responses are cached in-memory with a 5-second TTL (`tools.py` `_cache` dict). Mutating tools (`swap_program`) invalidate relevant cache keys.

## Resources

- Server: `src/dspytools/mcp/server.py` (64 built-in tools)
- Tools: `src/dspytools/mcp/tools.py`
- Session pool: `src/dspytools/mcp/loader.py`
