#!/usr/bin/env bash
# Wrapper to run dspytools MCP server from project root
# Used by .mcp.json stdio transport for agent integration
cd /home/octagono/dev/dspytools || exit 1
export PATH="/home/octagono/dev/dspytools/.venv/bin:$PATH"
exec dspytools mcp serve --transport stdio "$@"
