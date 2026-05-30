"""Lance le serveur MCP (SSE transport) sur le port 8001."""

import os

# Configuration FastMCP Settings (env prefix FASTMCP_)
os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")
os.environ.setdefault("FASTMCP_PORT", "8001")
os.environ.setdefault("FASTMCP_LOG_LEVEL", "INFO")

from trading.mcp.server import mcp

if __name__ == "__main__":
    # SSE transport — Hermes se connecte à /sse
    mcp.run(transport="sse")
