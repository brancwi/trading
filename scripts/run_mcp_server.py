"""Lance le serveur MCP (SSE transport) sur le port configuré."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.core.config import get_settings

settings = get_settings()

# Configuration FastMCP Settings (env prefix FASTMCP_)
os.environ.setdefault("FASTMCP_HOST", settings.api_host)
os.environ.setdefault("FASTMCP_PORT", str(settings.resolved_mcp_port))
os.environ.setdefault("FASTMCP_LOG_LEVEL", "INFO")

from trading.mcp.server import mcp

if __name__ == "__main__":
    # SSE transport — Hermes se connecte à /sse
    mcp.run(transport="sse")
