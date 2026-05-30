"""Lance le serveur MCP (SSE transport) sur le port 8001."""

import asyncio
from trading.mcp.server import mcp

if __name__ == "__main__":
    # SSE transport — Hermes se connecte à /sse
    mcp.run(transport="sse", port=8001)
