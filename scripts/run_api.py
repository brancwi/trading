#!/usr/bin/env python3
"""Lance le serveur API FastAPI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn
from trading.core.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "trading.api.main:app",
        host=settings.api_host,
        port=settings.resolved_api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
