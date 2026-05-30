#!/usr/bin/env python3
"""Lance le Event Listener (websocket + polling)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import asyncio
import logging

from trading.events.listener import EventListener

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    listener = EventListener()
    try:
        asyncio.run(listener.run())
    except KeyboardInterrupt:
        print("\nListener arrêté.")
