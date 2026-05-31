"""MessageLogger — traces every inbound message / event by channel.

Channels:
  news_finnhub    — News fetched from Finnhub API
  api_rest        — Incoming REST API requests
  websocket       — WebSocket messages
  command_bus     — Commands queued for execution
  mcp_tool        — MCP tool invocations
  prefect_flow    — Prefect flow runs
  system          — Internal system events
"""

import hashlib
import json
import logging
import time
from typing import Any

from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)

VALID_CHANNELS = {
    "news_finnhub",
    "api_rest",
    "websocket",
    "command_bus",
    "mcp_tool",
    "prefect_flow",
    "system",
}


class MessageLogger:
    """Singleton message logger."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def log(
        channel: str,
        source: str,
        content: str | None = None,
        metadata: dict | None = None,
        processed: bool = False,
        processing_time_ms: float | None = None,
    ) -> None:
        """Log a message to the monitoring DB.

        Args:
            channel: One of the VALID_CHANNELS.
            source:  Human-readable source (e.g. "finnhub_company_news",
                     "POST /decisions", "mcp.get_system_status").
            content: Raw message content (will be hashed, not stored raw).
            metadata: Free-form JSON-able metadata dict.
            processed: Whether the message has already been processed.
            processing_time_ms: How long processing took, if known.
        """
        if channel not in VALID_CHANNELS:
            logger.warning("[MessageLogger] Unknown channel '%s', using 'system'", channel)
            channel = "system"

        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        metadata_json = json.dumps(metadata, default=str) if metadata else None

        MonitorService().log_message(
            channel=channel,
            source=source,
            content_hash=content_hash,
            metadata_json=metadata_json,
            processed=1 if processed else 0,
            processing_time_ms=processing_time_ms,
        )

    @staticmethod
    def timed_log(
        channel: str,
        source: str,
        content: str | None = None,
        metadata: dict | None = None,
    ):
        """Context manager that logs a message and measures processing time.

        Usage:
            with MessageLogger.timed_log("news_finnhub", "finnhub") as log:
                articles = fetch_news()
                log.metadata["article_count"] = len(articles)
        """
        return _TimedLogContext(channel, source, content, metadata)


class _TimedLogContext:
    """Context manager for timed message logging."""

    def __init__(self, channel: str, source: str, content: str | None, metadata: dict | None):
        self.channel = channel
        self.source = source
        self.content = content
        self.metadata = metadata or {}
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.perf_counter() - self.start) * 1000, 2) if self.start else None
        MessageLogger.log(
            channel=self.channel,
            source=self.source,
            content=self.content,
            metadata=self.metadata,
            processed=exc_type is None,
            processing_time_ms=duration_ms,
        )
        return False
