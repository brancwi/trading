"""ORM models for the monitoring time-series tables.

These tables live in the monitoring DB (TimescaleDB or SQLite) and are
separate from the main trading business tables.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text

from trading.monitoring.database import MonitoringBase


class LLMCallLog(MonitoringBase):
    """Detailed trace of every LLM inference call."""

    __tablename__ = "llm_call_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    # Identification
    function_name = Column(String, nullable=False, index=True)
    triggered_by = Column(String, default="decision_llm", index=True)
    portfolio_id = Column(String, index=True)

    # Model / Provider
    model = Column(String, nullable=False, index=True)
    provider = Column(String, nullable=False, index=True)
    backend = Column(String, default="local")  # local | cloud | hybrid

    # Tokens & Cost
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)

    # Performance
    duration_ms = Column(Float, default=0.0)

    # Content hashes (anonymised)
    prompt_hash = Column(String)
    response_hash = Column(String)

    # Provider metadata (JSON)
    provider_info_json = Column(Text)

    # Error tracking
    error_message = Column(Text)


class MessageLog(MonitoringBase):
    """Trace of every incoming message / event by channel."""

    __tablename__ = "message_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    # Channel taxonomy
    channel = Column(String, nullable=False, index=True)
    source = Column(String, nullable=False, index=True)

    # Content (hashed for privacy)
    content_hash = Column(String)

    # Flexible metadata
    metadata_json = Column(Text)

    # Processing state
    processed = Column(Integer, default=0)
    processing_time_ms = Column(Float)


class PerformanceSnapshot(MonitoringBase):
    """Time-series performance metrics (latencies, throughputs, queue depths)."""

    __tablename__ = "performance_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    metric_name = Column(String, nullable=False, index=True)
    value = Column(Float, nullable=False)
    unit = Column(String)

    # JSON tags for dimensions (portfolio_id, ticker, endpoint, etc.)
    tags_json = Column(Text)
