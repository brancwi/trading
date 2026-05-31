"""MonitorService — collecte et stockage des métriques, audit, token usage.

Now extended with a dedicated monitoring DB (TimescaleDB or SQLite)
for LLM call traces, message logs, and performance snapshots.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from trading.core.database import db_session
from trading.core.models import AuditLog, MonitoringMetric, TokenUsageLog
from trading.monitoring.database import monitoring_session, IS_TIMESCALEDB
from trading.monitoring.models import LLMCallLog, MessageLog, PerformanceSnapshot

logger = logging.getLogger(__name__)


class MonitorService:
    """Service singleton pour le monitoring et l'audit."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------
    # Métriques time-series
    # ------------------------------------------------------------------

    @staticmethod
    def record_metric(
        name: str,
        value: float,
        unit: str | None = None,
        source: str = "system",
        tags: dict | None = None,
        db: Session | None = None,
    ) -> None:
        """Enregistre une métrique time-series."""
        if db is None:
            with db_session() as session:
                MonitorService._write_metric(session, name, value, unit, source, tags)
                session.commit()
        else:
            MonitorService._write_metric(db, name, value, unit, source, tags)

    @staticmethod
    def _write_metric(db, name, value, unit, source, tags):
        metric = MonitoringMetric(
            metric_name=name,
            value=value,
            unit=unit,
            source=source,
            tags_json=json.dumps(tags) if tags else None,
        )
        db.add(metric)

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    @staticmethod
    def log_event(
        event_type: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        actor: str = "system",
        details: dict | None = None,
        severity: str = "info",
        db: Session | None = None,
    ) -> None:
        """Écrit une entrée dans le journal d'audit."""
        if db is None:
            with db_session() as session:
                MonitorService._write_audit(session, event_type, entity_type, entity_id, actor, details, severity)
                session.commit()
        else:
            MonitorService._write_audit(db, event_type, entity_type, entity_id, actor, details, severity)

    @staticmethod
    def _write_audit(db, event_type, entity_type, entity_id, actor, details, severity):
        entry = AuditLog(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            details_json=json.dumps(details, default=str) if details else None,
            severity=severity,
        )
        db.add(entry)

    # ------------------------------------------------------------------
    # Token usage
    # ------------------------------------------------------------------

    @staticmethod
    def log_token_usage(
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        text: str | None = None,
        triggered_by: str = "qwen_arbitration",
        db: Session | None = None,
    ) -> None:
        """Enregistre la consommation tokens d'une inférence."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16] if text else None
        if db is None:
            with db_session() as session:
                MonitorService._write_token_usage(
                    session, model, provider, input_tokens, output_tokens, cost_usd, text_hash, triggered_by
                )
                session.commit()
        else:
            MonitorService._write_token_usage(
                db, model, provider, input_tokens, output_tokens, cost_usd, text_hash, triggered_by
            )

    @staticmethod
    def _write_token_usage(db, model, provider, input_tokens, output_tokens, cost_usd, text_hash, triggered_by):
        entry = TokenUsageLog(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            text_hash=text_hash,
            triggered_by=triggered_by,
        )
        db.add(entry)

    # ------------------------------------------------------------------
    # Queries pour l'API / Hermes
    # ------------------------------------------------------------------

    @staticmethod
    def get_metrics_summary(hours: int = 24, db: Session | None = None) -> dict:
        """Résumé des métriques sur les N dernières heures."""
        since = datetime.utcnow() - timedelta(hours=hours)
        with db_session() as session:
            metrics = (
                session.query(
                    MonitoringMetric.metric_name,
                    MonitoringMetric.source,
                    func.count(MonitoringMetric.id).label("count"),
                    func.avg(MonitoringMetric.value).label("avg_value"),
                    func.min(MonitoringMetric.value).label("min_value"),
                    func.max(MonitoringMetric.value).label("max_value"),
                )
                .filter(MonitoringMetric.timestamp >= since)
                .group_by(MonitoringMetric.metric_name, MonitoringMetric.source)
                .all()
            )
            return {
                "period_hours": hours,
                "metrics": [
                    {
                        "name": m.metric_name,
                        "source": m.source,
                        "count": m.count,
                        "avg": round(m.avg_value, 4) if m.avg_value else None,
                        "min": round(m.min_value, 4) if m.min_value else None,
                        "max": round(m.max_value, 4) if m.max_value else None,
                    }
                    for m in metrics
                ],
            }

    @staticmethod
    def get_token_usage_summary(hours: int = 24, db: Session | None = None) -> dict:
        """Résumé de la consommation tokens sur les N dernières heures."""
        since = datetime.utcnow() - timedelta(hours=hours)
        with db_session() as session:
            result = (
                session.query(
                    TokenUsageLog.provider,
                    TokenUsageLog.model,
                    func.count(TokenUsageLog.id).label("calls"),
                    func.sum(TokenUsageLog.input_tokens).label("total_input"),
                    func.sum(TokenUsageLog.output_tokens).label("total_output"),
                    func.sum(TokenUsageLog.cost_usd).label("total_cost"),
                )
                .filter(TokenUsageLog.timestamp >= since)
                .group_by(TokenUsageLog.provider, TokenUsageLog.model)
                .all()
            )
            return {
                "period_hours": hours,
                "usage": [
                    {
                        "provider": r.provider,
                        "model": r.model,
                        "calls": r.calls,
                        "input_tokens": r.total_input or 0,
                        "output_tokens": r.total_output or 0,
                        "cost_usd": round(r.total_cost or 0, 6),
                    }
                    for r in result
                ],
            }

    @staticmethod
    def get_audit_log(
        event_type: str | None = None,
        severity: str | None = None,
        hours: int = 24,
        limit: int = 100,
        db: Session | None = None,
    ) -> list[dict]:
        """Récupère les entrées d'audit récentes."""
        since = datetime.utcnow() - timedelta(hours=hours)
        with db_session() as session:
            query = session.query(AuditLog).filter(AuditLog.timestamp >= since)
            if event_type:
                query = query.filter(AuditLog.event_type == event_type)
            if severity:
                query = query.filter(AuditLog.severity == severity)
            entries = query.order_by(AuditLog.timestamp.desc()).limit(limit).all()
            return [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "entity_type": e.entity_type,
                    "entity_id": e.entity_id,
                    "actor": e.actor,
                    "severity": e.severity,
                    "details": json.loads(e.details_json) if e.details_json else None,
                }
                for e in entries
            ]

    @staticmethod
    def get_cost_projection(hours: int = 24, target_provider: str = "openai", target_model: str = "gpt-4o", db: Session | None = None) -> dict:
        """Projette le coût si tous les appels Qwen étaient redirigés vers un modèle cloud cible."""
        since = datetime.utcnow() - timedelta(hours=hours)
        with db_session() as session:
            from trading.sentiment.token_tracker import get_pricing

            result = (
                session.query(
                    func.sum(TokenUsageLog.input_tokens).label("total_input"),
                    func.sum(TokenUsageLog.output_tokens).label("total_output"),
                    func.count(TokenUsageLog.id).label("calls"),
                )
                .filter(TokenUsageLog.timestamp >= since)
                .one()
            )
            total_input = result.total_input or 0
            total_output = result.total_output or 0
            calls = result.calls or 0

            rates = get_pricing(target_provider, target_model)
            projected_cost = (
                total_input * rates["input"] / 1_000_000
                + total_output * rates["output"] / 1_000_000
            )

            return {
                "period_hours": hours,
                "calls": calls,
                "input_tokens": total_input,
                "output_tokens": total_output,
                "target_provider": target_provider,
                "target_model": target_model,
                "projected_cost_usd": round(projected_cost, 6),
                "rates_per_1m": rates,
            }

    # ==================================================================
    # NEW — Monitoring DB methods (TimescaleDB / SQLite)
    # ==================================================================

    @staticmethod
    def log_llm_call(
        function_name: str,
        model: str,
        provider: str,
        backend: str = "local",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: float = 0.0,
        prompt_hash: str | None = None,
        response_hash: str | None = None,
        provider_info: dict | None = None,
        error_message: str | None = None,
        triggered_by: str = "llm_inference",
        portfolio_id: str | None = None,
    ) -> None:
        """Persist a detailed LLM call trace to the monitoring DB."""
        try:
            with monitoring_session() as db:
                entry = LLMCallLog(
                    function_name=function_name,
                    model=model,
                    provider=provider,
                    backend=backend,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    duration_ms=duration_ms,
                    prompt_hash=prompt_hash,
                    response_hash=response_hash,
                    provider_info_json=json.dumps(provider_info, default=str) if provider_info else None,
                    error_message=error_message,
                    triggered_by=triggered_by,
                    portfolio_id=portfolio_id,
                )
                db.add(entry)
        except Exception as e:
            logger.warning("[MonitorService] Failed to log LLM call: %s", e)

    @staticmethod
    def log_message(
        channel: str,
        source: str,
        content_hash: str | None = None,
        metadata_json: str | None = None,
        processed: int = 0,
        processing_time_ms: float | None = None,
    ) -> None:
        """Persist a message trace to the monitoring DB."""
        try:
            with monitoring_session() as db:
                entry = MessageLog(
                    channel=channel,
                    source=source,
                    content_hash=content_hash,
                    metadata_json=metadata_json,
                    processed=processed,
                    processing_time_ms=processing_time_ms,
                )
                db.add(entry)
        except Exception as e:
            logger.warning("[MonitorService] Failed to log message: %s", e)

    @staticmethod
    def log_performance(
        metric_name: str,
        value: float,
        unit: str | None = None,
        tags: dict | None = None,
    ) -> None:
        """Persist a performance snapshot to the monitoring DB."""
        try:
            with monitoring_session() as db:
                entry = PerformanceSnapshot(
                    metric_name=metric_name,
                    value=value,
                    unit=unit,
                    tags_json=json.dumps(tags, default=str) if tags else None,
                )
                db.add(entry)
        except Exception as e:
            logger.warning("[MonitorService] Failed to log performance: %s", e)

    # ------------------------------------------------------------------
    # Queries — optimised for TimescaleDB time_bucket
    # ------------------------------------------------------------------

    @staticmethod
    def get_llm_calls(
        hours: int = 24,
        provider: str | None = None,
        model: str | None = None,
        portfolio_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch recent LLM calls from the monitoring DB."""
        since = datetime.utcnow() - timedelta(hours=hours)
        try:
            with monitoring_session() as db:
                query = db.query(LLMCallLog).filter(LLMCallLog.timestamp >= since)
                if provider:
                    query = query.filter(LLMCallLog.provider == provider)
                if model:
                    query = query.filter(LLMCallLog.model == model)
                if portfolio_id:
                    query = query.filter(LLMCallLog.portfolio_id == portfolio_id)
                rows = query.order_by(LLMCallLog.timestamp.desc()).limit(limit).all()
                return [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat(),
                        "function_name": r.function_name,
                        "model": r.model,
                        "provider": r.provider,
                        "backend": r.backend,
                        "input_tokens": r.input_tokens,
                        "output_tokens": r.output_tokens,
                        "cost_usd": r.cost_usd,
                        "duration_ms": r.duration_ms,
                        "error_message": r.error_message,
                        "triggered_by": r.triggered_by,
                        "portfolio_id": r.portfolio_id,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning("[MonitorService] Failed to query LLM calls: %s", e)
            return []

    @staticmethod
    def get_llm_summary(hours: int = 24) -> dict:
        """Aggregated summary of LLM calls (cost, tokens, duration)."""
        since = datetime.utcnow() - timedelta(hours=hours)
        try:
            with monitoring_session() as db:
                result = (
                    db.query(
                        func.count(LLMCallLog.id).label("calls"),
                        func.sum(LLMCallLog.input_tokens).label("total_input"),
                        func.sum(LLMCallLog.output_tokens).label("total_output"),
                        func.sum(LLMCallLog.cost_usd).label("total_cost"),
                        func.avg(LLMCallLog.duration_ms).label("avg_duration"),
                        func.count(LLMCallLog.error_message).label("errors"),
                    )
                    .filter(LLMCallLog.timestamp >= since)
                    .one()
                )
                return {
                    "period_hours": hours,
                    "calls": result.calls or 0,
                    "input_tokens": result.total_input or 0,
                    "output_tokens": result.total_output or 0,
                    "cost_usd": round(result.total_cost or 0, 6),
                    "avg_duration_ms": round(result.avg_duration or 0, 2),
                    "errors": result.errors or 0,
                }
        except Exception as e:
            logger.warning("[MonitorService] Failed to query LLM summary: %s", e)
            return {}

    @staticmethod
    def get_time_series(
        metric_name: str,
        interval: str = "1 hour",
        hours: int = 24,
    ) -> list[dict]:
        """Return time-bucketed aggregates for Grafana (TimescaleDB optimised).

        Falls back to plain GROUP BY on SQLite.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        try:
            with monitoring_session() as db:
                if IS_TIMESCALEDB:
                    # time_bucket is a TimescaleDB extension
                    sql = text(f"""
                        SELECT
                            time_bucket(:interval, timestamp) AS bucket,
                            COUNT(*) AS cnt,
                            AVG(input_tokens) AS avg_input,
                            AVG(output_tokens) AS avg_output,
                            SUM(cost_usd) AS sum_cost,
                            AVG(duration_ms) AS avg_duration
                        FROM llm_call_log
                        WHERE timestamp >= :since
                        GROUP BY bucket
                        ORDER BY bucket
                    """)
                    result = db.execute(sql, {"interval": interval, "since": since})
                    return [
                        {
                            "bucket": row.bucket.isoformat() if row.bucket else None,
                            "calls": row.cnt,
                            "avg_input_tokens": round(row.avg_input or 0, 2),
                            "avg_output_tokens": round(row.avg_output or 0, 2),
                            "cost_usd": round(row.sum_cost or 0, 6),
                            "avg_duration_ms": round(row.avg_duration or 0, 2),
                        }
                        for row in result
                    ]
                else:
                    # SQLite fallback — plain strftime grouping
                    sql = text("""
                        SELECT
                            strftime('%Y-%m-%d %H:00:00', timestamp) AS bucket,
                            COUNT(*) AS cnt,
                            AVG(input_tokens) AS avg_input,
                            AVG(output_tokens) AS avg_output,
                            SUM(cost_usd) AS sum_cost,
                            AVG(duration_ms) AS avg_duration
                        FROM llm_call_log
                        WHERE timestamp >= :since
                        GROUP BY bucket
                        ORDER BY bucket
                    """)
                    result = db.execute(sql, {"since": since})
                    return [
                        {
                            "bucket": row.bucket,
                            "calls": row.cnt,
                            "avg_input_tokens": round(row.avg_input or 0, 2),
                            "avg_output_tokens": round(row.avg_output or 0, 2),
                            "cost_usd": round(row.sum_cost or 0, 6),
                            "avg_duration_ms": round(row.avg_duration or 0, 2),
                        }
                        for row in result
                    ]
        except Exception as e:
            logger.warning("[MonitorService] Failed to query time series: %s", e)
            return []

    @staticmethod
    def get_messages(
        channel: str | None = None,
        hours: int = 24,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch recent messages from the monitoring DB."""
        since = datetime.utcnow() - timedelta(hours=hours)
        try:
            with monitoring_session() as db:
                query = db.query(MessageLog).filter(MessageLog.timestamp >= since)
                if channel:
                    query = query.filter(MessageLog.channel == channel)
                rows = query.order_by(MessageLog.timestamp.desc()).limit(limit).all()
                return [
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat(),
                        "channel": r.channel,
                        "source": r.source,
                        "content_hash": r.content_hash,
                        "metadata": json.loads(r.metadata_json) if r.metadata_json else None,
                        "processed": r.processed,
                        "processing_time_ms": r.processing_time_ms,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.warning("[MonitorService] Failed to query messages: %s", e)
            return []

    @staticmethod
    def get_message_channels() -> list[dict]:
        """Active channels with message counts (last 24h)."""
        since = datetime.utcnow() - timedelta(hours=24)
        try:
            with monitoring_session() as db:
                result = (
                    db.query(
                        MessageLog.channel,
                        func.count(MessageLog.id).label("cnt"),
                        func.avg(MessageLog.processing_time_ms).label("avg_time"),
                    )
                    .filter(MessageLog.timestamp >= since)
                    .group_by(MessageLog.channel)
                    .all()
                )
                return [
                    {
                        "channel": r.channel,
                        "message_count": r.cnt,
                        "avg_processing_ms": round(r.avg_time or 0, 2),
                    }
                    for r in result
                ]
        except Exception as e:
            logger.warning("[MonitorService] Failed to query channels: %s", e)
            return []


def get_monitor_service() -> MonitorService:
    return MonitorService()
