"""MonitorService — collecte et stockage des métriques, audit, token usage."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from trading.core.database import db_session
from trading.core.models import AuditLog, MonitoringMetric, TokenUsageLog

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


def get_monitor_service() -> MonitorService:
    return MonitorService()
