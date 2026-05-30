"""Routes API pour le monitoring, l'audit et le token usage.

Endpoints accessibles par Hermes via MCP (HTTP) pour supervision du système.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, Query

from trading.api.dependencies import verify_api_key
from trading.monitoring.service import MonitorService

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
monitor = MonitorService()


@router.get("/metrics", dependencies=[Depends(verify_api_key)])
def get_metrics(
    hours: int = Query(24, ge=1, le=720),
    metric_name: str | None = Query(None),
    source: str | None = Query(None),
):
    """Récupère les métriques time-series sur une période donnée.

    Exemple: /monitoring/metrics?hours=24&metric_name=inference_latency
    """
    summary = monitor.get_metrics_summary(hours=hours)
    if metric_name:
        summary["metrics"] = [m for m in summary["metrics"] if m["name"] == metric_name]
    if source:
        summary["metrics"] = [m for m in summary["metrics"] if m["source"] == source]
    return summary


@router.get("/token-usage", dependencies=[Depends(verify_api_key)])
def get_token_usage(hours: int = Query(24, ge=1, le=720)):
    """Résumé de la consommation tokens (local + cloud) sur N heures."""
    return monitor.get_token_usage_summary(hours=hours)


@router.get("/audit", dependencies=[Depends(verify_api_key)])
def get_audit(
    hours: int = Query(24, ge=1, le=720),
    event_type: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Journal d'audit filtrable par type d'événement et sévérité."""
    return {
        "audit_entries": monitor.get_audit_log(
            event_type=event_type,
            severity=severity,
            hours=hours,
            limit=limit,
        )
    }


@router.get("/cost-projection", dependencies=[Depends(verify_api_key)])
def get_cost_projection(
    hours: int = Query(24, ge=1, le=720),
    provider: str = Query("openai"),
    model: str = Query("gpt-4o"),
):
    """Projette le coût si tous les appels Qwen étaient redirigés vers un modèle cloud cible.

    Exemple: /monitoring/cost-projection?provider=openai&model=gpt-4o-mini
    """
    return monitor.get_cost_projection(
        hours=hours,
        target_provider=provider,
        target_model=model,
    )


@router.get("/summary", dependencies=[Depends(verify_api_key)])
def get_system_summary(
    hours: int = Query(24, ge=1, le=720),
):
    """Résumé complet du système (métriques + tokens + audit count)."""
    metrics = monitor.get_metrics_summary(hours=hours)
    tokens = monitor.get_token_usage_summary(hours=hours)
    audit_count = len(monitor.get_audit_log(hours=hours, limit=1000))

    total_cost = sum(u.get("cost_usd", 0) for u in tokens.get("usage", []))
    total_calls = sum(u.get("calls", 0) for u in tokens.get("usage", []))

    return {
        "period_hours": hours,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics_count": len(metrics.get("metrics", [])),
        "audit_entries": audit_count,
        "token_usage": {
            "total_calls": total_calls,
            "total_cost_usd": round(total_cost, 6),
            "breakdown": tokens.get("usage", []),
        },
        "top_metrics": metrics.get("metrics", [])[:10],
    }


@router.post("/metric", dependencies=[Depends(verify_api_key)])
def push_metric(
    name: str,
    value: float,
    unit: str | None = None,
    source: str = "hermes",
    tags: dict | None = None,
):
    """Permet à Hermes de pousser une métrique personnalisée."""
    monitor.record_metric(name=name, value=value, unit=unit, source=source, tags=tags)
    return {"status": "recorded", "metric": name, "value": value}
