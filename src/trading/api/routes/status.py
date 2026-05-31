"""Route /status - heartbeat et synthèse globale."""

import socket
from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from trading.api.dependencies import DbDep, AuthDep
from trading.core.config import get_settings
from trading.core.models import Portfolio, PortfolioHistory, Signal, Command, StatusRead, AuditLog

router = APIRouter(prefix="/status", tags=["status"])


@router.get("", response_model=StatusRead)
def get_status(db: Session = DbDep, _: str = AuthDep):
    """Retourne l'état global du système."""
    # Dernier snapshot par portefeuille
    subq = (
        db.query(
            PortfolioHistory.portfolio_id,
            func.max(PortfolioHistory.timestamp).label("last_ts")
        )
        .group_by(PortfolioHistory.portfolio_id)
        .subquery()
    )
    last_hist = (
        db.query(PortfolioHistory)
        .join(subq, (PortfolioHistory.portfolio_id == subq.c.portfolio_id) & (PortfolioHistory.timestamp == subq.c.last_ts))
        .all()
    )
    portfolios = {h.portfolio_id: round(h.total_value, 2) for h in last_hist}
    # Fallback sur cash si pas d'historique
    for p in db.query(Portfolio).all():
        if p.id not in portfolios:
            portfolios[p.id] = round(p.cash_current, 2)

    last_run = max((h.timestamp for h in last_hist), default=None)
    pending = db.query(Command).filter(Command.status == "pending").count()
    unread = db.query(Signal).filter(Signal.consumed == 0).count()

    return StatusRead(
        pipeline="running",
        last_run=last_run,
        portfolios=portfolios,
        pending_commands=pending,
        unread_signals=unread,
    )


@router.get("/diagnostic")
def get_diagnostic(db: Session = DbDep, _: str = AuthDep):
    """Diagnostic système complet — détecte les blocages de config, DB, services."""
    settings = get_settings()
    db_url = settings.database_url
    db_type = "postgresql" if db_url.startswith("postgresql") else "sqlite" if db_url.startswith("sqlite") else "unknown"

    # DB version
    try:
        if db_type == "postgresql":
            db_version = db.execute(text("SELECT version()")).scalar()
        else:
            db_version = db.execute(text("SELECT sqlite_version()")).scalar()
    except Exception as e:
        db_version = f"error: {e}"

    # Service health checks
    services = {
        "api": {"status": "up", "version": "1.0.0"},
        "postgres": {"status": "up" if db_type == "postgresql" else "n/a"},
        "mcp_server": {"status": _check_port(8001)},
        "prefect_server": {"status": _check_port(4200)},
    }

    # Recent errors
    since = datetime.utcnow() - __import__('datetime', fromlist=['timedelta']).timedelta(hours=1)
    recent_errors = (
        db.query(AuditLog)
        .filter(AuditLog.severity.in_(["error", "critical"]))
        .filter(AuditLog.timestamp >= since)
        .count()
    )

    return {
        "environment": settings.environment,
        "database": {
            "type": db_type,
            "version": str(db_version) if db_version else "unknown",
            "url_masked": _mask_db_url(db_url),
        },
        "services": services,
        "config_summary": {
            "ml_device": settings.ml_device,
            "ml_models": {
                "roberta": settings.ml_model_roberta,
                "modern": settings.ml_model_modern,
                "qwen": settings.ml_model_qwen,
            },
            "market_hours": f"{settings.market_open_hour}h-{settings.market_close_hour}h",
            "pipeline_interval_minutes": settings.pipeline_interval_minutes,
        },
        "recent_errors_last_1h": recent_errors,
        "generated_at": datetime.utcnow().isoformat(),
    }


def _check_port(port: int, host: str = "127.0.0.1") -> str:
    try:
        with socket.create_connection((host, port), timeout=1):
            return "up"
    except (socket.timeout, ConnectionRefusedError, OSError):
        return "down"


def _mask_db_url(url: str) -> str:
    """Masque le mot de passe dans une URL de DB."""
    try:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(url)
        if parsed.password:
            netloc = parsed.netloc.replace(f":{parsed.password}@", ":***@")
            parsed = parsed._replace(netloc=netloc)
        return urlunparse(parsed)
    except Exception:
        return "masked"
