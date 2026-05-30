"""Route /status - heartbeat et synthèse globale."""

from datetime import datetime
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from trading.api.dependencies import DbDep, AuthDep
from trading.core.models import Portfolio, PortfolioHistory, Signal, Command, StatusRead

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
