"""Routes /portfolios - gestion des portefeuilles."""

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from sqlalchemy import text
from trading.api.dependencies import DbDep, AuthDep
from trading.core.models import Portfolio, PortfolioHistory, Position, Trade, PortfolioRead, PortfolioSummary, Command, CommandCreate
from trading.execution.commands import CommandProcessor

router = APIRouter(prefix="/portfolios", tags=["portfolios"])


@router.get("", response_model=list[PortfolioRead])
def list_portfolios(db: Session = DbDep, _: str = AuthDep):
    return db.query(Portfolio).all()


@router.get("/summary", response_model=list[PortfolioSummary])
def portfolios_summary(db: Session = DbDep, _: str = AuthDep):
    """Vue agrégée performances + positions."""
    rows = db.execute(text("""
        SELECT
            p.id, p.name, p.strategy_type, p.status, p.cash_current,
            COALESCE(SUM(pos.current_value), 0) as positions_value,
            p.cash_current + COALESCE(SUM(pos.current_value), 0) as total_value,
            (p.cash_current + COALESCE(SUM(pos.current_value), 0)) - p.cash_initial as total_pnl,
            ROUND(((p.cash_current + COALESCE(SUM(pos.current_value), 0)) - p.cash_initial) / p.cash_initial * 100, 2) as total_pnl_pct,
            COUNT(DISTINCT pos.ticker) as nb_positions
        FROM portfolios p
        LEFT JOIN positions pos ON pos.portfolio_id = p.id
        GROUP BY p.id
    """))
    return [
        PortfolioSummary(
            id=row.id, name=row.name, strategy_type=row.strategy_type, status=row.status,
            cash_current=row.cash_current, positions_value=row.positions_value,
            total_value=row.total_value, total_pnl=row.total_pnl,
            total_pnl_pct=row.total_pnl_pct, nb_positions=row.nb_positions
        )
        for row in rows
    ]


@router.get("/{portfolio_id}", response_model=PortfolioRead)
def get_portfolio(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    port = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not port:
        raise HTTPException(status_code=404, detail="Portefeuille introuvable")
    return port


@router.get("/{portfolio_id}/positions")
def get_positions(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    return db.query(Position).filter(Position.portfolio_id == portfolio_id).all()


@router.get("/{portfolio_id}/trades")
def get_trades(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    return (
        db.query(Trade)
        .filter(Trade.portfolio_id == portfolio_id)
        .order_by(Trade.executed_at.desc())
        .limit(100)
        .all()
    )


@router.get("/{portfolio_id}/history")
def get_history(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    return (
        db.query(PortfolioHistory)
        .filter(PortfolioHistory.portfolio_id == portfolio_id)
        .order_by(PortfolioHistory.timestamp.desc())
        .limit(200)
        .all()
    )


@router.post("/{portfolio_id}/liquidate")
def liquidate_portfolio(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    """Met en file une liquidation complète."""
    cmd = Command(command_type="LIQUIDATE", portfolio_id=portfolio_id, requested_by="hermes")
    db.add(cmd)
    db.commit()
    # Traitement immédiat (optionnel)
    processor = CommandProcessor(db)
    processor.process_pending()
    return {"status": "liquidating", "command_id": cmd.id}


@router.post("/{portfolio_id}/pause")
def pause_portfolio(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    cmd = Command(command_type="PAUSE", portfolio_id=portfolio_id, requested_by="hermes")
    db.add(cmd)
    db.commit()
    processor = CommandProcessor(db)
    processor.process_pending()
    return {"status": "paused", "command_id": cmd.id}


@router.post("/{portfolio_id}/resume")
def resume_portfolio(portfolio_id: str, db: Session = DbDep, _: str = AuthDep):
    cmd = Command(command_type="RESUME", portfolio_id=portfolio_id, requested_by="hermes")
    db.add(cmd)
    db.commit()
    processor = CommandProcessor(db)
    processor.process_pending()
    return {"status": "active", "command_id": cmd.id}
