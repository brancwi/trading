"""Flow indépendant — exécution d'une stratégie sur un portefeuille donné."""

import logging
from typing import Any

from prefect import flow, task

from trading.core.database import db_session
from trading.core.models import MarketData, Portfolio
from trading.execution.engine import ExecutionEngine
from trading.strategies.base import StrategyBase
from trading.strategies.simulation import SimulationStrategy
from trading.strategies.rotation import RotationStrategy
from trading.strategies.ninja import NinjaStrategy
from trading.events.emitters import emit_portfolio_updated, emit_trade_executed

logger = logging.getLogger(__name__)

STRATEGY_MAP: dict[str, type[StrategyBase]] = {
    "simulation": SimulationStrategy,
    "rotation": RotationStrategy,
    "ninja": NinjaStrategy,
}


@task
def get_latest_prices() -> dict[str, float]:
    """Récupère les derniers prix connus en DB."""
    with db_session() as db:
        # Sous-requête pour le dernier prix par ticker
        from sqlalchemy import func
        subq = (
            db.query(
                MarketData.ticker,
                func.max(MarketData.timestamp).label("last_ts")
            )
            .group_by(MarketData.ticker)
            .subquery()
        )
        rows = (
            db.query(MarketData)
            .join(subq, (MarketData.ticker == subq.c.ticker) & (MarketData.timestamp == subq.c.last_ts))
            .all()
        )
        return {r.ticker: r.price for r in rows}


@task
def execute_strategy(portfolio_id: str, prices: dict[str, float]) -> list[dict[str, Any]]:
    """Exécute la stratégie d'un portefeuille donné."""
    with db_session() as db:
        port = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not port:
            raise ValueError(f"Portefeuille {portfolio_id} introuvable")
        if port.status != "active":
            logger.info(f"Portfolio {portfolio_id} is {port.status} — skipping")
            return []
        strategy_cls = STRATEGY_MAP.get(port.strategy_type)
        if not strategy_cls:
            raise ValueError(f"Stratégie inconnue: {port.strategy_type}")
        engine = ExecutionEngine(db)
        trades = engine.run_strategy(portfolio_id, prices)
        results = []
        for trade in trades:
            results.append({
                "trade_id": trade.id,
                "ticker": trade.ticker,
                "action": trade.action,
                "amount": trade.amount,
            })
            emit_trade_executed(trade.id, portfolio_id, trade.ticker, trade.action, trade.amount)
        # Snapshot + event
        strategy = strategy_cls(portfolio_id)
        strategy.snapshot_history(db)
        # Calcul value approx
        positions_value = sum(p.current_value or 0 for p in strategy.get_positions(db))
        total = port.cash_current + positions_value
        pnl = total - port.cash_initial
        emit_portfolio_updated(portfolio_id, total, pnl)
        return results


@flow(name="strategy_execution_flow", log_prints=True)
def strategy_execution_flow(portfolio_id: str | None = None) -> dict:
    """Flow d'exécution stratégique — déclenché par signal ou prix."""
    if portfolio_id is None:
        portfolio_id = "simulation"
    prices = get_latest_prices()
    if not prices:
        logger.warning("Aucun prix disponible — skip")
        return {"trades": [], "portfolio_id": portfolio_id}
    trades = execute_strategy(portfolio_id, prices)
    logger.info(f"Strategy {portfolio_id}: {len(trades)} trades")
    return {"trades": trades, "portfolio_id": portfolio_id}


if __name__ == "__main__":
    strategy_execution_flow()
