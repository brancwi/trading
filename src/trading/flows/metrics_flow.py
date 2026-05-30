"""Flow indépendant — calcul des métriques et snapshots."""

import logging

from prefect import flow, task

from trading.core.database import db_session
from trading.core.models import Portfolio, Position, PortfolioHistory
from trading.strategies.base import StrategyBase
from trading.strategies.simulation import SimulationStrategy
from trading.strategies.rotation import RotationStrategy
from trading.strategies.ninja import NinjaStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "simulation": SimulationStrategy,
    "rotation": RotationStrategy,
    "ninja": NinjaStrategy,
}


@task
def snapshot_all_portfolios() -> list[dict]:
    """Sauvegarde l'état de tous les portefeuilles actifs."""
    results = []
    with db_session() as db:
        for port in db.query(Portfolio).filter(Portfolio.status.in_(["active", "paused"])).all():
            strategy_cls = STRATEGY_MAP.get(port.strategy_type)
            if not strategy_cls:
                continue
            strategy = strategy_cls(port.id)
            snap = strategy.snapshot_history(db)
            results.append({
                "portfolio_id": port.id,
                "total_value": snap.total_value,
                "pnl_pct": snap.total_pnl_pct,
            })
    return results


@flow(name="metrics_flow", log_prints=True)
def metrics_flow() -> dict:
    """Flow de métriques — déclenché périodiquement ou par portfolio.updated."""
    snaps = snapshot_all_portfolios()
    logger.info(f"Metrics: {len(snaps)} portfolios snapshotted")
    return {"snapshots": snaps}


if __name__ == "__main__":
    metrics_flow()
