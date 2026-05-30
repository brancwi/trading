"""Execution Engine - orchestre les stratégies sur les signaux."""

import logging
from typing import Type

from sqlalchemy.orm import Session

from trading.core.models import Trade, Portfolio
from trading.strategies.base import StrategyBase
from trading.strategies.simulation import SimulationStrategy
from trading.strategies.rotation import RotationStrategy
from trading.strategies.ninja import NinjaStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP: dict[str, Type[StrategyBase]] = {
    "simulation": SimulationStrategy,
    "rotation": RotationStrategy,
    "ninja": NinjaStrategy,
}


class ExecutionEngine:
    """Route les signaux vers les bonnes stratégies et exécute."""

    def __init__(self, db: Session):
        self.db = db

    def run_all(self, prices: dict[str, float]) -> dict[str, list[Trade]]:
        """Exécute toutes les stratégies actives."""
        results: dict[str, list[Trade]] = {}
        portfolios = self.db.query(Portfolio).filter(Portfolio.status.in_(["active", "paused"])).all()
        for port in portfolios:
            strategy_cls = STRATEGY_MAP.get(port.strategy_type)
            if not strategy_cls:
                logger.warning(f"Stratégie inconnue: {port.strategy_type}")
                continue
            strategy = strategy_cls(port.id)
            try:
                trades = strategy.run(self.db, prices)
                results[port.id] = trades
                logger.info(f"{port.id}: {len(trades)} trades")
            except Exception as e:
                logger.exception(f"Erreur stratégie {port.id}: {e}")
                results[port.id] = []
        return results

    def run_strategy(self, portfolio_id: str, prices: dict[str, float]) -> list[Trade]:
        """Exécute une stratégie spécifique."""
        port = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not port:
            raise ValueError(f"Portefeuille {portfolio_id} introuvable")
        strategy_cls = STRATEGY_MAP.get(port.strategy_type)
        if not strategy_cls:
            raise ValueError(f"Stratégie {port.strategy_type} inconnue")
        strategy = strategy_cls(port.id)
        return strategy.run(self.db, prices)
