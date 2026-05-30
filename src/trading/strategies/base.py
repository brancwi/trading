"""Classe de base pour toutes les stratégies."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime

from sqlalchemy.orm import Session

from trading.core.models import (
    Portfolio,
    Position,
    Trade,
    Signal,
    PortfolioHistory,
)

logger = logging.getLogger(__name__)


class StrategyBase(ABC):
    """Contrat commun pour chaque stratégie."""

    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id

    def get_portfolio(self, db: Session) -> Portfolio:
        port = db.query(Portfolio).filter(Portfolio.id == self.portfolio_id).first()
        if not port:
            raise ValueError(f"Portefeuille {self.portfolio_id} introuvable")
        return port

    def get_position(self, db: Session, ticker: str) -> Position | None:
        return (
            db.query(Position)
            .filter(Position.portfolio_id == self.portfolio_id, Position.ticker == ticker)
            .first()
        )

    def get_positions(self, db: Session) -> list[Position]:
        return db.query(Position).filter(Position.portfolio_id == self.portfolio_id).all()

    def get_signals(self, db: Session, limit: int = 50) -> list[Signal]:
        return (
            db.query(Signal)
            .filter(Signal.consumed == 0)
            .order_by(Signal.timestamp.desc())
            .limit(limit)
            .all()
        )

    def buy(self, db: Session, ticker: str, quantity: float, price: float, signal_id: int | None = None) -> Trade:
        """Exécute un achat."""
        port = self.get_portfolio(db)
        amount = quantity * price
        fees = port.fee_per_order or 1.0
        total_cost = amount + fees
        if port.cash_current < total_cost:
            raise ValueError(f"Cash insuffisant: {port.cash_current} < {total_cost}")
        port.cash_current -= total_cost
        position = self.get_position(db, ticker)
        if position:
            new_qty = position.quantity + quantity
            position.avg_entry_price = (
                position.avg_entry_price * position.quantity + price * quantity
            ) / new_qty
            position.quantity = new_qty
        else:
            position = Position(
                portfolio_id=self.portfolio_id,
                ticker=ticker,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                current_value=amount,
            )
            db.add(position)
        trade = Trade(
            portfolio_id=self.portfolio_id,
            ticker=ticker,
            action="BUY",
            quantity=quantity,
            price=price,
            amount=amount,
            fees=fees,
            signal_id=signal_id,
            strategy_type=port.strategy_type,
        )
        db.add(trade)
        db.commit()
        logger.info(f"[BUY] {self.portfolio_id} {ticker} {quantity} @ {price}")
        return trade

    def sell(self, db: Session, ticker: str, quantity: float, price: float, signal_id: int | None = None) -> Trade:
        """Exécute une vente."""
        port = self.get_portfolio(db)
        position = self.get_position(db, ticker)
        if not position or position.quantity < quantity:
            raise ValueError(f"Position insuffisante pour {ticker}")
        amount = quantity * price
        fees = port.fee_per_order or 1.0
        realized = (price - position.avg_entry_price) * quantity - fees
        port.cash_current += amount - fees
        position.quantity -= quantity
        if position.quantity <= 0:
            db.delete(position)
        trade = Trade(
            portfolio_id=self.portfolio_id,
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price=price,
            amount=amount,
            fees=fees,
            realized_pnl=realized,
            signal_id=signal_id,
            strategy_type=port.strategy_type,
        )
        db.add(trade)
        db.commit()
        logger.info(f"[SELL] {self.portfolio_id} {ticker} {quantity} @ {price} PnL={realized:.2f}")
        return trade

    def update_position_prices(self, db: Session, prices: dict[str, float]) -> None:
        """Met à jour les prix courants des positions."""
        for pos in self.get_positions(db):
            if pos.ticker in prices:
                pos.current_price = prices[pos.ticker]
                pos.current_value = pos.quantity * pos.current_price
                pos.unrealized_pnl = pos.current_value - pos.quantity * pos.avg_entry_price
                if pos.avg_entry_price > 0:
                    pos.unrealized_pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price * 100
        db.commit()

    def snapshot_history(self, db: Session) -> PortfolioHistory:
        """Sauvegarde l'état courant dans l'historique."""
        port = self.get_portfolio(db)
        positions_value = sum(p.current_value or 0 for p in self.get_positions(db))
        total = port.cash_current + positions_value
        pnl = total - port.cash_initial
        pnl_pct = (pnl / port.cash_initial * 100) if port.cash_initial else 0
        snap = PortfolioHistory(
            portfolio_id=self.portfolio_id,
            cash=port.cash_current,
            positions_value=positions_value,
            total_value=total,
            total_pnl=pnl,
            total_pnl_pct=pnl_pct,
        )
        db.add(snap)
        db.commit()
        return snap

    def liquidate(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        """Vend toutes les positions."""
        trades: list[Trade] = []
        for pos in self.get_positions(db):
            price = prices.get(pos.ticker, pos.current_price or pos.avg_entry_price)
            trade = self.sell(db, pos.ticker, pos.quantity, price)
            trades.append(trade)
        port = self.get_portfolio(db)
        port.status = "liquidated"
        db.commit()
        return trades

    @abstractmethod
    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        """Point d'entrée principal : exécute la stratégie sur les signaux disponibles."""
        ...
