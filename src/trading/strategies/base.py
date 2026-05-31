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

    # ------------------------------------------------------------------
    # FX helper — conversion automatique USD → devise du portfolio
    # ------------------------------------------------------------------

    def _fx_rate(self, port: Portfolio) -> float:
        """Retourne le taux de conversion USD → devise du portfolio.

        Les prix de marché sont en USD.  Si le portfolio est en EUR,
        on applique le taux fx_eur_usd (ex: 1.08 → 1 USD = 1/1.08 EUR).
        """
        if port.base_currency == "EUR":
            from trading.core.config import get_settings
            return get_settings().fx_eur_usd
        return 1.0

    def _to_portfolio_currency(self, amount_usd: float, port: Portfolio) -> float:
        """Convertit un montant USD dans la devise du portfolio."""
        fx = self._fx_rate(port)
        return amount_usd / fx if fx else amount_usd

    def _to_usd(self, amount_portfolio: float, port: Portfolio) -> float:
        """Convertit un montant de la devise du portfolio en USD."""
        fx = self._fx_rate(port)
        return amount_portfolio * fx

    # ------------------------------------------------------------------
    # Ordres
    # ------------------------------------------------------------------

    def buy(self, db: Session, ticker: str, quantity: float, price: float, signal_id: int | None = None) -> Trade:
        """Exécute un achat.

        *price* est le prix de marché en USD.  Le cash du portfolio est
        débité dans sa devise native (EUR ou USD) après conversion FX.
        """
        port = self.get_portfolio(db)
        fx = self._fx_rate(port)
        amount_usd = quantity * price
        amount_portfolio = amount_usd / fx if fx else amount_usd
        fees = port.fee_per_order or 1.0
        total_cost = amount_portfolio + fees
        if port.cash_available < total_cost:
            raise ValueError(
                f"Cash disponible insuffisant: {port.cash_available:.2f} {port.base_currency} < "
                f"{total_cost:.2f} {port.base_currency} (réservé: {port.reserved_cash:.2f})"
            )
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
                current_value=amount_portfolio,
            )
            db.add(position)
        trade = Trade(
            portfolio_id=self.portfolio_id,
            ticker=ticker,
            action="BUY",
            quantity=quantity,
            price=price,
            amount=amount_usd,
            fees=fees,
            signal_id=signal_id,
            strategy_type=port.strategy_type,
        )
        db.add(trade)
        db.commit()
        logger.info(
            f"[BUY] {self.portfolio_id} {ticker} {quantity:.4f} @ ${price:.2f} "
            f"(≈ {amount_portfolio:.2f} {port.base_currency})"
        )
        return trade

    def sell(self, db: Session, ticker: str, quantity: float, price: float, signal_id: int | None = None) -> Trade:
        """Exécute une vente.

        *price* est le prix de marché en USD.  Le cash du portfolio est
        crédité dans sa devise native après conversion FX.
        """
        port = self.get_portfolio(db)
        fx = self._fx_rate(port)
        position = self.get_position(db, ticker)
        if not position or position.quantity < quantity:
            raise ValueError(f"Position insuffisante pour {ticker}")
        amount_usd = quantity * price
        amount_portfolio = amount_usd / fx if fx else amount_usd
        fees = port.fee_per_order or 1.0
        entry_value_portfolio = position.avg_entry_price * quantity / fx if fx else position.avg_entry_price * quantity
        realized = amount_portfolio - entry_value_portfolio - fees
        port.cash_current += amount_portfolio - fees
        position.quantity -= quantity
        if position.quantity <= 0:
            db.delete(position)
        trade = Trade(
            portfolio_id=self.portfolio_id,
            ticker=ticker,
            action="SELL",
            quantity=quantity,
            price=price,
            amount=amount_usd,
            fees=fees,
            realized_pnl=realized,
            signal_id=signal_id,
            strategy_type=port.strategy_type,
        )
        db.add(trade)
        db.commit()
        logger.info(
            f"[SELL] {self.portfolio_id} {ticker} {quantity:.4f} @ ${price:.2f} "
            f"PnL={realized:.2f} {port.base_currency}"
        )
        return trade

    def update_position_prices(self, db: Session, prices: dict[str, float]) -> None:
        """Met à jour les prix courants des positions (conversion FX incluse)."""
        port = self.get_portfolio(db)
        fx = self._fx_rate(port)
        for pos in self.get_positions(db):
            if pos.ticker in prices:
                pos.current_price = prices[pos.ticker]
                pos.current_value = pos.quantity * pos.current_price / fx if fx else pos.quantity * pos.current_price
                entry_value = pos.quantity * pos.avg_entry_price / fx if fx else pos.quantity * pos.avg_entry_price
                pos.unrealized_pnl = pos.current_value - entry_value
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
