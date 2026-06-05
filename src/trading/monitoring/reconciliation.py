"""Service de réconciliation P&L réel vs simulé (implementation shortfall)."""

import logging
from typing import Sequence

from sqlalchemy.orm import Session
from sqlalchemy import func

from trading.core.models import (
    Trade,
    Signal,
    Position,
    Portfolio,
    PnLReconciliation,
)

logger = logging.getLogger(__name__)


class PnLReconciliationService:
    """Calcule et persiste l'écart entre P&L théorique et P&L réel."""

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------
    # Calcul unitaire
    # ------------------------------------------------------------------

    def reconcile_trade(self, trade: Trade) -> PnLReconciliation | None:
        """Calcule la réconciliation pour un Trade donné.

        Retourne None si le trade n'a pas de signal_id ou si le signal
        n'a pas de price_at_signal.
        """
        if not trade.signal_id:
            logger.debug(f"Trade {trade.id} sans signal_id — skip reconciliation")
            return None

        signal = self.db.query(Signal).filter(Signal.id == trade.signal_id).first()
        if not signal:
            logger.warning(f"Signal {trade.signal_id} introuvable pour trade {trade.id}")
            return None

        expected_price = signal.price_at_signal
        if expected_price is None:
            logger.debug(f"Signal {signal.id} sans price_at_signal — skip")
            return None

        portfolio = self.db.query(Portfolio).filter(Portfolio.id == trade.portfolio_id).first()
        if not portfolio:
            logger.warning(f"Portfolio {trade.portfolio_id} introuvable")
            return None

        fx = self._fx_rate(portfolio)

        executed_price = trade.price
        quantity = trade.quantity
        expected_fees = portfolio.fee_per_order or 1.0
        executed_fees = trade.fees

        # --- Slippage (prix) ---
        slippage = executed_price - expected_price
        slippage_pct = (slippage / expected_price * 100) if expected_price else 0.0

        # --- P&L attendu (simulé) ---
        # On recalcule le P&L comme dans StrategyBase.sell() mais avec expected_price
        # Il faut retrouver le avg_entry_price de la position au moment du trade.
        # Comme la position peut avoir été supprimée après vente totale, on
        # récupère l'historique des trades d'achat pour ce ticker.
        avg_entry = self._avg_entry_price(trade)

        if trade.action == "SELL" and avg_entry is not None:
            expected_amount_portfolio = expected_price * quantity / fx if fx else expected_price * quantity
            entry_value_portfolio = avg_entry * quantity / fx if fx else avg_entry * quantity
            expected_pnl = expected_amount_portfolio - entry_value_portfolio - expected_fees
            realized_pnl = trade.realized_pnl or 0.0
        elif trade.action == "BUY":
            # Pour un BUY, le P&L "réalisé" n'existe pas encore.
            # On stocke un expected cost vs executed cost.
            expected_cost_portfolio = expected_price * quantity / fx if fx else expected_price * quantity
            executed_cost_portfolio = executed_price * quantity / fx if fx else executed_price * quantity
            expected_pnl = -(expected_cost_portfolio + expected_fees)  # négatif = coût
            realized_pnl = -(executed_cost_portfolio + executed_fees)
            slippage = executed_cost_portfolio - expected_cost_portfolio  # coût supplémentaire
            slippage_pct = (slippage / expected_cost_portfolio * 100) if expected_cost_portfolio else 0.0
        else:
            expected_pnl = trade.realized_pnl or 0.0
            realized_pnl = expected_pnl

        implementation_shortfall = expected_pnl - realized_pnl
        denom = abs(expected_pnl) if expected_pnl else abs(realized_pnl) if realized_pnl else 1.0
        implementation_shortfall_pct = (implementation_shortfall / denom * 100) if denom else 0.0

        rec = PnLReconciliation(
            trade_id=trade.id,
            signal_id=signal.id,
            portfolio_id=trade.portfolio_id,
            ticker=trade.ticker,
            strategy_type=trade.strategy_type or portfolio.strategy_type,
            expected_price=expected_price,
            executed_price=executed_price,
            slippage=slippage,
            slippage_pct=slippage_pct,
            expected_quantity=quantity,
            executed_quantity=quantity,
            expected_fees=expected_fees,
            executed_fees=executed_fees,
            expected_pnl=expected_pnl,
            realized_pnl=realized_pnl,
            implementation_shortfall=implementation_shortfall,
            implementation_shortfall_pct=implementation_shortfall_pct,
        )

        # Évite les doublons — supprime l'ancienne entrée pour ce trade
        existing = (
            self.db.query(PnLReconciliation)
            .filter(PnLReconciliation.trade_id == trade.id)
            .first()
        )
        if existing:
            self.db.delete(existing)
            self.db.flush()

        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)

        logger.info(
            f"[Reconcile] {trade.portfolio_id} {trade.ticker} "
            f"IS={implementation_shortfall:.2f} ({implementation_shortfall_pct:.2f}%) "
            f"slippage={slippage:.4f}"
        )
        return rec

    # ------------------------------------------------------------------
    # Batch / historique
    # ------------------------------------------------------------------

    def reconcile_portfolio(self, portfolio_id: str) -> int:
        """Recalcule la réconciliation pour tous les trades d'un portfolio."""
        trades = (
            self.db.query(Trade)
            .filter(
                Trade.portfolio_id == portfolio_id,
                Trade.signal_id.isnot(None),
            )
            .all()
        )
        count = 0
        for trade in trades:
            rec = self.reconcile_trade(trade)
            if rec:
                count += 1
        logger.info(f"[Reconcile] {portfolio_id}: {count}/{len(trades)} trades reconciliés")
        return count

    def reconcile_all(self) -> int:
        """Recalcule la réconciliation pour tous les trades."""
        trades = (
            self.db.query(Trade)
            .filter(Trade.signal_id.isnot(None))
            .all()
        )
        count = 0
        for trade in trades:
            rec = self.reconcile_trade(trade)
            if rec:
                count += 1
        logger.info(f"[Reconcile] Global: {count}/{len(trades)} trades reconciliés")
        return count

    # ------------------------------------------------------------------
    # Agrégations (tracking error)
    # ------------------------------------------------------------------

    def tracking_error_summary(self, portfolio_id: str | None = None) -> list[dict]:
        """Retourne un résumé agrégé par portfolio."""
        q = self.db.query(
            PnLReconciliation.portfolio_id,
            PnLReconciliation.strategy_type,
            func.count(PnLReconciliation.id).label("total_trades"),
            func.avg(PnLReconciliation.slippage).label("avg_slippage"),
            func.avg(PnLReconciliation.slippage_pct).label("avg_slippage_pct"),
            func.avg(PnLReconciliation.implementation_shortfall).label("avg_implementation_shortfall"),
            func.avg(PnLReconciliation.implementation_shortfall_pct).label("avg_implementation_shortfall_pct"),
            func.sum(PnLReconciliation.expected_pnl).label("total_expected_pnl"),
            func.sum(PnLReconciliation.realized_pnl).label("total_realized_pnl"),
            func.sum(PnLReconciliation.implementation_shortfall).label("total_tracking_error"),
            func.min(PnLReconciliation.slippage_pct).label("worst_slippage_pct"),
            func.max(PnLReconciliation.slippage_pct).label("best_slippage_pct"),
        )
        if portfolio_id:
            q = q.filter(PnLReconciliation.portfolio_id == portfolio_id)
        q = q.group_by(PnLReconciliation.portfolio_id, PnLReconciliation.strategy_type)

        results = []
        for row in q.all():
            results.append({
                "portfolio_id": row.portfolio_id,
                "strategy_type": row.strategy_type,
                "total_trades": row.total_trades,
                "avg_slippage": round(row.avg_slippage or 0, 4),
                "avg_slippage_pct": round(row.avg_slippage_pct or 0, 4),
                "avg_implementation_shortfall": round(row.avg_implementation_shortfall or 0, 2),
                "avg_implementation_shortfall_pct": round(row.avg_implementation_shortfall_pct or 0, 4),
                "total_expected_pnl": round(row.total_expected_pnl or 0, 2),
                "total_realized_pnl": round(row.total_realized_pnl or 0, 2),
                "total_tracking_error": round(row.total_tracking_error or 0, 2),
                "worst_trade_slippage_pct": round(row.worst_slippage_pct or 0, 4),
                "best_trade_slippage_pct": round(row.best_slippage_pct or 0, 4),
            })
        return results

    def ticker_breakdown(self, portfolio_id: str) -> list[dict]:
        """Breakdown par ticker pour un portfolio donné."""
        rows = (
            self.db.query(
                PnLReconciliation.ticker,
                func.count(PnLReconciliation.id).label("nb_trades"),
                func.avg(PnLReconciliation.implementation_shortfall).label("avg_is"),
                func.avg(PnLReconciliation.implementation_shortfall_pct).label("avg_is_pct"),
                func.sum(PnLReconciliation.implementation_shortfall).label("total_is"),
            )
            .filter(PnLReconciliation.portfolio_id == portfolio_id)
            .group_by(PnLReconciliation.ticker)
            .all()
        )
        return [
            {
                "ticker": r.ticker,
                "nb_trades": r.nb_trades,
                "avg_implementation_shortfall": round(r.avg_is or 0, 2),
                "avg_implementation_shortfall_pct": round(r.avg_is_pct or 0, 4),
                "total_implementation_shortfall": round(r.total_is or 0, 2),
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Helpers privés
    # ------------------------------------------------------------------

    def _fx_rate(self, portfolio: Portfolio) -> float:
        if portfolio.base_currency == "EUR":
            from trading.utils.fx import get_fx_eur_usd
            return get_fx_eur_usd() or 1.0
        return 1.0

    def _avg_entry_price(self, trade: Trade) -> float | None:
        """Retrouve le prix moyen d'entrée pour une position au moment du trade.

        Si la position existe encore, on utilise avg_entry_price.
        Sinon on recalcule à partir des trades BUY précédents pour ce ticker.
        """
        position = (
            self.db.query(Position)
            .filter(
                Position.portfolio_id == trade.portfolio_id,
                Position.ticker == trade.ticker,
            )
            .first()
        )
        if position:
            return position.avg_entry_price

        # Fallback : recalcule depuis l'historique des achats
        buys = (
            self.db.query(Trade)
            .filter(
                Trade.portfolio_id == trade.portfolio_id,
                Trade.ticker == trade.ticker,
                Trade.action == "BUY",
                Trade.executed_at <= trade.executed_at,
            )
            .order_by(Trade.executed_at.asc())
            .all()
        )
        if not buys:
            return None

        total_qty = sum(b.quantity for b in buys)
        total_cost = sum(b.quantity * b.price for b in buys)
        return total_cost / total_qty if total_qty else None
