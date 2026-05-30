"""Stratégie Rotation Sectorielle - rebalancement et stop/take profit."""

import json
import logging

from sqlalchemy.orm import Session

from trading.strategies.base import StrategyBase
from trading.core.models import Trade

logger = logging.getLogger(__name__)


class RotationStrategy(StrategyBase):
    """Vente si -12%, prise de profit partielle +20%, rééquilibrage."""

    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        port = self.get_portfolio(db)
        if port.status != "active":
            return []
        config = json.loads(port.config_json or "{}")
        stop_loss = config.get("stop_loss_pct", -12)
        take_profit = config.get("take_profit_pct", 20)
        take_profit_sell = config.get("take_profit_sell_pct", 50)

        trades: list[Trade] = []

        # 1. Gestion des positions existantes
        for pos in self.get_positions(db):
            if pos.ticker not in prices:
                continue
            price = prices[pos.ticker]
            pnl_pct = (price - pos.avg_entry_price) / pos.avg_entry_price * 100

            if pnl_pct <= stop_loss:
                # Stop loss total
                trade = self.sell(db, pos.ticker, pos.quantity, price)
                trades.append(trade)
                logger.info(f"[ROTATION STOP] {pos.ticker} {pnl_pct:.1f}%")
            elif pnl_pct >= take_profit:
                # Take profit partiel
                sell_qty = pos.quantity * (take_profit_sell / 100)
                trade = self.sell(db, pos.ticker, sell_qty, price)
                trades.append(trade)
                logger.info(f"[ROTATION TP] {pos.ticker} {pnl_pct:.1f}%")

        # 2. Achat sur signaux (si secteur faible / non couvert)
        # Simplifié : achat si sentiment fort et pas de position
        for sig in self.get_signals(db):
            if sig.action not in ("BUY", "STRONG_BUY"):
                continue
            if sig.ticker not in prices:
                continue
            if self.get_position(db, sig.ticker):
                continue
            price = prices[sig.ticker]
            max_trade = port.max_trade_amount or 600
            if port.cash_available < max_trade + port.fee_per_order:
                continue
            qty = max_trade / price
            try:
                trade = self.buy(db, sig.ticker, qty, price, signal_id=sig.id)
                trades.append(trade)
                sig.consumed = 1
                db.commit()
            except ValueError:
                pass

        self.update_position_prices(db, prices)
        self.snapshot_history(db)
        return trades
