"""Stratégie Ninja - Opportuniste mixte growth/value/emerging."""

import json
import logging

from sqlalchemy.orm import Session

from trading.strategies.base import StrategyBase
from trading.core.models import Trade

logger = logging.getLogger(__name__)


class NinjaStrategy(StrategyBase):
    """Montants plus petits, diversification secteur obligatoire."""

    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        port = self.get_portfolio(db)
        if port.status != "active":
            return []
        config = json.loads(port.config_json or "{}")
        cash_min = config.get("cash_min", 50)
        min_sectors = config.get("min_sectors", 3)
        max_trade = port.max_trade_amount or 150

        trades: list[Trade] = []
        positions = self.get_positions(db)
        current_sectors = {p.sector for p in positions if p.sector}

        for sig in self.get_signals(db):
            if sig.action not in ("BUY", "STRONG_BUY", "SELL"):
                continue
            if sig.ticker not in prices:
                continue
            price = prices[sig.ticker]

            if sig.action in ("BUY", "STRONG_BUY"):
                # Vérification diversification
                if len(current_sectors) < min_sectors and not self.get_position(db, sig.ticker):
                    pass  # On autorise l'achat pour diversifier
                elif self.get_position(db, sig.ticker):
                    pass
                else:
                    continue  # Déjà assez diversifié et pas de position

                trade_amount = min(max_trade, port.cash_available - cash_min)
                if trade_amount < cash_min:
                    continue
                qty = trade_amount / price
                try:
                    trade = self.buy(db, sig.ticker, qty, price, signal_id=sig.id)
                    trades.append(trade)
                    sig.consumed = 1
                    db.commit()
                except ValueError:
                    pass
            elif sig.action == "SELL":
                pos = self.get_position(db, sig.ticker)
                if pos:
                    try:
                        trade = self.sell(db, sig.ticker, pos.quantity, price, signal_id=sig.id)
                        trades.append(trade)
                        sig.consumed = 1
                        db.commit()
                    except ValueError:
                        pass

        self.update_position_prices(db, prices)
        self.snapshot_history(db)
        return trades
