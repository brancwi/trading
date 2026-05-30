"""Stratégie Simulation - Day Trading sur signaux news."""

import json
import logging

from sqlalchemy.orm import Session

from trading.strategies.base import StrategyBase
from trading.core.models import Trade, Signal
from trading.ml.signal_model import SignalModel

logger = logging.getLogger(__name__)


class SimulationStrategy(StrategyBase):
    """Achat sur sentiment fort, filtré par SignalModel ML si entraîné."""

    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        port = self.get_portfolio(db)
        if port.status != "active":
            logger.info(f"Simulation {port.status} → skip")
            return []
        config = json.loads(port.config_json or "{}")
        threshold = config.get("sentiment_threshold", 0.5)
        cash_min = config.get("cash_min", 100)
        max_trade = port.max_trade_amount or 500

        # Chargement lazy du SignalModel
        signal_model = SignalModel()
        use_ml = signal_model.trained
        if use_ml:
            logger.info("[Simulation] SignalModel entraîné — filtrage ML actif")

        trades: list[Trade] = []
        signals = self.get_signals(db)

        for sig in signals:
            if sig.action not in ("BUY", "STRONG_BUY"):
                continue
            if sig.ticker not in prices:
                continue
            price = prices[sig.ticker]
            if price <= 0:
                continue

            # Filtrage ML : si le modèle prédit HOLD/SELL, on ignore
            if use_ml:
                ml_pred = signal_model.predict(
                    ticker=sig.ticker,
                    sentiment_combined=sig.sentiment,
                    sentiment_confidence=sig.confidence,
                )
                if ml_pred["action"] in ("HOLD", "SELL"):
                    logger.info(
                        f"[Simulation] ML filtre {sig.ticker}: "
                        f"sentiment={sig.sentiment:.2f} mais ML={ml_pred['action']} "
                        f"(conf={ml_pred['confidence']:.2f})"
                    )
                    continue
                logger.debug(
                    f"[Simulation] ML confirme {sig.ticker}: "
                    f"ML={ml_pred['action']} conf={ml_pred['confidence']:.2f}"
                )

            # Montant du trade
            trade_amount = min(max_trade, port.cash_available - cash_min)
            if trade_amount < cash_min:
                continue
            qty = trade_amount / price
            try:
                trade = self.buy(db, sig.ticker, qty, price, signal_id=sig.id)
                trades.append(trade)
                sig.consumed = 1
                db.commit()
            except ValueError as e:
                logger.warning(f"Simulation trade échoué: {e}")

        # Mise à jour des prix
        self.update_position_prices(db, prices)
        self.snapshot_history(db)
        return trades
