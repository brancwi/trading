"""Stratégie Simulation - Day Trading sur signaux news."""

import json
import logging

from sqlalchemy.orm import Session

from trading.strategies.base import StrategyBase
from trading.core.models import Trade, Signal
from trading.ml.signal_model import SignalModel

logger = logging.getLogger(__name__)


class SimulationStrategy(StrategyBase):
    """Achat sur sentiment fort, filtré par DecisionLLM ou SignalModel ML."""

    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        port = self.get_portfolio(db)
        if port.status != "active":
            logger.info(f"Simulation {port.status} → skip")
            return []
        config = json.loads(port.config_json or "{}")
        threshold = config.get("sentiment_threshold", 0.5)
        cash_min = config.get("cash_min", 100)
        max_trade = port.max_trade_amount or 500
        use_decision_llm = config.get("enable_decision_llm", False)

        # Chargement lazy du SignalModel
        signal_model = SignalModel()
        use_ml = signal_model.trained
        if use_ml:
            logger.info("[Simulation] SignalModel entraîné — filtrage ML actif")

        trades: list[Trade] = []
        signals = self.get_signals(db)

        # ── Agrégation par ticker : 1 seul trade par ticker ──
        from collections import defaultdict
        ticker_signals: dict[str, list[Signal]] = defaultdict(list)
        for sig in signals:
            if sig.action not in ("BUY", "STRONG_BUY"):
                continue
            if sig.ticker not in prices:
                continue
            price = prices[sig.ticker]
            if price <= 0:
                continue
            ticker_signals[sig.ticker].append(sig)

        # ── Mode DecisionLLM : vision globale du portfolio ──
        if use_decision_llm:
            try:
                from trading.strategies.decision_llm import DecisionLLM
                llm = DecisionLLM()
                # Prépare la liste de tous les signaux filtrés pour le LLM
                all_signals = []
                for sigs in ticker_signals.values():
                    all_signals.extend(sigs)
                decisions = llm.decide(port, all_signals, prices)
                for decision in decisions:
                    ticker = decision["ticker"]
                    amount = min(decision["amount"], max_trade, port.cash_available - cash_min)
                    if amount < cash_min or ticker not in prices:
                        continue
                    qty = amount / prices[ticker]
                    try:
                        trade = self.buy(db, ticker, qty, prices[ticker])
                        trades.append(trade)
                        # Consomme tous les signaux de ce ticker
                        for s in ticker_signals.get(ticker, []):
                            s.consumed = 1
                        db.commit()
                    except ValueError as e:
                        logger.warning(f"Simulation LLM trade échoué: {e}")
            except Exception as e:
                logger.exception(f"[Simulation] DecisionLLM erreur: {e} — fallback mode règles")
                use_decision_llm = False

        # ── Mode règles (fallback ou par défaut) ──
        if not use_decision_llm:
            for ticker, sigs in ticker_signals.items():
                # Sélection du meilleur signal (plus haut sentiment * confidence)
                best_sig = max(sigs, key=lambda s: abs(s.sentiment) * s.confidence)

                # Filtrage ML : si le modèle prédit HOLD/SELL, on ignore tout le ticker
                if use_ml:
                    ml_pred = signal_model.predict(
                        ticker=ticker,
                        sentiment_combined=best_sig.sentiment,
                        sentiment_confidence=best_sig.confidence,
                    )
                    if ml_pred["action"] in ("HOLD", "SELL"):
                        logger.info(
                            f"[Simulation] ML filtre {ticker}: "
                            f"sentiment={best_sig.sentiment:.2f} mais ML={ml_pred['action']} "
                            f"(conf={ml_pred['confidence']:.2f})"
                        )
                        continue

                # Montant du trade (une seule fois par ticker)
                trade_amount = min(max_trade, port.cash_available - cash_min)
                if trade_amount < cash_min:
                    continue
                qty = trade_amount / prices[ticker]
                try:
                    trade = self.buy(db, ticker, qty, prices[ticker], signal_id=best_sig.id)
                    trades.append(trade)
                    # Consomme TOUS les signaux du ticker pour éviter les doublons futurs
                    for s in sigs:
                        s.consumed = 1
                    db.commit()
                except ValueError as e:
                    logger.warning(f"Simulation trade échoué: {e}")

        # Mise à jour des prix
        self.update_position_prices(db, prices)
        self.snapshot_history(db)
        return trades
