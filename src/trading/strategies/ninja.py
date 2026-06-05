"""Stratégie Ninja — Momentum court terme avec DecisionLLM.

Architecture:
  1. XGBoost génère des candidats (signaux BUY/SELL)
  2. OpportunityAgent découvre des opportunités externes
  3. DecisionLLM raisonne et sélectionne les meilleures
  4. Exécution avec contraintes cash/SL/TP
"""

import json
import logging

from sqlalchemy.orm import Session

from trading.strategies.base import StrategyBase
from trading.core.models import Trade, Signal
from trading.strategies.decision_llm import DecisionLLM
from trading.opportunities.agent import OpportunityAgent

logger = logging.getLogger(__name__)


class NinjaStrategy(StrategyBase):
    """Ninja momentum H=2 avec DecisionLLM."""

    def run(self, db: Session, prices: dict[str, float]) -> list[Trade]:
        port = self.get_portfolio(db)
        if port.status != "active":
            return []

        config = json.loads(port.config_json or "{}")
        cash_min = config.get("cash_min", 50)
        max_trade = port.max_trade_amount or 150
        sl_pct = config.get("stop_loss_pct", 0.03)
        tp_pct = config.get("take_profit_pct", 0.06)

        trades: list[Trade] = []

        # ── 1. Candidats XGBoost ──────────────────────────────────────
        xgboost_signals = self.get_signals(
            db, source_prefix=f"ml_xgboost_{self.portfolio_id}"
        )
        logger.info("[Ninja] %d signaux XGBoost", len(xgboost_signals))

        # ── 2. Opportunités externes ──────────────────────────────────
        opp_agent = OpportunityAgent()
        opportunities = opp_agent.find_opportunities(db, self.portfolio_id)
        logger.info("[Ninja] %d opportunités découvertes", len(opportunities))

        # ── 3. SL/TP des positions existantes ─────────────────────────
        positions = self.get_positions(db)
        for pos in positions:
            if pos.current_price is None or pos.avg_entry_price is None:
                continue
            pnl_pct = (pos.current_price / pos.avg_entry_price - 1)

            if pnl_pct <= -sl_pct:
                # Forcer la vente SL
                logger.info("[Ninja] SL déclenché: %s %.1f%%", pos.ticker, pnl_pct * 100)
                try:
                    trade = self.sell(db, pos.ticker, pos.quantity, pos.current_price)
                    trades.append(trade)
                except ValueError:
                    pass
                continue

            if pnl_pct >= tp_pct:
                # Forcer la vente TP
                logger.info("[Ninja] TP déclenché: %s +%.1f%%", pos.ticker, pnl_pct * 100)
                try:
                    trade = self.sell(db, pos.ticker, pos.quantity, pos.current_price)
                    trades.append(trade)
                except ValueError:
                    pass
                continue

        # Si SL/TP ont vidé des positions, recharger le portfolio
        db.refresh(port)

        # ── 4. DecisionLLM ────────────────────────────────────────────
        # Ne passer au LLM que si on a des candidats et du cash
        if xgboost_signals or opportunities:
            try:
                llm = DecisionLLM()
                # Agréger tous les signaux/candidats
                all_signals = list(xgboost_signals)

                # Convertir les opportunités en pseudo-signaux pour le LLM
                for opp in opportunities:
                    if opp["ticker"] in prices:
                        pseudo_signal = Signal(
                            ticker=opp["ticker"],
                            action=opp["action"],
                            sentiment=1.0 if opp["action"] == "BUY" else -1.0,
                            strength=opp["confidence"],
                            confidence=opp["confidence"],
                            source=opp["source"],
                            price_at_signal=prices[opp["ticker"]],
                        )
                        all_signals.append(pseudo_signal)

                # Filtrer les doublons (même ticker → garder le plus confiant)
                seen = {}
                for sig in all_signals:
                    key = sig.ticker
                    if key not in seen or sig.confidence > seen[key].confidence:
                        seen[key] = sig
                unique_signals = list(seen.values())

                # Appel LLM
                decisions = llm.decide(port, unique_signals, prices)
                logger.info("[Ninja] DecisionLLM: %d décisions", len(decisions))

                # ── 5. Exécution des décisions LLM ─────────────────────
                # Map ticker -> signal_id pour lier chaque trade à son signal d'origine
                signal_map = {s.ticker: s.id for s in unique_signals if s.id}

                for dec in decisions:
                    ticker = dec.get("ticker", "")
                    action = dec.get("action", "")
                    amount = dec.get("amount", 0.0)
                    sig_id = signal_map.get(ticker)

                    if ticker not in prices:
                        continue
                    price = prices[ticker]

                    if action in ("BUY", "STRONG_BUY"):
                        # Vérifier cash
                        trade_amount = min(amount, port.cash_available - cash_min)
                        if trade_amount < cash_min:
                            logger.warning("[Ninja] Pas assez de cash pour %s", ticker)
                            continue

                        qty = trade_amount / price
                        try:
                            trade = self.buy(db, ticker, qty, price, signal_id=sig_id)
                            trades.append(trade)
                            logger.info("[Ninja] ACHAT %s: %.2f parts @ %.2f (sig=%s)", ticker, qty, price, sig_id)
                        except ValueError as e:
                            logger.warning("[Ninja] Échec achat %s: %s", ticker, e)

                    elif action == "SELL":
                        pos = self.get_position(db, ticker)
                        if pos:
                            try:
                                trade = self.sell(db, ticker, pos.quantity, price, signal_id=sig_id)
                                trades.append(trade)
                                logger.info("[Ninja] VENTE %s: %.2f parts @ %.2f (sig=%s)", ticker, pos.quantity, price, sig_id)
                            except ValueError as e:
                                logger.warning("[Ninja] Échec vente %s: %s", ticker, e)

            except Exception as e:
                logger.exception("[Ninja] DecisionLLM failed: %s — fallback XGBoost direct", e)
                # Fallback: exécuter les signaux XGBoost directement
                trades.extend(self._execute_xgboost_fallback(db, xgboost_signals, prices, port, cash_min))

        self.update_position_prices(db, prices)
        self.snapshot_history(db)
        return trades

    def _execute_xgboost_fallback(
        self, db: Session, signals: list[Signal], prices: dict[str, float],
        port, cash_min: float
    ) -> list[Trade]:
        """Fallback si le LLM échoue — exécute les signaux XGBoost directement."""
        trades = []
        for sig in signals:
            if sig.action not in ("BUY", "STRONG_BUY", "SELL"):
                continue
            if sig.ticker not in prices:
                continue
            price = prices[sig.ticker]

            if sig.action in ("BUY", "STRONG_BUY"):
                trade_amount = min(100, port.cash_available - cash_min)
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
        return trades
