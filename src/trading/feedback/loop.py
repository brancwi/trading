"""FeedbackLoop — suit les performances et ajuste la confiance XGBoost vs LLM.

La boucle de rétroaction:
1. Après chaque trade, évalue si XGBoost ou LLM avait raison
2. Ajuste un poids de confiance dynamique (alpha XGBoost, beta LLM)
3. Si le LLM surperforme → augmentation du poids LLM
4. Si XGBoost surperforme → augmentation du poids XGBoost
5. Réentraînement périodique du modèle XGBoost

Usage:
    fb = FeedbackLoop(portfolio_id="staging-ninja")
    fb.record_trade(trade, source="llm")  # ou "xgboost"
    weights = fb.get_weights()  # {"xgboost": 0.6, "llm": 0.4}
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from trading.core.database import db_session
from trading.core.models import Trade, Portfolio

logger = logging.getLogger(__name__)

# Fenêtre de mémoire pour le feedback (derniers N trades)
FEEDBACK_WINDOW = 50

# Alpha d'apprentissage (vitesse d'ajustement)
LEARNING_RATE = 0.05


class FeedbackLoop:
    """Boucle de rétroaction pour ajuster les poids XGBoost vs LLM."""

    def __init__(self, portfolio_id: str):
        self.portfolio_id = portfolio_id
        self.weights = self._load_weights()

    def _load_weights(self) -> dict[str, float]:
        """Charge les poids depuis le portfolio config."""
        with db_session() as db:
            port = db.query(Portfolio).filter(Portfolio.id == self.portfolio_id).first()
            if port and port.config_json:
                config = json.loads(port.config_json)
                return config.get("feedback_weights", {"xgboost": 0.5, "llm": 0.5})
        return {"xgboost": 0.5, "llm": 0.5}

    def _save_weights(self, db: Session) -> None:
        """Sauvegarde les poids dans le portfolio config."""
        port = db.query(Portfolio).filter(Portfolio.id == self.portfolio_id).first()
        if port:
            config = json.loads(port.config_json or "{}")
            config["feedback_weights"] = self.weights
            port.config_json = json.dumps(config)
            db.commit()

    def record_trade(
        self,
        trade: Trade,
        source: str = "unknown",
    ) -> None:
        """Enregistre un trade pour feedback futur."""
        with db_session() as db:
            trade.feedback_source = source
            db.commit()
        logger.info("[Feedback] Trade %s enregistré (source=%s)", trade.ticker, source)

    def evaluate_recent_trades(self) -> dict[str, Any]:
        """Évalue les performances des N derniers trades par source."""
        with db_session() as db:
            since = datetime.utcnow() - timedelta(days=30)
            trades = (
                db.query(Trade)
                .filter(
                    Trade.portfolio_id == self.portfolio_id,
                    Trade.created_at >= since,
                    Trade.feedback_source.isnot(None),
                )
                .order_by(Trade.created_at.desc())
                .limit(FEEDBACK_WINDOW)
                .all()
            )

            results: dict[str, dict[str, Any]] = {}
            for trade in trades:
                src = trade.feedback_source or "unknown"
                if src not in results:
                    results[src] = {"count": 0, "pnl_sum": 0.0, "winners": 0}

                results[src]["count"] += 1
                results[src]["pnl_sum"] += trade.realized_pnl or 0
                if (trade.realized_pnl or 0) > 0:
                    results[src]["winners"] += 1

            # Calculer les métriques
            for src, data in results.items():
                if data["count"] > 0:
                    data["avg_pnl"] = data["pnl_sum"] / data["count"]
                    data["win_rate"] = data["winners"] / data["count"]

            return results

    def adjust_weights(self) -> dict[str, float]:
        """Ajuste les poids XGBoost vs LLM basé sur les performances récentes."""
        results = self.evaluate_recent_trades()

        xgb = results.get("xgboost", {})
        llm = results.get("llm", {})

        xgb_avg = xgb.get("avg_pnl", 0)
        llm_avg = llm.get("avg_pnl", 0)

        # Si le LLM surperforme → augmenter son poids
        if llm_avg > xgb_avg:
            delta = LEARNING_RATE * (llm_avg - xgb_avg) / max(abs(xgb_avg), 1)
            self.weights["llm"] = min(0.8, self.weights["llm"] + delta)
            self.weights["xgboost"] = 1.0 - self.weights["llm"]
            logger.info(
                "[Feedback] LLM surperforme (%.2f vs %.2f) → poids LLM: %.2f",
                llm_avg, xgb_avg, self.weights["llm"],
            )
        elif xgb_avg > llm_avg:
            delta = LEARNING_RATE * (xgb_avg - llm_avg) / max(abs(llm_avg), 1)
            self.weights["xgboost"] = min(0.8, self.weights["xgboost"] + delta)
            self.weights["llm"] = 1.0 - self.weights["xgboost"]
            logger.info(
                "[Feedback] XGBoost surperforme (%.2f vs %.2f) → poids XGB: %.2f",
                xgb_avg, llm_avg, self.weights["xgboost"],
            )
        else:
            logger.info("[Feedback] Performances égales — poids inchangés")

        with db_session() as db:
            self._save_weights(db)

        return self.weights

    def get_weights(self) -> dict[str, float]:
        """Retourne les poids actuels."""
        return self.weights

    def should_use_llm(self, llm_confidence: float, xgboost_confidence: float) -> bool:
        """Décide si on fait confiance au LLM ou au XGBoost pour ce ticker."""
        # Si le LLM a un poids élevé ET une confiance élevée → utiliser le LLM
        if self.weights["llm"] > 0.6 and llm_confidence > 0.7:
            return True
        # Si XGBoost a un poids élevé → utiliser XGBoost
        if self.weights["xgboost"] > 0.6 and xgboost_confidence > 0.5:
            return False
        # Par défaut, utiliser celui qui a la plus haute confiance
        return llm_confidence > xgboost_confidence
