"""SignalModel v1 — Random Forest sur sentiment + indicateurs techniques.

Usage:
    from trading.ml.signal_model import SignalModel
    sm = SignalModel()
    sm.train()          # Entraîne sur l'historique
    signal = sm.predict(ticker="AAPL")  # {"action": "BUY", "confidence": 0.82}
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score

from trading.core.database import db_session
from trading.core.models import MarketData, SentimentScore
from trading.ml.features import FeatureEngine

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).with_suffix(".joblib")
_META_PATH = Path(__file__).with_suffix(".meta.json")


def _label_from_price_change(change_pct: float, threshold: float = 1.0) -> int:
    """Convertit une variation de prix en label : 1=UP, 0=HOLD, -1=DOWN."""
    if change_pct > threshold:
        return 1
    if change_pct < -threshold:
        return -1
    return 0


class SignalModel:
    """Modèle de décision : prédit hausse/baisse à partir de features marché + sentiment."""

    FEATURE_COLS = [
        "sma_10", "sma_20", "rsi_14", "momentum_5", "momentum_10",
        "bb_width_20", "volume_avg_10", "sentiment_combined", "sentiment_confidence",
        "sentiment_divergence",
    ]

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.trained = False
        self._load()

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if _MODEL_PATH.exists():
            try:
                self.model = joblib.load(_MODEL_PATH)
                self.trained = True
                logger.info("[SignalModel] Modèle chargé depuis %s", _MODEL_PATH)
            except Exception as e:
                logger.warning("[SignalModel] Échec chargement: %s", e)

    def _save(self) -> None:
        if self.model is not None:
            joblib.dump(self.model, _MODEL_PATH)
            meta = {
                "feature_cols": self.FEATURE_COLS,
                "trained": self.trained,
                "model_type": "RandomForestClassifier",
            }
            _META_PATH.write_text(json.dumps(meta, indent=2))

    # ------------------------------------------------------------------
    # Construction du dataset
    # ------------------------------------------------------------------

    def _build_dataset(self, window_hours: int = 4, min_samples: int = 50) -> tuple[np.ndarray, np.ndarray] | None:
        """Construit X, y à partir de market_data + sentiment_scores."""
        engine = FeatureEngine()

        with db_session() as db:
            # Récupère tous les sentiment_scores avec prix d'analyse
            scores = (
                db.query(SentimentScore)
                .filter(SentimentScore.price_at_analysis.isnot(None))
                .order_by(SentimentScore.timestamp.asc())
                .all()
            )

        X_rows = []
        y_rows = []

        for score in scores:
            # Features techniques au moment de l'analyse
            tech = engine.compute(score.ticker)
            if not tech or tech.get("sma_10") is None:
                continue

            # Features sentiment
            sentiment_combined = score.combined_score or 0.0
            sentiment_confidence = score.confidence or 0.0
            sentiment_divergence = score.divergence or 0.0

            # Label : variation du prix à T+window_hours
            # Si validated_label existe, on l'utilise
            if score.validated_label is not None:
                label = _label_from_price_change(score.price_change_pct or 0, threshold=1.0)
            else:
                # Sinon on calcule à la volée
                price_t = score.price_at_analysis
                price_t_plus = self._get_price_at(score.ticker, score.timestamp, offset_hours=window_hours)
                if price_t_plus is None or price_t is None or price_t == 0:
                    continue
                change_pct = ((price_t_plus - price_t) / price_t) * 100
                label = _label_from_price_change(change_pct, threshold=1.0)

            row = [
                tech.get("sma_10", 0.0),
                tech.get("sma_20", 0.0),
                tech.get("rsi_14", 50.0),
                tech.get("momentum_5", 0.0),
                tech.get("momentum_10", 0.0),
                tech.get("bb_width_20", 0.0),
                tech.get("volume_avg_10", 0.0),
                sentiment_combined,
                sentiment_confidence,
                sentiment_divergence,
            ]
            X_rows.append(row)
            y_rows.append(label)

        if len(X_rows) < min_samples:
            logger.warning("[SignalModel] Pas assez d'échantillons: %d < %d", len(X_rows), min_samples)
            return None

        return np.array(X_rows), np.array(y_rows)

    def _get_price_at(self, ticker: str, timestamp, offset_hours: int = 4) -> float | None:
        """Récupère le prix à timestamp + offset_hours."""
        from datetime import timedelta
        from trading.core.database import engine
        from sqlalchemy import text
        target = timestamp + timedelta(hours=offset_hours)
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT price FROM market_data
                    WHERE ticker = :ticker AND timestamp <= :ts
                    ORDER BY timestamp DESC
                    LIMIT 1
                """),
                {"ticker": ticker, "ts": target},
            ).fetchone()
            if row:
                return float(row[0])
        return None

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Entraîne le modèle sur l'historique."""
        dataset = self._build_dataset()
        if dataset is None:
            return {"trained": False, "reason": "insufficient_data"}

        X, y = dataset
        # TimeSeriesSplit pour éviter le leakage temporel
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))

        # Entraînement final sur tout le dataset
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y)
        self.trained = True
        self._save()

        # Feature importances
        importances = dict(zip(self.FEATURE_COLS, self.model.feature_importances_.tolist()))

        logger.info("[SignalModel] Entraîné sur %d samples. CV accuracies: %s", len(X), scores)
        return {
            "trained": True,
            "samples": len(X),
            "cv_accuracies": scores,
            "mean_cv_accuracy": float(np.mean(scores)),
            "feature_importances": importances,
        }

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(self, ticker: str, sentiment_combined: float = 0.0,
                sentiment_confidence: float = 0.0, sentiment_divergence: float = 0.0) -> dict[str, Any]:
        """Prédit le signal pour un ticker donné."""
        if not self.trained or self.model is None:
            return {"action": "HOLD", "confidence": 0.0, "model_trained": False}

        engine = FeatureEngine()
        tech = engine.compute(ticker)
        if not tech:
            return {"action": "HOLD", "confidence": 0.0, "reason": "no_market_data"}

        x = np.array([[
            tech.get("sma_10", 0.0),
            tech.get("sma_20", 0.0),
            tech.get("rsi_14", 50.0),
            tech.get("momentum_5", 0.0),
            tech.get("momentum_10", 0.0),
            tech.get("bb_width_20", 0.0),
            tech.get("volume_avg_10", 0.0),
            sentiment_combined,
            sentiment_confidence,
            sentiment_divergence,
        ]])

        proba = self.model.predict_proba(x)[0]
        pred = self.model.predict(x)[0]

        # Mapping label → action
        action_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        action = action_map.get(int(pred), "HOLD")
        confidence = float(np.max(proba))

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "probabilities": {
                "SELL": round(float(proba[0]), 4) if len(proba) > 0 else 0,
                "HOLD": round(float(proba[1]), 4) if len(proba) > 1 else 0,
                "BUY": round(float(proba[2]), 4) if len(proba) > 2 else 0,
            },
            "model_trained": True,
        }

    def info(self) -> dict[str, Any]:
        return {
            "trained": self.trained,
            "feature_cols": self.FEATURE_COLS,
            "model_path": str(_MODEL_PATH),
            "meta_path": str(_META_PATH),
        }
