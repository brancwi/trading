"""Fusion apprenante — régression linéaire sur l'historique sentiment_scores.

Apprend les poids optimaux pour combiner roberta, modern, qwen, lexical
à partir des annotations humaines (human_label).

Usage:
    from trading.sentiment.fusion_model import FusionModel
    fm = FusionModel()
    fm.train()          # Entraîne sur les lignes avec human_label
    score = fm.predict(roberta=0.8, modern=0.3, qwen=0.6)
"""

import json
import logging
from pathlib import Path

import numpy as np

from trading.core.database import db_session
from trading.core.models import SentimentScore

logger = logging.getLogger(__name__)

# Fichier de persistance des poids
_WEIGHTS_PATH = Path(__file__).with_suffix(".weights.json")


def _label_to_score(label: str) -> float:
    label = label.lower().strip()
    if "positive" in label:
        return 1.0
    if "negative" in label:
        return -1.0
    return 0.0


class FusionModel:
    """Régression linéaire pour combiner les scores des modèles."""

    FEATURES = ["roberta", "modern", "qwen", "lexical"]

    def __init__(self):
        self.weights: np.ndarray | None = None  # [w1, w2, w3, w4, bias]
        self.trained = False
        self._load_weights()

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def _load_weights(self) -> None:
        if _WEIGHTS_PATH.exists():
            try:
                data = json.loads(_WEIGHTS_PATH.read_text())
                self.weights = np.array(data["weights"])
                self.trained = data.get("trained", False)
                logger.info(f"[FusionModel] Poids chargés: {self.weights.tolist()}")
            except Exception as e:
                logger.warning(f"[FusionModel] Échec chargement poids: {e}")

    def _save_weights(self) -> None:
        if self.weights is not None:
            data = {
                "weights": self.weights.tolist(),
                "trained": self.trained,
                "features": self.FEATURES,
            }
            _WEIGHTS_PATH.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------

    def train(self) -> bool:
        """Entraîne sur les lignes sentiment_scores avec human_label.

        Retourne True si entraînement réussi, False sinon.
        """
        with db_session() as db:
            rows = (
                db.query(SentimentScore)
                .filter(
                    (SentimentScore.human_label.isnot(None)) |
                    (SentimentScore.validated_label.isnot(None))
                )
                .all()
            )

        if len(rows) < 10:
            logger.warning(
                f"[FusionModel] Pas assez de données annotées ({len(rows)} < 10). "
                "Utilisez la fusion fixe 50/50."
            )
            return False

        X = []
        y = []
        for r in rows:
            features = [
                r.roberta_score if r.roberta_score is not None else 0.0,
                r.modern_score if r.modern_score is not None else 0.0,
                r.qwen_score if r.qwen_score is not None else 0.0,
                r.lexical_score if r.lexical_score is not None else 0.0,
                1.0,  # biais
            ]
            X.append(features)
            label = r.human_label or r.validated_label or "neutral"
            y.append(_label_to_score(label))

        X_arr = np.array(X)
        y_arr = np.array(y)

        # Régression linéaire par moindres carrés
        # w = (X^T X)^-1 X^T y
        try:
            self.weights, residuals, rank, s = np.linalg.lstsq(X_arr, y_arr, rcond=None)
            self.trained = True
            self._save_weights()
            logger.info(f"[FusionModel] Entraîné sur {len(rows)} samples. Poids: {self.weights.tolist()}")
            return True
        except Exception as e:
            logger.error(f"[FusionModel] Échec entraînement: {e}")
            return False

    # ------------------------------------------------------------------
    # Prédiction
    # ------------------------------------------------------------------

    def predict(self, roberta: float | None, modern: float | None,
                qwen: float | None, lexical: float | None) -> float:
        """Prédit le score combiné avec les poids appris."""
        if not self.trained or self.weights is None:
            # Fallback : fusion fixe 50/50 (roberta + modern) ou qwen si présent
            scores = []
            if roberta is not None:
                scores.append(roberta)
            if modern is not None:
                scores.append(modern)
            if qwen is not None:
                scores.append(qwen)
            if lexical is not None:
                scores.append(lexical)
            if not scores:
                return 0.0
            return float(np.mean(scores))

        x = np.array([
            roberta if roberta is not None else 0.0,
            modern if modern is not None else 0.0,
            qwen if qwen is not None else 0.0,
            lexical if lexical is not None else 0.0,
            1.0,  # biais
        ])
        score = float(np.dot(x, self.weights))
        # Clamp entre -1 et 1
        return max(-1.0, min(1.0, score))

    def info(self) -> dict:
        """Retourne les infos du modèle pour le debug."""
        return {
            "trained": self.trained,
            "features": self.FEATURES,
            "weights": self.weights.tolist() if self.weights is not None else None,
            "weights_file": str(_WEIGHTS_PATH),
        }
