"""Générateur de signaux ML — utilise les modèles XGBoost entraînés par portfolio.

Usage:
    from trading.ml.signal_generator import MLSignalGenerator
    gen = MLSignalGenerator("staging-ninja")
    signals = gen.generate_signals(db)
    # Insère des Signal() dans la DB pour chaque prédiction BUY/SELL
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler

from trading.core.database import db_session
from trading.core.models import MarketData, Signal, Portfolio
from trading.ml.features_engineering import build_technical_features
from trading.ml.trainer import FEATURE_COLS

logger = logging.getLogger(__name__)

from trading.core.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"

# Mapping labels XGBoost → actions
LABEL_TO_ACTION = {0: "HOLD", 1: "BUY", 2: "SELL"}


class MLSignalGenerator:
    """Génère des signaux de trading à partir d'un modèle XGBoost entraîné."""

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Optimisé par grid search

    def __init__(self, portfolio_id: str, confidence_threshold: float | None = None):
        self.portfolio_id = portfolio_id
        self.confidence_threshold = confidence_threshold or self.DEFAULT_CONFIDENCE_THRESHOLD
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self) -> None:
        """Charge le modèle XGBoost le plus récent pour ce portfolio."""
        pattern = f"signal_{self.portfolio_id}_H*_th*_walkforward.pkl"
        files = list(MODELS_DIR.glob(pattern))
        if not files:
            logger.warning("[MLSignal] Aucun modèle trouvé pour %s", self.portfolio_id)
            return
        
        model_path = max(files, key=lambda p: p.stat().st_mtime)
        try:
            with open(model_path, "rb") as f:
                result = pickle.load(f)
            self.model = result.get("model")
            self.scaler = result.get("scaler")
            self.feature_cols = result.get("feature_names", FEATURE_COLS)
            # Charge les hyperparams du modèle
            hyper = result.get("hyperparams", {})
            if "confidence_threshold" in hyper:
                self.confidence_threshold = hyper["confidence_threshold"]
                logger.info("[MLSignal] Threshold chargé depuis modèle: %.2f", self.confidence_threshold)
            # Charge les tickers autorisés (si le modèle est spécialisé)
            self.allowed_tickers = hyper.get("tickers_trained") or result.get("tickers_trained")
            if self.allowed_tickers:
                logger.info("[MLSignal] Tickers autorisés: %s", self.allowed_tickers)
            logger.info("[MLSignal] Modèle chargé: %s", model_path.name)
        except Exception as e:
            logger.error("[MLSignal] Échec chargement modèle %s: %s", model_path, e)

    def _build_features(self, db: Session, tickers: list[str] | None = None) -> pd.DataFrame:
        """Construit les features techniques sur les dernières données."""
        # Si le modèle est spécialisé, ne construit les features que sur les tickers autorisés
        if tickers is None and self.allowed_tickers:
            tickers = self.allowed_tickers
        df = build_technical_features(db, tickers=tickers)
        if df.empty:
            return df
        
        # On ne garde que la dernière ligne par ticker
        df = df.sort_values(["ticker", "timestamp"])
        df_latest = df.groupby("ticker").last().reset_index()
        return df_latest

    def generate_signals(self, db: Session) -> list[Signal]:
        """Génère des signaux BUY/SELL à partir du modèle ML.
        
        Returns:
            Liste des signaux créés (déjà commités en DB)
        """
        if self.model is None or self.scaler is None:
            logger.warning("[MLSignal] Modèle non chargé pour %s — skip", self.portfolio_id)
            return []
        
        # Récupère les dernières features
        df = self._build_features(db)
        if df.empty:
            logger.warning("[MLSignal] Pas de données marché — skip")
            return []
        
        # Garde uniquement les colonnes utilisées par le modèle
        available_cols = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(df.columns)
        if missing:
            logger.warning("[MLSignal] Colonnes manquantes: %s", missing)
        
        signals: list[Signal] = []
        for _, row in df.iterrows():
            ticker = row["ticker"]
            
            # Vérifie que toutes les features sont disponibles
            if any(pd.isna(row.get(c)) for c in available_cols):
                continue
            
            X = np.array([[row.get(c, 0.0) for c in available_cols]], dtype=np.float32)
            X_scaled = self.scaler.transform(X)
            
            # Prédiction
            proba = self.model.predict_proba(X_scaled)[0]
            pred = int(self.model.predict(X_scaled)[0])
            action = LABEL_TO_ACTION.get(pred, "HOLD")
            confidence = float(proba[pred])
            
            if action == "HOLD":
                continue
            
            # Filtre par confiance minimum
            if confidence < self.confidence_threshold:
                continue
            
            # Crée le signal (conversion numpy → float natif pour PostgreSQL)
            signal = Signal(
                ticker=ticker,
                action=action,
                sentiment=float(proba[1] - proba[2]),  # sentiment = prob(BUY) - prob(SELL)
                strength=float(confidence),
                confidence=float(confidence),
                source=f"ml_xgboost_{self.portfolio_id}",
                price_at_signal=float(row["close"]),
            )
            db.add(signal)
            signals.append(signal)
            logger.info("[MLSignal] %s %s → %s (conf=%.2f)", self.portfolio_id, ticker, action, confidence)
        
        if signals:
            db.commit()
            logger.info("[MLSignal] %d signaux générés pour %s", len(signals), self.portfolio_id)
        
        return signals


def generate_all_portfolio_signals() -> dict[str, int]:
    """Génère des signaux ML pour tous les portfolios actifs."""
    results: dict[str, int] = {}
    
    with db_session() as db:
        portfolios = db.query(Portfolio).filter(Portfolio.status == "active").all()
        
        for port in portfolios:
            try:
                gen = MLSignalGenerator(port.id)
                signals = gen.generate_signals(db)
                results[port.id] = len(signals)
            except Exception as e:
                logger.exception("[MLSignal] Échec pour %s: %s", port.id, e)
                results[port.id] = 0
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    counts = generate_all_portfolio_signals()
    print("Signaux générés:", counts)
