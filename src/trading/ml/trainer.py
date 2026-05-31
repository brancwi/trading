"""Entraînement des modèles de prédiction de signaux.

Baseline : XGBoost multi-classe (BUY / SELL / HOLD)
Avancé : LSTM (optionnel — nécessite plus de données)
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "atr_14",
    "momentum_10", "roc_10", "volatility_20",
    "price_sma20_ratio", "price_sma50_ratio",
    "sentiment_mean", "sentiment_confidence",
]

LABEL_MAP = {"HOLD": 0, "BUY": 1, "SELL": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


def _prepare_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Extrait X, y et fit le scaler."""
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["label"].map(LABEL_MAP).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_xgboost(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Entraîne un XGBClassifier multi-classe.

    Returns:
        Dict avec modèle, scaler, métriques, predictions
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ImportError("xgboost is required. Install with: pip install xgboost") from e

    X, y, scaler = _prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info("[Trainer] XGBoost train=%d test=%d features=%d", len(X_train), len(X_test), X.shape[1])

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=4,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
    }

    logger.info("[Trainer] XGBoost metrics: %s", metrics)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["HOLD", "BUY", "SELL"]))

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "feature_names": FEATURE_COLS,
    }


def save_model(result: dict[str, Any], path: str | Path) -> None:
    """Sauvegarde le modèle + scaler + métriques."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "model": result["model"],
            "scaler": result["scaler"],
            "metrics": result["metrics"],
            "feature_names": result["feature_names"],
            "label_map": LABEL_MAP,
        }, f)
    # Métriques JSON lisibles
    json_path = path.with_suffix(".metrics.json")
    with open(json_path, "w") as f:
        json.dump(result["metrics"], f, indent=2)
    logger.info("[Trainer] Model saved to %s", path)


def load_model(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from trading.core.database import db_session
    from trading.ml.dataset_builder import build_dataset

    with db_session() as db:
        df = build_dataset(db, horizon=5, threshold=0.03)
        result = train_xgboost(df)
        save_model(result, "models/signal_xgboost.pkl")
