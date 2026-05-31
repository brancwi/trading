"""Construction du dataset final d'entraînement.

Fusionne :
  1. Features techniques (prix OHLCV + indicateurs)
  2. Features de sentiment (agrégées par ticker/date)
  3. Labels (training_labels : BUY / SELL / HOLD)

Produit un DataFrame prêt pour l'entraînement scikit-learn / PyTorch.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from trading.ml.features_engineering import build_technical_features

logger = logging.getLogger(__name__)


def _aggregate_sentiment(db: Session) -> pd.DataFrame:
    """Agrège les scores de sentiment par (ticker, date)."""
    rows = db.execute(text("""
        SELECT
            ticker,
            DATE(timestamp) as date,
            AVG(combined_score) as sentiment_mean,
            COUNT(*) as sentiment_count,
            AVG(confidence) as sentiment_confidence
        FROM sentiment_scores
        GROUP BY ticker, DATE(timestamp)
    """)).fetchall()

    if not rows:
        logger.warning("[Dataset] No sentiment data found")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "date": pd.to_datetime(r.date),
            "sentiment_mean": r.sentiment_mean,
            "sentiment_count": r.sentiment_count,
            "sentiment_confidence": r.sentiment_confidence,
        }
        for r in rows
    ])
    return df


def _load_labels(db: Session, horizon: int = 5, threshold: float = 0.03) -> pd.DataFrame:
    """Charge les labels pour un horizon/seuil donné."""
    rows = db.execute(text("""
        SELECT
            ticker,
            timestamp,
            label,
            future_return,
            close_at_signal
        FROM training_labels
        WHERE horizon_days = :horizon AND threshold_used = :threshold
    """), {"horizon": horizon, "threshold": threshold}).fetchall()

    if not rows:
        logger.warning("[Dataset] No labels found for H=%d th=%.2f", horizon, threshold)
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "label": r.label,
            "future_return": r.future_return,
            "close_at_signal": r.close_at_signal,
        }
        for r in rows
    ])
    return df


def build_dataset(
    db: Session,
    horizon: int = 5,
    threshold: float = 0.03,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Construit le dataset final fusionné.

    Returns:
        DataFrame avec features + label (BUY/SELL/HOLD)
    """
    logger.info("[Dataset] Building for H=%d th=%.2f", horizon, threshold)

    # 1) Features techniques
    tech = build_technical_features(db, tickers=tickers)
    if tech.empty:
        raise ValueError("No technical features available")
    tech["date"] = pd.to_datetime(tech["timestamp"]).dt.date
    tech["date"] = pd.to_datetime(tech["date"])

    # 2) Sentiment
    sent = _aggregate_sentiment(db)
    if not sent.empty:
        tech = tech.merge(sent, on=["ticker", "date"], how="left")
        tech["sentiment_mean"] = tech["sentiment_mean"].fillna(0)
        tech["sentiment_count"] = tech["sentiment_count"].fillna(0)
        tech["sentiment_confidence"] = tech["sentiment_confidence"].fillna(0)
    else:
        tech["sentiment_mean"] = 0.0
        tech["sentiment_count"] = 0
        tech["sentiment_confidence"] = 0.0

    # 3) Labels
    labels = _load_labels(db, horizon=horizon, threshold=threshold)
    if labels.empty:
        raise ValueError("No labels available")

    # Merge sur (ticker, timestamp) — on arrondit à la date pour matcher
    labels["date"] = pd.to_datetime(labels["timestamp"]).dt.date
    labels["date"] = pd.to_datetime(labels["date"])

    # On merge tech + labels sur (ticker, date)
    merged = tech.merge(labels[["ticker", "date", "label", "future_return"]], on=["ticker", "date"], how="inner")

    # Drop NaN sur les features critiques
    feature_cols = [
        "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_pct", "atr_14",
        "momentum_10", "roc_10", "volatility_20",
        "price_sma20_ratio", "price_sma50_ratio",
        "sentiment_mean", "sentiment_confidence",
    ]
    merged = merged.dropna(subset=feature_cols + ["label"])

    logger.info(
        "[Dataset] Final shape: %s  (BUY=%d SELL=%d HOLD=%d)",
        merged.shape,
        (merged["label"] == "BUY").sum(),
        (merged["label"] == "SELL").sum(),
        (merged["label"] == "HOLD").sum(),
    )
    return merged


if __name__ == "__main__":
    from trading.core.database import db_session
    with db_session() as db:
        df = build_dataset(db, horizon=5, threshold=0.03)
        print(df[["ticker", "date", "close", "rsi_14", "sentiment_mean", "label"]].head(10))
