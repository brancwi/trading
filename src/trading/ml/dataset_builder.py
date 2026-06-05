"""Construction du dataset final d'entraînement.

Fusionne :
  1. Features techniques (prix OHLCV + indicateurs classiques + crypto-natives)
  2. Features de sentiment (agrégées par ticker/date)
  3. Fear & Greed Index (macro crypto sentiment)
  4. Funding Rates (perp futures premium)
  5. Labels (training_labels : BUY / SELL / HOLD)

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


def _load_fear_greed(db: Session) -> pd.DataFrame:
    """Charge l'historique Fear & Greed Index."""
    rows = db.execute(text("""
        SELECT date, value, classification FROM fear_greed ORDER BY date
    """)).fetchall()

    if not rows:
        logger.warning("[Dataset] No Fear & Greed data found")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "date": pd.to_datetime(r.date),
            "fear_greed": r.value,
            "fg_classification": r.classification,
        }
        for r in rows
    ])
    return df


def _load_funding_rates(db: Session) -> pd.DataFrame:
    """Charge les funding rates agrégés par (symbol, date)."""
    rows = db.execute(text("""
        SELECT 
            symbol,
            DATE(timestamp) as date,
            AVG(funding_rate) as funding_rate_avg,
            MAX(ABS(funding_rate)) as funding_rate_max_abs
        FROM funding_rates
        GROUP BY symbol, DATE(timestamp)
    """)).fetchall()

    if not rows:
        logger.warning("[Dataset] No funding rates data found")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "symbol": r.symbol,
            "date": pd.to_datetime(r.date),
            "funding_rate_avg": r.funding_rate_avg,
            "funding_rate_max_abs": r.funding_rate_max_abs,
        }
        for r in rows
    ])
    return df


def _load_macro_indicators(db: Session) -> pd.DataFrame:
    """Charge les indicateurs macro (VIX, DXY, TNX)."""
    rows = db.execute(text("""
        SELECT date, indicator, value FROM macro_indicators ORDER BY date
    """)).fetchall()

    if not rows:
        logger.warning("[Dataset] No macro indicators data found")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "date": pd.to_datetime(r.date),
            "indicator": r.indicator,
            "value": r.value,
        }
        for r in rows
    ])
    # Pivot pour avoir une colonne par indicateur
    df = df.pivot(index="date", columns="indicator", values="value").reset_index()
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

    # 3) Fear & Greed (macro sentiment) — merge par date seule
    fg = _load_fear_greed(db)
    if not fg.empty:
        tech = tech.merge(fg, on="date", how="left")
        tech["fear_greed"] = tech["fear_greed"].fillna(50)  # neutral default
    else:
        tech["fear_greed"] = 50.0

    # 4) Funding Rates — map symbol to ticker
    fr = _load_funding_rates(db)
    if not fr.empty:
        # Map Binance symbol (BTCUSDT) to our ticker format (BTC-USD)
        fr["ticker"] = fr["symbol"].str.replace("USDT", "-USD")
        tech = tech.merge(
            fr[["ticker", "date", "funding_rate_avg", "funding_rate_max_abs"]],
            on=["ticker", "date"],
            how="left",
        )
        tech["funding_rate_avg"] = tech["funding_rate_avg"].fillna(0)
        tech["funding_rate_max_abs"] = tech["funding_rate_max_abs"].fillna(0)
    else:
        tech["funding_rate_avg"] = 0.0
        tech["funding_rate_max_abs"] = 0.0

    # 5) Macro Indicators (VIX, DXY, TNX) — merge par date seulement
    macro = _load_macro_indicators(db)
    if not macro.empty:
        tech = tech.merge(macro, on="date", how="left")
        for col in ["VIX", "DXY", "TNX"]:
            if col in tech.columns:
                tech[col] = tech[col].ffill().bfill()
            else:
                tech[col] = 0.0
    else:
        tech["VIX"] = 0.0
        tech["DXY"] = 0.0
        tech["TNX"] = 0.0

    # 5) Labels
    labels = _load_labels(db, horizon=horizon, threshold=threshold)
    if labels.empty:
        raise ValueError("No labels available")

    labels["date"] = pd.to_datetime(labels["timestamp"]).dt.date
    labels["date"] = pd.to_datetime(labels["date"])

    merged = tech.merge(
        labels[["ticker", "date", "label", "future_return"]],
        on=["ticker", "date"],
        how="inner",
    )

    # Drop NaN sur les features techniques (features externes sont optionnelles)
    feature_cols = [
        "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
        "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_pct", "atr_14",
        "momentum_10", "roc_10", "volatility_20",
        "price_sma20_ratio", "price_sma50_ratio",
        # Crypto-native
        "price_change_1d", "price_change_3d", "price_change_7d",
        "volume_ratio", "volume_change_1d",
        "adx_14", "stoch_k", "stoch_d",
        "obv_ratio", "williams_r", "tr_normalized",
        "volatility_ratio",
        # External
        "fear_greed", "funding_rate_avg", "funding_rate_max_abs",
    ]

    # Sentiment est optionnel
    for sent_col in ["sentiment_mean", "sentiment_confidence", "sentiment_count"]:
        if sent_col in merged.columns:
            merged[sent_col] = merged[sent_col].fillna(0)

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
        print(df[["ticker", "date", "close", "rsi_14", "fear_greed", "funding_rate_avg", "label"]].head(10))
