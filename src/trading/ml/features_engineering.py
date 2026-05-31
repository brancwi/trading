"""Extraction de features techniques à partir des données de prix historiques.

Calculé en Pandas pour performance — produit un DataFrame avec toutes les
features classiques (RSI, MACD, Bollinger, ATR, momentum, etc.).
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from trading.core.models import MarketData

logger = logging.getLogger(__name__)


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def _compute_bollinger(series: pd.Series, window: int = 20, num_std: int = 2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def build_technical_features(db: Session, tickers: list[str] | None = None) -> pd.DataFrame:
    """Construit un DataFrame avec toutes les features techniques par ticker/date.

    Returns:
        DataFrame avec colonnes :
        [ticker, timestamp, close, volume,
         sma_10, sma_20, sma_50, ema_10, ema_20,
         rsi_14, macd, macd_signal, macd_hist,
         bb_upper, bb_middle, bb_lower, bb_width, bb_pct,
         atr_14, momentum_10, roc_10, volatility_20,
         price_sma20_ratio, price_sma50_ratio]
    """
    query = db.query(MarketData).order_by(MarketData.ticker, MarketData.timestamp.asc())
    if tickers:
        query = query.filter(MarketData.ticker.in_(tickers))

    rows = query.all()
    if not rows:
        logger.warning("[Features] No market data found")
        return pd.DataFrame()

    df = pd.DataFrame([
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "open": r.open_price,
            "high": r.high,
            "low": r.low,
            "close": r.price,
            "volume": r.volume,
        }
        for r in rows
    ])

    # Supprimer les doublons (ticker, timestamp)
    df = df.drop_duplicates(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"])

    all_features: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy().set_index("timestamp").sort_index()

        # Moyennes mobiles
        g["sma_10"] = g["close"].rolling(window=10).mean()
        g["sma_20"] = g["close"].rolling(window=20).mean()
        g["sma_50"] = g["close"].rolling(window=50).mean()
        g["ema_10"] = g["close"].ewm(span=10, adjust=False).mean()
        g["ema_20"] = g["close"].ewm(span=20, adjust=False).mean()

        # RSI
        g["rsi_14"] = _compute_rsi(g["close"], 14)

        # MACD
        macd, macd_sig, macd_hist = _compute_macd(g["close"])
        g["macd"] = macd
        g["macd_signal"] = macd_sig
        g["macd_hist"] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = _compute_bollinger(g["close"])
        g["bb_upper"] = bb_upper
        g["bb_middle"] = bb_middle
        g["bb_lower"] = bb_lower
        g["bb_width"] = (bb_upper - bb_lower) / bb_middle
        g["bb_pct"] = (g["close"] - bb_lower) / (bb_upper - bb_lower)

        # ATR
        g["atr_14"] = _compute_atr(g, 14)

        # Momentum & ROC
        g["momentum_10"] = g["close"] - g["close"].shift(10)
        g["roc_10"] = (g["close"] / g["close"].shift(10) - 1) * 100

        # Volatilité (std 20j)
        g["volatility_20"] = g["close"].rolling(window=20).std()

        # Ratios prix / MA
        g["price_sma20_ratio"] = g["close"] / g["sma_20"]
        g["price_sma50_ratio"] = g["close"] / g["sma_50"]

        g = g.reset_index()
        g["ticker"] = ticker
        all_features.append(g)

    result = pd.concat(all_features, ignore_index=True)
    logger.info("[Features] Built %d rows with %d features", len(result), result.shape[1])
    return result


if __name__ == "__main__":
    from trading.core.database import db_session
    with db_session() as db:
        df = build_technical_features(db)
        print(df.head())
        print(f"\nShape: {df.shape}")
