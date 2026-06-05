"""Extraction de features techniques et crypto-natives à partir des données de prix historiques.

Features classiques (RSI, MACD, Bollinger, ATR) + features crypto-spécifiques
(volume anomaly, ADX, Stochastic, OBV, multi-horizon momentum).
"""

import logging

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


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — force de la tendance (0-100)."""
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm.abs() > 0), 0.0)
    
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.rolling(window=period).mean()
    return adx


def _compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator (%K, %D)."""
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    stoch_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def _compute_obv(df: pd.DataFrame) -> pd.Series:
    """On Balance Volume — accumulation/distribution."""
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)


def build_technical_features(db: Session, tickers: list[str] | None = None) -> pd.DataFrame:
    """Construit un DataFrame avec toutes les features techniques par ticker/date."""
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

    df = df.drop_duplicates(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"])

    all_features: list[pd.DataFrame] = []
    for ticker, group in df.groupby("ticker"):
        g = group.copy().set_index("timestamp").sort_index()

        # ── Moyennes mobiles ──
        g["sma_10"] = g["close"].rolling(window=10).mean()
        g["sma_20"] = g["close"].rolling(window=20).mean()
        g["sma_50"] = g["close"].rolling(window=50).mean()
        g["ema_10"] = g["close"].ewm(span=10, adjust=False).mean()
        g["ema_20"] = g["close"].ewm(span=20, adjust=False).mean()

        # ── RSI ──
        g["rsi_14"] = _compute_rsi(g["close"], 14)

        # ── MACD ──
        macd, macd_sig, macd_hist = _compute_macd(g["close"])
        g["macd"] = macd
        g["macd_signal"] = macd_sig
        g["macd_hist"] = macd_hist

        # ── Bollinger Bands ──
        bb_upper, bb_middle, bb_lower = _compute_bollinger(g["close"])
        g["bb_upper"] = bb_upper
        g["bb_middle"] = bb_middle
        g["bb_lower"] = bb_lower
        g["bb_width"] = (bb_upper - bb_lower) / bb_middle
        g["bb_pct"] = (g["close"] - bb_lower) / (bb_upper - bb_lower)

        # ── ATR ──
        g["atr_14"] = _compute_atr(g, 14)

        # ── Momentum & ROC ──
        g["momentum_10"] = g["close"] - g["close"].shift(10)
        g["roc_10"] = (g["close"] / g["close"].shift(10) - 1) * 100

        # ── Multi-horizon price changes (crypto momentum) ──
        g["price_change_1d"] = g["close"].pct_change(1)
        g["price_change_3d"] = g["close"].pct_change(3)
        g["price_change_7d"] = g["close"].pct_change(7)

        # ── Volatilité ──
        g["volatility_20"] = g["close"].rolling(window=20).std()
        g["volatility_ratio"] = g["volatility_20"] / g["atr_14"]

        # ── Ratios prix / MA ──
        g["price_sma20_ratio"] = g["close"] / g["sma_20"]
        g["price_sma50_ratio"] = g["close"] / g["sma_50"]

        # ── Volume features (crypto-specific) ──
        g["volume_sma20"] = g["volume"].rolling(window=20).mean()
        g["volume_ratio"] = g["volume"] / g["volume_sma20"]
        g["volume_change_1d"] = g["volume"].pct_change(1)

        # ── ADX (trend strength) ──
        g["adx_14"] = _compute_adx(g, 14)

        # ── Stochastic ──
        stoch_k, stoch_d = _compute_stochastic(g)
        g["stoch_k"] = stoch_k
        g["stoch_d"] = stoch_d

        # ── OBV ──
        g["obv"] = _compute_obv(g)
        g["obv_sma20"] = g["obv"].rolling(window=20).mean()
        g["obv_ratio"] = g["obv"] / g["obv_sma20"]

        # ── Williams %R ──
        highest_14 = g["high"].rolling(window=14).max()
        lowest_14 = g["low"].rolling(window=14).min()
        g["williams_r"] = -100 * (highest_14 - g["close"]) / (highest_14 - lowest_14)

        # ── True Range normalized ──
        g["tr_normalized"] = (g["high"] - g["low"]) / g["close"]

        g = g.reset_index()
        g["ticker"] = ticker
        all_features.append(g)

    result = pd.concat(all_features, ignore_index=True)
    logger.info("[Features] Built %d rows with %d features", len(result), result.shape[1])
    return result
