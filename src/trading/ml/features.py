"""Feature engineering — indicateurs techniques sur market_data."""

import logging
from typing import Any

import numpy as np

from trading.core.database import db_session
from trading.core.models import MarketData

logger = logging.getLogger(__name__)


def _get_prices_and_volumes(ticker: str, limit: int = 100) -> dict[str, Any]:
    """Récupère les N dernières lignes de market_data pour un ticker."""
    with db_session() as db:
        rows = (
            db.query(MarketData)
            .filter(MarketData.ticker == ticker)
            .order_by(MarketData.timestamp.desc())
            .limit(limit)
            .all()
        )
        if not rows:
            return {}
        rows = list(reversed(rows))  # chronologique
        return {
            "timestamps": [r.timestamp for r in rows],
            "prices": np.array([r.price for r in rows], dtype=float),
            "volumes": np.array([r.volume or 0 for r in rows], dtype=float),
            "highs": np.array([r.high or r.price for r in rows], dtype=float),
            "lows": np.array([r.low or r.price for r in rows], dtype=float),
        }


def sma(prices: np.ndarray, window: int) -> float | None:
    """Simple Moving Average."""
    if len(prices) < window:
        return None
    return float(np.mean(prices[-window:]))


def rsi(prices: np.ndarray, window: int = 14) -> float | None:
    """Relative Strength Index (0-100)."""
    if len(prices) < window + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-window:])
    avg_loss = np.mean(losses[-window:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def momentum(prices: np.ndarray, window: int = 5) -> float | None:
    """Momentum = (price_now - price_before) / price_before."""
    if len(prices) < window + 1:
        return None
    return float((prices[-1] - prices[-(window + 1)]) / prices[-(window + 1)])


def bollinger_band_width(prices: np.ndarray, window: int = 20) -> float | None:
    """Bollinger Band Width = (upper - lower) / sma."""
    if len(prices) < window:
        return None
    recent = prices[-window:]
    mean = np.mean(recent)
    std = np.std(recent)
    if mean == 0:
        return None
    return float((2 * std) / mean)


def volume_avg(volumes: np.ndarray, window: int = 10) -> float | None:
    if len(volumes) < window:
        return None
    return float(np.mean(volumes[-window:]))


class FeatureEngine:
    """Calcule les features techniques pour un ticker donné."""

    def compute(self, ticker: str) -> dict[str, float | None]:
        data = _get_prices_and_volumes(ticker, limit=50)
        if not data:
            return {}
        prices = data["prices"]
        volumes = data["volumes"]
        return {
            "sma_10": sma(prices, 10),
            "sma_20": sma(prices, 20),
            "rsi_14": rsi(prices, 14),
            "momentum_5": momentum(prices, 5),
            "momentum_10": momentum(prices, 10),
            "bb_width_20": bollinger_band_width(prices, 20),
            "volume_avg_10": volume_avg(volumes, 10),
            "price": float(prices[-1]),
        }
