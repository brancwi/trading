"""Téléchargement et stockage de l'historique complet des prix via Yahoo Finance.

Conformément au deep-research-report, on accumule l'historique OHLCV
pour chaque ticker surveillé plutôt que de se contenter des snapshots
temps réel.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

from trading.core.models import MarketData

logger = logging.getLogger(__name__)

# Tickers surveillés par le système de trading
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JNJ", "JPM", "V",
    "PG", "UNH", "HD", "MA", "BAC",
    "ABBV", "PFE", "KO", "PEP", "LLY",
    "MRK", "TMO", "COST", "AVGO", "DIS",
    "WMT", "ABT", "ADBE", "CRM", "ACN",
    "VZ", "DHR", "TXN", "NKE", "BMY",
    "QCOM", "NEE", "PM", "RTX", "HON",
    "LOW", "SPGI", "UNP", "UPS", "IBM",
    "GS", "CAT", "MS", "CVX", "XOM",
    "PLTR", "LMT",
]


def download_history(
    tickers: list[str] | None = None,
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, dict]:
    """Télécharge l'historique OHLCV pour une liste de tickers.

    Returns:
        Dict mapping ticker -> {"data": DataFrame, "info": dict}
    """
    tickers = tickers or DEFAULT_TICKERS
    results: dict[str, dict] = {}
    for ticker in tickers:
        try:
            logger.info(f"[Historical] Downloading {ticker} ({period}, {interval})")
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                logger.warning(f"[Historical] No data for {ticker}")
                continue
            results[ticker] = {"data": hist, "info": stock.info}
        except Exception as e:
            logger.error(f"[Historical] Error downloading {ticker}: {e}")
    return results


def _to_native(value):
    """Convertit numpy/pandas types en types Python natifs pour SQLAlchemy."""
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if hasattr(value, "to_pydatetime"):  # pandas Timestamp
        dt = value.to_pydatetime()
        # Supprimer le timezone pour éviter les problèmes PostgreSQL
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    return value


def store_history(db: Session, results: dict[str, dict]) -> int:
    """Persiste l'historique téléchargé dans la table MarketData.

    Les données sont accumulées (INSERT) — pas d'écrasement.
    On évite les doublons par (ticker, timestamp).
    """
    total = 0
    for ticker, result in results.items():
        hist = result["data"]
        ticker_rows = 0
        for ts, row in hist.iterrows():
            dt = _to_native(ts)
            # Vérifier doublon
            exists = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker, MarketData.timestamp == dt)
                .first()
            )
            if exists:
                continue
            close_val = _to_native(row.get("Close"))
            open_val = _to_native(row.get("Open"))
            high_val = _to_native(row.get("High"))
            low_val = _to_native(row.get("Low"))
            volume_val = _to_native(row.get("Volume"))
            md = MarketData(
                ticker=ticker,
                timestamp=dt,
                price=round(float(close_val), 4) if close_val is not None else None,
                open_price=round(float(open_val), 4) if open_val is not None else None,
                high=round(float(high_val), 4) if high_val is not None else None,
                low=round(float(low_val), 4) if low_val is not None else None,
                volume=int(volume_val) if volume_val is not None and not pd.isna(volume_val) else None,
                change_pct=round(
                    (close_val - open_val) / open_val * 100, 4
                ) if open_val and close_val else None,
                source="yfinance",
            )
            db.add(md)
            ticker_rows += 1
        db.commit()
        total += ticker_rows
        logger.info("[Historical] %s: %d new rows stored", ticker, ticker_rows)
    return total


def backfill_all(
    db: Session,
    tickers: list[str] | None = None,
    period: str = "2y",
    interval: str = "1d",
) -> int:
    """Pipeline complète : télécharge + stocke l'historique."""
    results = download_history(tickers=tickers, period=period, interval=interval)
    if not results:
        logger.warning("[Historical] No data downloaded")
        return 0
    return store_history(db, results)


if __name__ == "__main__":
    import sys
    from trading.core.database import db_session

    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    with db_session() as db:
        count = backfill_all(db, tickers=tickers)
        print(f"Stored {count} historical rows")
