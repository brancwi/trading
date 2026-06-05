"""Télécharge l'historique complet crypto via Binance API (4+ ans).

Stocke dans market_data avec upsert. Ne re-télécharge jamais ce qui existe.
Usage:
    env TRADING_ENVIRONMENT=staging PYTHONPATH=src .venv/bin/python src/scripts/ingest_crypto_binance.py
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BINANCE_BASE = "https://api.binance.com/api/v3/klines"

# Mapping : nom yfinance → symbol Binance
CRYPTO_PAIRS = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "ADA-USD": "ADAUSDT",
    "XRP-USD": "XRPUSDT",
    "DOT-USD": "DOTUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "UNI-USD": "UNIUSDT",
    "LTC-USD": "LTCUSDT",
    "BCH-USD": "BCHUSDT",
    "ETC-USD": "ETCUSDT",
    "XLM-USD": "XLMUSDT",
    "ALGO-USD": "ALGOUSDT",
    "MATIC-USD": "MATICUSDT",
}


def _get_last_date(db, ticker: str) -> datetime | None:
    row = db.execute(
        text("SELECT MAX(timestamp) FROM market_data WHERE ticker = :ticker"),
        {"ticker": ticker}
    ).fetchone()
    return row[0] if row and row[0] else None


def fetch_binance_klines(symbol: str, start_ms: int, limit: int = 1000):
    """Récupère les klines Binance depuis start_ms."""
    url = f"{BINANCE_BASE}?symbol={symbol}&interval=1d&startTime={start_ms}&limit={limit}"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("[Binance] %s error: %s", symbol, e)
        return []


def download_full_history(ticker_yf: str, symbol_binance: str) -> int:
    """Télécharge tout l'historique disponible pour un crypto."""
    with db_session() as db:
        last_date = _get_last_date(db, ticker_yf)
        
        if last_date and (datetime.utcnow() - last_date).days <= 1:
            logger.info("[%s] Déjà à jour", ticker_yf)
            return 0
        
        start_ms = int((last_date + timedelta(days=1)).timestamp() * 1000) if last_date else 1609459200000  # 2021-01-01
        
        logger.info("[%s] Téléchargement depuis %s...", ticker_yf, datetime.fromtimestamp(start_ms/1000).date())
        
        total_inserted = 0
        while True:
            klines = fetch_binance_klines(symbol_binance, start_ms)
            if not klines:
                break
            
            for k in klines:
                ts = datetime.fromtimestamp(k[0] / 1000)
                db.execute(
                    text("""
                        INSERT INTO market_data (timestamp, ticker, open_price, high, low, price, volume, source)
                        VALUES (:ts, :ticker, :open, :high, :low, :close, :volume, 'binance')
                        ON CONFLICT (ticker, timestamp) DO UPDATE SET
                            open_price = EXCLUDED.open_price,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            price = EXCLUDED.price,
                            volume = EXCLUDED.volume,
                            source = EXCLUDED.source
                    """),
                    {
                        "ts": ts,
                        "ticker": ticker_yf,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )
                total_inserted += 1
            
            db.commit()
            logger.info("[%s] Batch: %d rows (up to %s)", ticker_yf, len(klines), datetime.fromtimestamp(klines[-1][0]/1000).date())
            
            # Prochaine requête commence après la dernière bougie
            start_ms = klines[-1][6] + 1  # close_time + 1ms
            
            # Arrête si on a atteint aujourd'hui
            if start_ms > int(datetime.utcnow().timestamp() * 1000):
                break
            
            time.sleep(0.2)  # Rate limit Binance
        
        logger.info("[%s] Total upsertés: %d", ticker_yf, total_inserted)
        return total_inserted


def main():
    print("="*70)
    print("  INGESTION CRYPTO — BINANCE API (4+ ans)")
    print("="*70)
    
    total = 0
    for ticker_yf, symbol_binance in CRYPTO_PAIRS.items():
        try:
            n = download_full_history(ticker_yf, symbol_binance)
            total += n
        except Exception as e:
            logger.error("[%s] Échec: %s", ticker_yf, e)
    
    print()
    print("="*70)
    print(f"  TOTAL: {total} rows upsertés")
    print("="*70)
    
    # Résumé
    with db_session() as db:
        result = db.execute(text("""
            SELECT ticker, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM market_data
            WHERE ticker LIKE '%-USD'
            GROUP BY ticker
            ORDER BY COUNT(*) DESC
        """)).fetchall()
        
        print("\nRésumé crypto:")
        for t, c, mn, mx in result:
            print(f"  {t:12s} {c:5d} rows  {mn.date()} → {mx.date()}")


if __name__ == "__main__":
    main()
