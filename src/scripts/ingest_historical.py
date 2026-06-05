"""Ingestion historique robuste — upsert dans PostgreSQL avec cache.

Ne re-télécharge jamais ce qui est déjà en base. Met à jour incrémentalement.
Usage:
    env TRADING_ENVIRONMENT=staging PYTHONPATH=src .venv/bin/python src/scripts/ingest_historical.py
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Univers d'investissement
ACTIONS_US = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JNJ",
    "VZ", "KO", "PG", "UNH", "HD", "MA", "DIS", "NKE", "CRM", "HON",
    "NEE", "ABT", "PFE", "MRK", "LLY", "TMO", "AVGO", "QCOM", "TXN",
    "ADBE", "PYPL", "INTC", "AMD", "NFLX", "UBER", "LYFT", "ZM",
    "PLTR", "SNOW", "SHOP", "XYZ", "ROKU", "TWLO", "DDOG", "CRWD",
    "OKTA", "FSLY", "NET", "DDOG", "PLTR", "SNOW", "ZM",
]

# Semi-conducteurs + Défense européens
ACTIONS_EU = [
    # Semi-conducteurs
    "ASML.AS",      # ASML (Pays-Bas) — lithographie
    "STMPA.PA",     # STMicroelectronics (FR/IT)
    "IFX.DE",       # Infineon (Allemagne)
    "SOI.PA",       # Soitec (France)
    "BESI.AS",      # BE Semiconductor (Pays-Bas)
    "ASM.AS",       # ASM International (Pays-Bas)
    # Défense
    "HO.PA",        # Thales (France)
    "SAF.PA",       # Safran (France)
    "LDO.MI",       # Leonardo (Italie)
    "RHM.DE",       # Rheinmetall (Allemagne)
    "BA.L",         # BAE Systems (UK)
    "AIR.PA",       # Airbus (France)
    # Autres
    "MT.AS",        # ArcelorMittal (Pays-Bas)
    "DSY.PA",       # Dassault Systèmes (France)
]

CRYPTOS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD",
    "DOT-USD", "MATIC-USD", "AVAX-USD", "LINK-USD", "UNI-USD",
    "LTC-USD", "BCH-USD", "ETC-USD", "XLM-USD", "ALGO-USD",
]

ALL_TICKERS = ACTIONS_US + ACTIONS_EU + CRYPTOS


def _get_last_date(db, ticker: str) -> datetime | None:
    """Retourne la date la plus récente pour un ticker en DB."""
    row = db.execute(
        text("SELECT MAX(timestamp) FROM market_data WHERE ticker = :ticker"),
        {"ticker": ticker}
    ).fetchone()
    return row[0] if row and row[0] else None


def _upsert_ohlcv(db, ticker: str, df: pd.DataFrame) -> int:
    """Insère ou met à jour les données OHLCV pour un ticker."""
    if df.empty:
        return 0
    
    inserted = 0
    for _, row in df.iterrows():
        ts = pd.to_datetime(row["date"])
        db.execute(
            text("""
                INSERT INTO market_data (timestamp, ticker, open_price, high, low, price, volume, source)
                VALUES (:ts, :ticker, :open, :high, :low, :close, :volume, 'yfinance')
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
                "ticker": ticker,
                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "volume": int(float(row["volume"])) if pd.notna(row["volume"]) else None,
            }
        )
        inserted += 1
    
    db.commit()
    return inserted


def download_and_store(ticker: str, period: str = "4y") -> int:
    """Télécharge l'historique yfinance et stocke en DB (upsert)."""
    with db_session() as db:
        last_date = _get_last_date(db, ticker)
        
        if last_date:
            # On a déjà des données — calcule la période manquante
            days_since = (datetime.utcnow() - last_date).days
            if days_since <= 1:
                logger.info("[%s] Déjà à jour (dernier: %s)", ticker, last_date.date())
                return 0
            # Télécharge depuis la dernière date
            start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info("[%s] Mise à jour depuis %s", ticker, start)
            df = yf.download(ticker, start=start, progress=False)
        else:
            # Pas de données — télécharge tout
            logger.info("[%s] Téléchargement complet (%s)", ticker, period)
            df = yf.download(ticker, period=period, progress=False)
        
        if df.empty:
            logger.warning("[%s] Aucune donnée reçue", ticker)
            return 0
        
        # Normalise les colonnes
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(col).strip() if col[1] not in ['nan', 'NaN'] else col[0] for col in df.columns.values]
            rename_map = {}
            for c in df.columns:
                if 'Open' in c: rename_map[c] = 'open'
                elif 'High' in c: rename_map[c] = 'high'
                elif 'Low' in c: rename_map[c] = 'low'
                elif 'Close' in c: rename_map[c] = 'close'
                elif 'Volume' in c: rename_map[c] = 'volume'
                elif 'Date' in c or 'index' in c.lower(): rename_map[c] = 'date'
            df = df.rename(columns=rename_map)
        else:
            df = df.rename(columns={
                'Open': 'open', 'High': 'high', 'Low': 'low',
                'Close': 'close', 'Volume': 'volume', 'Date': 'date'
            })
        
        # Garde uniquement les colonnes nécessaires
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        inserted = _upsert_ohlcv(db, ticker, df)
        logger.info("[%s] %d rows upsertés", ticker, inserted)
        return inserted


def main():
    print("="*70)
    print("  INGESTION HISTORIQUE — US + EU + CRYPTO")
    print("="*70)
    print(f"Tickers: {len(ALL_TICKERS)} ({len(ACTIONS_US)} US + {len(ACTIONS_EU)} EU + {len(CRYPTOS)} cryptos)")
    print()
    
    total = 0
    for ticker in ALL_TICKERS:
        try:
            n = download_and_store(ticker, period="4y")
            total += n
        except Exception as e:
            logger.error("[%s] Échec: %s", ticker, e)
    
    print()
    print("="*70)
    print(f"  TOTAL: {total} rows upsertés")
    print("="*70)
    
    # Résumé
    with db_session() as db:
        result = db.execute(text("""
            SELECT ticker, COUNT(*), MIN(timestamp), MAX(timestamp)
            FROM market_data
            WHERE ticker = ANY(:tickers)
            GROUP BY ticker
            ORDER BY COUNT(*) DESC
        """), {"tickers": ALL_TICKERS}).fetchall()
        
        print("\nRésumé par ticker:")
        for t, c, mn, mx in result:
            print(f"  {t:12s} {c:5d} rows  {mn.date()} → {mx.date()}")


if __name__ == "__main__":
    main()
