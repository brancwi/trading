"""Ingère VIX, DXY, TNX depuis yfinance."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.core.models import MacroIndicator
from sqlalchemy.dialects.postgresql import insert as pg_insert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDICATORS = {
    "VIX": "^VIX",
    "DXY": "DX-Y.NYB",
    "TNX": "^TNX",  # 10-Year Treasury Yield
}


def ingest_indicator(name: str, symbol: str):
    logger.info(f"Fetching {name} ({symbol})...")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5y", interval="1d")
    
    if hist.empty:
        logger.warning(f"No data for {name}")
        return
    
    records = []
    for date, row in hist.iterrows():
        records.append({
            "date": pd.Timestamp(date).tz_localize(None).normalize(),
            "indicator": name,
            "value": float(row["Close"]),
        })
    
    with db_session() as db:
        if records:
            stmt = pg_insert(MacroIndicator).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=["date", "indicator"])
            db.execute(stmt)
            db.commit()
            logger.info(f"Inserted {len(records)} records for {name}")


def main():
    for name, symbol in INDICATORS.items():
        ingest_indicator(name, symbol)


if __name__ == "__main__":
    main()
