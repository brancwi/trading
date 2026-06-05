"""Ingère l'historique des funding rates depuis Binance Futures API."""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.core.models import FundingRate
from sqlalchemy.dialects.postgresql import insert as pg_insert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"

# Crypto symbols we track (mapped to Binance futures symbols)
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", 
           "DOTUSDT", "AVAXUSDT", "ETCUSDT", "LTCUSDT", "LINKUSDT",
           "XLMUSDT", "ALGOUSDT", "BCHUSDT", "UNIUSDT"]


def fetch_funding_rates(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch funding rate history from Binance Futures."""
    url = f"{BINANCE_FAPI}/fapi/v1/fundingRate"
    all_data = []
    current_start = start_ms
    
    while current_start < end_ms:
        params = {
            "symbol": symbol,
            "startTime": current_start,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current_start = data[-1]["fundingTime"] + 1
            logger.info(f"  {symbol}: fetched {len(data)} records, up to {datetime.fromtimestamp(data[-1]['fundingTime']/1000)}")
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            break
    
    return all_data


def ingest_symbol(symbol: str, start_date: str = "2022-01-01"):
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.utcnow().timestamp() * 1000)
    
    data = fetch_funding_rates(symbol, start_ms, end_ms)
    
    records = []
    for item in data:
        records.append({
            "timestamp": datetime.fromtimestamp(item["fundingTime"] / 1000),
            "symbol": symbol,
            "funding_rate": float(item["fundingRate"]),
            "mark_price": float(item.get("markPrice") or 0),
        })
    
    with db_session() as db:
        if records:
            stmt = pg_insert(FundingRate).values(records)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["timestamp", "symbol"]
            )
            db.execute(stmt)
            db.commit()
            logger.info(f"Inserted {len(records)} funding rates for {symbol}")
        else:
            logger.info(f"No funding rates for {symbol}")


def main():
    for symbol in SYMBOLS:
        logger.info(f"Processing {symbol}...")
        ingest_symbol(symbol)


if __name__ == "__main__":
    main()
