"""Ingère l'historique Fear & Greed Index depuis alternative.me."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.core.models import FearGreed
from sqlalchemy import insert

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_fear_greed(limit: int = 1000):
    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()["data"]

    records = []
    for item in data:
        ts = datetime.fromtimestamp(int(item["timestamp"]))
        records.append({
            "date": ts,
            "value": int(item["value"]),
            "classification": item["value_classification"],
        })

    with db_session() as db:
        # Upsert via on_conflict_do_nothing
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        if records:
            stmt = pg_insert(FearGreed).values(records)
            stmt = stmt.on_conflict_do_nothing(index_elements=["date"])
            db.execute(stmt)
            db.commit()
            logger.info(f"Inserted {len(records)} Fear & Greed records")
        else:
            logger.info("No records to insert")


if __name__ == "__main__":
    ingest_fear_greed()
