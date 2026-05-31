"""Flow indépendant — ingestion de données marché et news."""

import logging
from datetime import datetime

from prefect import flow, task

from trading.core.config import get_settings
from trading.core.database import db_session
from trading.ingestion.collector import MarketDataCollector
from trading.events.emitters import emit_market_price_updated, emit_news_batch_available

logger = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "JNJ", "LMT"]


@task(retries=3, retry_delay_seconds=10)
def fetch_and_store_prices(tickers: list[str]) -> dict[str, float]:
    """Récupère et stocke les prix via REST (fallback si websocket down)."""
    collector = MarketDataCollector()
    prices = collector.fetch_prices_finnhub(tickers)
    with db_session() as db:
        collector.store_prices(db, prices)
    for ticker, price in prices.items():
        emit_market_price_updated(ticker, price)
    return prices


@task(retries=3, retry_delay_seconds=10)
def fetch_and_store_news(tickers: list[str] | None = None) -> int:
    """Récupère et stocke les news (company-news par ticker + générales mappées)."""
    collector = MarketDataCollector()
    articles = collector.fetch_news_finnhub(tickers=tickers)
    with db_session() as db:
        count = collector.store_news(db, articles)
    if count > 0:
        emit_news_batch_available(count)
    return count


@flow(name="ingestion_flow", log_prints=True)
def ingestion_flow(tickers: list[str] | None = None) -> dict:
    """Flow d'ingestion — peut être déclenché par schedule ou manuellement."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    prices = fetch_and_store_prices(tickers)
    news_count = fetch_and_store_news()
    logger.info(f"Ingestion: {len(prices)} prix, {news_count} news")
    return {"prices": prices, "news_count": news_count}


if __name__ == "__main__":
    ingestion_flow()
