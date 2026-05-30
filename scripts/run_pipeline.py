#!/usr/bin/env python3
"""Lance le pipeline de trading (mode CLI sans Prefect)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import logging

from trading.core.config import get_settings
from trading.core.database import db_session
from trading.ingestion.collector import MarketDataCollector
from trading.sentiment.analyzer import SentimentAnalyzer
from trading.execution.engine import ExecutionEngine
from trading.execution.commands import CommandProcessor
from trading.notifier.telegram import TelegramNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


def main():
    parser = argparse.ArgumentParser(description="Trading Pipeline V4")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL"])
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-sentiment", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()

    # 1. Commandes en attente (Hermes)
    with db_session() as db:
        processor = CommandProcessor(db)
        n = processor.process_pending()
        if n:
            logger.info(f"{n} commandes traitées")

    # 2. Ingestion
    prices = {}
    if not args.skip_ingestion:
        collector = MarketDataCollector()
        with db_session() as db:
            articles = collector.fetch_news_finnhub()
            collector.store_news(db, articles)
            prices = collector.fetch_prices_finnhub(args.tickers)
            collector.store_prices(db, prices)
        logger.info(f"Ingestion: {len(articles)} news, {len(prices)} prix")

    # 3. Sentiment
    if not args.skip_sentiment:
        analyzer = SentimentAnalyzer()
        with db_session() as db:
            count = analyzer.process_unprocessed_news(db)
        logger.info(f"Sentiment: {count} signaux générés")

    # 4. Exécution
    if not args.skip_execution and prices:
        with db_session() as db:
            engine = ExecutionEngine(db)
            results = engine.run_all(prices)
            notifier = TelegramNotifier()
            for port_id, trades in results.items():
                for trade in trades:
                    notifier.notify_trade(
                        portfolio=port_id,
                        action=trade.action,
                        ticker=trade.ticker,
                        qty=trade.quantity,
                        price=trade.price,
                        amount=trade.amount,
                        pnl=trade.realized_pnl,
                        db=db,
                    )
        logger.info(f"Exécution: {results}")

    logger.info("Pipeline terminé")


if __name__ == "__main__":
    main()
