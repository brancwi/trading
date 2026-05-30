"""Flow Prefect - orchestration complète du pipeline."""

import logging
from datetime import datetime

from prefect import flow, task
from prefect.tasks import task_input_hash

from trading.core.config import get_settings
from trading.core.database import db_session
from trading.ingestion.collector import MarketDataCollector
from trading.sentiment.analyzer import SentimentAnalyzer
from trading.execution.engine import ExecutionEngine
from trading.execution.commands import CommandProcessor
from trading.notifier.telegram import TelegramNotifier
from trading.core.models import PortfolioHistory

logger = logging.getLogger(__name__)
settings = get_settings()


@task(retries=2, retry_delay_seconds=30)
def fetch_data(tickers: list[str]) -> dict[str, float]:
    """Étape 1: ingestion données."""
    collector = MarketDataCollector()
    with db_session() as db:
        # News
        articles = collector.fetch_news_finnhub()
        collector.store_news(db, articles)
        # Prix
        prices = collector.fetch_prices_finnhub(tickers)
        collector.store_prices(db, prices)
    logger.info(f"Data fetched: {len(articles)} news, {len(prices)} prices")
    return prices


@task(retries=1)
def analyze_sentiment() -> int:
    """Étape 2: analyse des news non traitées."""
    analyzer = SentimentAnalyzer()
    with db_session() as db:
        count = analyzer.process_unprocessed_news(db)
    return count


@task
def execute_commands() -> int:
    """Traite les commandes en file (Hermes)."""
    with db_session() as db:
        processor = CommandProcessor(db)
        n = processor.process_pending()
    return n


@task
def run_strategies(prices: dict[str, float]) -> dict:
    """Étape 3: exécute toutes les stratégies."""
    with db_session() as db:
        engine = ExecutionEngine(db)
        results = engine.run_all(prices)
        # Notifier les trades
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
    return {k: len(v) for k, v in results.items()}


@task
def notify_summary() -> None:
    """Résumé si heure pile (ex: 20h)."""
    now = datetime.now()
    if now.hour == 20 and now.minute < 10:
        with db_session() as db:
            notifier = TelegramNotifier()
            subq = db.query(
                PortfolioHistory.portfolio_id,
                PortfolioHistory.timestamp,
                PortfolioHistory.total_value
            ).distinct(PortfolioHistory.portfolio_id).order_by(
                PortfolioHistory.portfolio_id, PortfolioHistory.timestamp.desc()
            ).subquery()
            rows = db.query(subq).all()
            values = {r.portfolio_id: r.total_value for r in rows}
            notifier.notify_summary(values, db=db)


@flow(name="trading_pipeline", log_prints=True)
def trading_flow(tickers: list[str] | None = None):
    """Flow principal - orchestration complète."""
    if tickers is None:
        tickers = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "JNJ", "LMT"]

    # 1. Traiter d'abord les commandes Hermes
    execute_commands()

    # 2. Ingestion
    prices = fetch_data(tickers)

    # 3. Sentiment
    signals = analyze_sentiment()

    # 4. Exécution stratégies
    trades = run_strategies(prices)

    # 5. Résumé horaire si besoin
    notify_summary()

    return {
        "prices_count": len(prices),
        "signals_generated": signals,
        "trades_by_strategy": trades,
    }


if __name__ == "__main__":
    trading_flow()
