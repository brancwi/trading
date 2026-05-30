"""Prefect Event emitters — helpers to emit domain events from anywhere."""

import logging
from typing import Any

from prefect.events import emit_event

logger = logging.getLogger(__name__)


def emit_market_price_updated(ticker: str, price: float) -> None:
    """Émis quand un prix temps réel arrive par websocket."""
    try:
        emit_event(
            event="market.price.updated",
            resource={
                "prefect.resource.id": f"market.{ticker}",
                "prefect.resource.name": ticker,
            },
            payload={"ticker": ticker, "price": price},
        )
    except Exception as e:
        logger.warning(f"Failed to emit market.price.updated: {e}")


def emit_news_batch_available(count: int, sources: list[str] | None = None) -> None:
    """Émis quand de nouvelles news ont été ingérées."""
    try:
        emit_event(
            event="news.batch.available",
            resource={
                "prefect.resource.id": "finnhub.news",
                "prefect.resource.name": "Finnhub News",
            },
            payload={"count": count, "sources": sources or ["finnhub"]},
        )
    except Exception as e:
        logger.warning(f"Failed to emit news.batch.available: {e}")


def emit_signal_generated(signal_id: int, ticker: str, action: str, sentiment: float) -> None:
    """Émis quand le sentiment engine génère un signal."""
    try:
        emit_event(
            event="signal.generated",
            resource={
                "prefect.resource.id": f"signal.{signal_id}",
                "prefect.resource.name": f"{action} {ticker}",
            },
            payload={
                "signal_id": signal_id,
                "ticker": ticker,
                "action": action,
                "sentiment": sentiment,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to emit signal.generated: {e}")


def emit_hermes_command_received(command_id: int, command_type: str, portfolio_id: str | None) -> None:
    """Émis quand Hermes injecte une commande via l'API."""
    try:
        emit_event(
            event="hermes.command.received",
            resource={
                "prefect.resource.id": f"command.{command_id}",
                "prefect.resource.name": f"{command_type}",
            },
            payload={
                "command_id": command_id,
                "command_type": command_type,
                "portfolio_id": portfolio_id,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to emit hermes.command.received: {e}")


def emit_portfolio_updated(portfolio_id: str, total_value: float, pnl: float) -> None:
    """Émis quand un portefeuille est modifié (trade, commande, etc.)."""
    try:
        emit_event(
            event="portfolio.updated",
            resource={
                "prefect.resource.id": f"portfolio.{portfolio_id}",
                "prefect.resource.name": portfolio_id,
            },
            payload={
                "portfolio_id": portfolio_id,
                "total_value": total_value,
                "pnl": pnl,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to emit portfolio.updated: {e}")


def emit_trade_executed(trade_id: int, portfolio_id: str, ticker: str, action: str, amount: float) -> None:
    """Émis quand un trade est exécuté."""
    try:
        emit_event(
            event="trade.executed",
            resource={
                "prefect.resource.id": f"trade.{trade_id}",
                "prefect.resource.name": f"{action} {ticker}",
            },
            payload={
                "trade_id": trade_id,
                "portfolio_id": portfolio_id,
                "ticker": ticker,
                "action": action,
                "amount": amount,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to emit trade.executed: {e}")
