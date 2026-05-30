"""Asyncio Event Listener — websocket + polling bridge to Prefect Events."""

import asyncio
import json
import logging
from datetime import datetime, timedelta

import aiohttp
import websockets

from trading.core.config import get_settings
from trading.core.database import db_session
from trading.ingestion.collector import MarketDataCollector
from trading.events.emitters import (
    emit_market_price_updated,
    emit_news_batch_available,
    emit_hermes_command_received,
)
from trading.core.models import Command

logger = logging.getLogger(__name__)
settings = get_settings()

TICKERS = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "PLTR", "JNJ", "LMT"]


class EventListener:
    """Écoute les sources de données et émet des Prefect Events."""

    def __init__(self) -> None:
        self.finnhub_key = settings.finnhub_api_key
        self.http: aiohttp.ClientSession | None = None
        self._last_news_check: datetime | None = None
        self._last_command_check: datetime | None = None

    async def run(self) -> None:
        """Lance toutes les coroutines d'écoute en parallèle."""
        self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        try:
            await asyncio.gather(
                self._websocket_finnhub_prices(),
                self._polling_finnhub_news(),
                self._polling_api_commands(),
            )
        finally:
            await self.http.close()

    # ------------------------------------------------------------------
    # WebSocket Finnhub — prix temps réel
    # ------------------------------------------------------------------
    async def _websocket_finnhub_prices(self) -> None:
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY manquante — websocket désactivé")
            return
        uri = f"wss://ws.finnhub.io?token={self.finnhub_key}"
        while True:
            try:
                async with websockets.connect(uri) as ws:
                    # Subscribe à tous les tickers
                    for ticker in TICKERS:
                        await ws.send(json.dumps({"type": "subscribe", "symbol": ticker}))
                    logger.info(f"WebSocket connecté — {len(TICKERS)} tickers")
                    async for message in ws:
                        await self._handle_ws_message(json.loads(message))
            except Exception as e:
                logger.error(f"WebSocket erreur: {e} — reconnexion dans 5s")
                await asyncio.sleep(5)

    async def _handle_ws_message(self, msg: dict) -> None:
        if msg.get("type") != "trade":
            return
        for trade in msg.get("data", []):
            ticker = trade.get("s")
            price = trade.get("p")
            if not ticker or not price:
                continue
            # Persiste en DB
            try:
                with db_session() as db:
                    from trading.core.models import MarketData
                    db.add(MarketData(ticker=ticker, price=price))
            except Exception as e:
                logger.warning(f"DB write failed for {ticker}: {e}")
            # Émet event Prefect
            emit_market_price_updated(ticker, price)

    # ------------------------------------------------------------------
    # Polling intelligent — news
    # ------------------------------------------------------------------
    async def _polling_finnhub_news(self) -> None:
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY manquante — news polling désactivé")
            return
        collector = MarketDataCollector()
        # On remplace le client httpx par aiohttp pour le contexte async
        while True:
            try:
                await asyncio.sleep(60)  # toutes les minutes
                # Récupère les news via REST
                articles = await self._fetch_news()
                if articles:
                    with db_session() as db:
                        count = collector.store_news(db, articles)
                    if count > 0:
                        emit_news_batch_available(count)
            except Exception as e:
                logger.error(f"News polling erreur: {e}")

    async def _fetch_news(self) -> list[dict]:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": self.finnhub_key}
        async with self.http.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data if isinstance(data, list) else []

    # ------------------------------------------------------------------
    # Polling — commandes Hermes en attente
    # ------------------------------------------------------------------
    async def _polling_api_commands(self) -> None:
        while True:
            try:
                await asyncio.sleep(30)  # toutes les 30s
                with db_session() as db:
                    pending = (
                        db.query(Command)
                        .filter(Command.status == "pending")
                        .order_by(Command.created_at)
                        .all()
                    )
                    for cmd in pending:
                        emit_hermes_command_received(cmd.id, cmd.command_type, cmd.portfolio_id)
            except Exception as e:
                logger.error(f"Command polling erreur: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    listener = EventListener()
    asyncio.run(listener.run())
