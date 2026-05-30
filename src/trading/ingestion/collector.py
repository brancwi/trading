"""Module d'ingestion de données marché et news."""

import logging
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.models import News, MarketData

logger = logging.getLogger(__name__)
settings = get_settings()


class MarketDataCollector:
    """Collecte les prix et news depuis les sources externes."""

    def __init__(self) -> None:
        self.finnhub_key = settings.finnhub_api_key
        self.http = httpx.Client(timeout=10.0)

    def fetch_news_finnhub(self, ticker: str = "AAPL") -> list[dict]:
        """Récupère les news Finnhub pour un ticker."""
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY manquante")
            return []
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": self.finnhub_key}
        try:
            resp = self.http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.error(f"Erreur Finnhub news: {e}")
            return []

    def fetch_prices_finnhub(self, tickers: list[str]) -> dict[str, float]:
        """Récupère les prix temps réel pour une liste de tickers."""
        prices: dict[str, float] = {}
        if not self.finnhub_key:
            return prices
        for ticker in tickers:
            try:
                url = f"https://finnhub.io/api/v1/quote"
                params = {"symbol": ticker, "token": self.finnhub_key}
                resp = self.http.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                prices[ticker] = data.get("c", 0.0)
            except Exception as e:
                logger.error(f"Erreur prix {ticker}: {e}")
        return prices

    def store_news(self, db: Session, articles: list[dict], source: str = "finnhub") -> int:
        """Persiste les news en base, évite les doublons par titre."""
        count = 0
        for art in articles:
            title = art.get("headline", art.get("title", ""))
            if not title:
                continue
            exists = db.query(News).filter(News.title == title).first()
            if exists:
                continue
            news = News(
                source=source,
                ticker=art.get("related", "GENERAL"),
                title=title,
                description=art.get("summary", art.get("description", "")),
                url=art.get("url", ""),
            )
            db.add(news)
            count += 1
        db.commit()
        logger.info(f"{count} nouvelles news stockées")
        return count

    def store_prices(self, db: Session, prices: dict[str, float]) -> int:
        """Persiste les prix en base."""
        for ticker, price in prices.items():
            md = MarketData(ticker=ticker, price=price)
            db.add(md)
        db.commit()
        logger.info(f"{len(prices)} prix stockés")
        return len(prices)
