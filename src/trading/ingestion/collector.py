"""Module d'ingestion de données marché et news."""

import logging
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.models import News, MarketData
from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)
settings = get_settings()


class MarketDataCollector:
    """Collecte les prix et news depuis les sources externes."""

    def __init__(self) -> None:
        self.finnhub_key = settings.finnhub_api_key
        self.http = httpx.Client(timeout=10.0)

    def fetch_news_finnhub(self, tickers: list[str] | None = None) -> list[dict]:
        """Récupère les news Finnhub — company-news par ticker + générales mappées."""
        if not self.finnhub_key:
            logger.warning("FINNHUB_API_KEY manquante")
            return []

        from datetime import timedelta
        from trading.ingestion.ticker_mapper import map_general_news_to_tickers

        all_articles: list[dict] = []
        seen_ids: set[int] = set()

        # ── 1. News spécifiques par ticker (company-news) ──
        if tickers:
            from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")
            for ticker in tickers:
                try:
                    url = "https://finnhub.io/api/v1/company-news"
                    params = {
                        "symbol": ticker,
                        "from": from_date,
                        "to": to_date,
                        "token": self.finnhub_key,
                    }
                    resp = self.http.get(url, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        for art in data:
                            art_id = art.get("id")
                            if art_id and art_id not in seen_ids:
                                art["related"] = ticker
                                seen_ids.add(art_id)
                                all_articles.append(art)
                except Exception as e:
                    logger.warning(f"Erreur company-news {ticker}: {e}")

        # ── 2. News générales + mapping mots-clés ──
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "token": self.finnhub_key}
            resp = self.http.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                for art in data:
                    art_id = art.get("id")
                    if art_id and art_id not in seen_ids:
                        seen_ids.add(art_id)
                        # Si la news a déjà un ticker non vide, on la garde telle quelle
                        if art.get("related"):
                            all_articles.append(art)
                        else:
                            # Extraction des tickers depuis le texte
                            mapped_tickers = map_general_news_to_tickers(
                                art.get("headline", art.get("title", "")),
                                art.get("summary", art.get("description", "")),
                            )
                            if mapped_tickers:
                                # On duplique la news pour chaque ticker trouvé
                                for mapped_ticker in mapped_tickers[:3]:  # max 3 tickers
                                    cloned = dict(art)
                                    cloned["related"] = mapped_ticker
                                    all_articles.append(cloned)
                            else:
                                # Sans mapping → news macro, on la garne avec ticker GENERAL
                                art["related"] = "GENERAL"
                                all_articles.append(art)
        except Exception as e:
            logger.error(f"Erreur Finnhub general news: {e}")

        MonitorService.log_event(
            channel="news_finnhub",
            source="finnhub",
            payload=all_articles,
            metadata={
                "article_count": len(all_articles),
                "tickers": tickers,
                "unique_ids": len(seen_ids),
            },
        )
        return all_articles

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
        MonitorService.log_event(
            channel="prices_finnhub",
            source="finnhub.quote",
            payload=prices,
            metadata={
                "price_count": len(prices),
                "tickers_requested": len(tickers),
                "tickers_found": list(prices.keys()),
            },
        )
        return prices

    def store_news(self, db: Session, articles: list[dict], source: str = "finnhub") -> int:
        """Persiste les news en base, évite les doublons par (titre, ticker)."""
        count = 0
        for art in articles:
            title = art.get("headline", art.get("title", ""))
            if not title:
                continue
            ticker = art.get("related", "GENERAL") or "GENERAL"
            exists = db.query(News).filter(
                News.title == title, News.ticker == ticker
            ).first()
            if exists:
                continue
            news = News(
                source=source,
                ticker=ticker,
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
