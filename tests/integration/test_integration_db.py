"""Tests d'intégration — initialisation DB et relations ORM."""

import pytest
from sqlalchemy import text

from trading.core.database import engine, init_db
from trading.core.models import Portfolio, Position, Trade, News, SentimentScore

pytestmark = pytest.mark.integration


class TestDatabaseInit:
    def test_init_db_creates_all_tables(self):
        """Vérifie que init_db() crée toutes les tables attendues."""
        expected_tables = {
            "portfolios",
            "positions",
            "trades",
            "signals",
            "sentiment_scores",
            "capital_movements",
            "portfolio_history",
            "commands",
            "alerts",
            "monitoring_metrics",
            "audit_log",
            "token_usage_log",
            "news",
            "market_data",
        }
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            )
            tables = {row[0] for row in result}
        for table in expected_tables:
            assert table in tables, f"Table manquante : {table}"


class TestORMRelations:
    def test_portfolio_position_relationship(self, db):
        """Portfolio → Position : création et lecture via relation."""
        portfolio = Portfolio(
            id="test-port-rel",
            name="Relation Test",
            strategy_type="simulation",
            cash_initial=1000.0,
            cash_current=1000.0,
        )
        db.add(portfolio)
        db.commit()

        position = Position(
            portfolio_id="test-port-rel",
            ticker="TSLA",
            quantity=10.0,
            avg_entry_price=200.0,
        )
        db.add(position)
        db.commit()

        db.refresh(portfolio)
        assert len(portfolio.positions) == 1
        assert portfolio.positions[0].ticker == "TSLA"

        # Cleanup
        db.delete(portfolio)
        db.commit()

    def test_portfolio_trade_relationship(self, db, test_portfolio):
        """Portfolio → Trade : création et lecture via relation."""
        trade = Trade(
            portfolio_id=test_portfolio.id,
            ticker="AAPL",
            action="BUY",
            quantity=5.0,
            price=150.0,
            amount=750.0,
            fees=1.0,
        )
        db.add(trade)
        db.commit()

        db.refresh(test_portfolio)
        tickers = [t.ticker for t in test_portfolio.trades]
        assert "AAPL" in tickers

    def test_news_sentiment_score_fk(self, db):
        """News → SentimentScore : la FK news_id est résolue correctement."""
        news = News(
            source="test",
            ticker="AAPL",
            title="Test News",
            description="Test description",
        )
        db.add(news)
        db.commit()
        db.refresh(news)

        score = SentimentScore(
            news_id=news.id,
            ticker="AAPL",
            combined_score=0.75,
            confidence=0.9,
        )
        db.add(score)
        db.commit()

        result = (
            db.query(SentimentScore).filter(SentimentScore.news_id == news.id).first()
        )
        assert result is not None
        assert result.combined_score == pytest.approx(0.75)

        # Cleanup
        db.delete(score)
        db.flush()
        db.delete(news)
        db.commit()

    def test_portfolio_cascade_delete_positions(self, db):
        """Cascade delete : supprimer un portfolio supprime ses positions."""
        portfolio = Portfolio(
            id="test-cascade",
            name="Cascade Test",
            strategy_type="simulation",
            cash_initial=1000.0,
            cash_current=1000.0,
        )
        db.add(portfolio)
        db.commit()

        position = Position(
            portfolio_id="test-cascade",
            ticker="MSFT",
            quantity=5.0,
            avg_entry_price=300.0,
        )
        db.add(position)
        db.commit()

        db.delete(portfolio)
        db.commit()

        remaining = (
            db.query(Position).filter(Position.portfolio_id == "test-cascade").first()
        )
        assert remaining is None
