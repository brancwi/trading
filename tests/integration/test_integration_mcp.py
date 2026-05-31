"""Tests d'intégration — outils MCP avec DB réelle."""

import pytest

from trading.core.models import Signal, MarketData, News
from trading.mcp.server import (
    list_portfolios,
    get_portfolio_details,
    get_positions,
    get_trade_history,
    get_signals,
    get_market_data,
    get_news,
    get_balance_history,
    get_token_usage,
    get_audit_log,
    execute_sql_query,
)

pytestmark = pytest.mark.integration


class TestMCPPortfolios:
    def test_list_portfolios(self, test_portfolio):
        portfolios = list_portfolios()
        ids = [p["id"] for p in portfolios]
        assert "test-e2e" in ids

    def test_get_portfolio_details(self, test_portfolio):
        details = get_portfolio_details("test-e2e")
        assert "portfolio" in details
        assert details["portfolio"]["id"] == "test-e2e"
        assert details["positions"] == []
        assert details["recent_trades"] == []

    def test_get_positions_empty(self, test_portfolio):
        positions = get_positions("test-e2e")
        assert positions == []

    def test_get_trade_history_empty(self, test_portfolio):
        history = get_trade_history("test-e2e")
        assert history == []


class TestMCPSignalsAndData:
    def test_get_signals_after_insert(self, db, test_portfolio):
        signals = get_signals()
        initial_count = len(signals)

        signal = Signal(
            ticker="AAPL",
            action="BUY",
            sentiment=0.8,
            strength=0.9,
            confidence=0.85,
            source="test",
        )
        db.add(signal)
        db.commit()

        signals = get_signals()
        assert len(signals) == initial_count + 1
        assert signals[0]["ticker"] == "AAPL"

    def test_get_market_data_after_insert(self, db):
        db.query(MarketData).filter(MarketData.ticker == "AAPL").delete()
        db.commit()

        data = get_market_data("AAPL")
        assert data == []

        md = MarketData(
            ticker="AAPL",
            price=150.0,
            open_price=148.0,
            high=152.0,
            low=147.0,
            volume=1000000,
        )
        db.add(md)
        db.commit()

        data = get_market_data("AAPL")
        assert len(data) == 1
        assert data[0]["price"] == pytest.approx(150.0)

    def test_get_news_after_insert(self, db):
        news_list = get_news("AAPL")
        initial_count = len(news_list)

        news = News(
            source="test",
            ticker="AAPL",
            title="Apple News",
            description="Test description",
        )
        db.add(news)
        db.commit()

        news_list = get_news("AAPL")
        assert len(news_list) == initial_count + 1
        assert any(n["title"] == "Apple News" for n in news_list)


class TestMCPHistoryAndAudit:
    def test_get_balance_history_empty(self, test_portfolio):
        history = get_balance_history("test-e2e")
        assert history == []

    def test_get_token_usage_empty(self, test_portfolio):
        usage = get_token_usage(hours=24)
        assert usage["period_hours"] == 24
        assert usage["breakdown"] == []

    def test_get_audit_log_empty(self, test_portfolio):
        log = get_audit_log(hours=24)
        assert log == []


class TestMCPSQL:
    def test_execute_sql_query(self, test_portfolio):
        result = execute_sql_query(
            "SELECT id FROM portfolios WHERE id = 'test-e2e'"
        )
        assert "error" not in result
        assert result["count"] == 1
        assert result["rows"][0]["id"] == "test-e2e"
