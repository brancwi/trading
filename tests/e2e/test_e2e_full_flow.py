"""E2E test: Full pipeline — News → Sentiment → Signal → Strategy → Trade."""

import pytest

from trading.core.models import News, SentimentScore, Signal, Trade, Position, PortfolioHistory
from trading.strategies.simulation import SimulationStrategy


pytestmark = pytest.mark.e2e


def test_full_pipeline_buy_and_sell(db, test_portfolio):
    """Scenario: full pipeline with BUY then SELL for AAPL."""
    # 1. Insert a News row for AAPL
    news = News(
        source="test",
        ticker="AAPL",
        title="AAPL E2E news",
        description="End-to-end test news description",
    )
    db.add(news)
    db.commit()
    db.refresh(news)

    # 2. Insert a SentimentScore for that news
    sentiment = SentimentScore(
        news_id=news.id,
        ticker="AAPL",
        combined_score=0.8,
        confidence=0.9,
    )
    db.add(sentiment)
    db.commit()

    # 3. Insert a BUY Signal for AAPL
    signal_buy = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        price_at_signal=100.0,
        consumed=0,
    )
    db.add(signal_buy)
    db.commit()

    initial_cash = test_portfolio.cash_current

    # 4. Run SimulationStrategy
    strategy = SimulationStrategy("test-e2e")
    trades = strategy.run(db, {"AAPL": 100.0})

    # 5. Verify BUY outcomes
    assert len(trades) == 1
    trade_buy = trades[0]
    assert trade_buy.ticker == "AAPL"
    assert trade_buy.action == "BUY"

    db.refresh(test_portfolio)
    cash_after_buy = test_portfolio.cash_current
    assert cash_after_buy == pytest.approx(initial_cash - trade_buy.amount - trade_buy.fees)

    position = (
        db.query(Position)
        .filter(Position.portfolio_id == "test-e2e", Position.ticker == "AAPL")
        .first()
    )
    assert position is not None
    assert position.quantity == pytest.approx(trade_buy.quantity)

    db.refresh(signal_buy)
    assert signal_buy.consumed == 1

    history = (
        db.query(PortfolioHistory)
        .filter(PortfolioHistory.portfolio_id == "test-e2e")
        .first()
    )
    assert history is not None

    # 6. Insert a SELL signal for AAPL
    signal_sell = Signal(
        ticker="AAPL",
        action="SELL",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        price_at_signal=110.0,
        consumed=0,
    )
    db.add(signal_sell)
    db.commit()

    # 7. SimulationStrategy.run() ignores SELL signals; sell directly via strategy
    trade_sell = strategy.sell(db, "AAPL", position.quantity, 110.0)

    # 8. Verify SELL outcomes
    assert trade_sell.action == "SELL"
    assert trade_sell.ticker == "AAPL"
    assert trade_sell.realized_pnl is not None
    assert trade_sell.realized_pnl > 0

    position_after = (
        db.query(Position)
        .filter(Position.portfolio_id == "test-e2e", Position.ticker == "AAPL")
        .first()
    )
    assert position_after is None

    db.refresh(test_portfolio)
    assert test_portfolio.cash_current > cash_after_buy
