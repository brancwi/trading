"""E2E test: Command execution via API (liquidate / resume)."""

import pytest

from trading.core.models import Position, Trade, Command


pytestmark = pytest.mark.e2e

API_KEY = "dev-secret-change-me"


def test_liquidate_and_resume_portfolio(client, db, test_portfolio):
    """Scenario: create position, liquidate via API, then resume."""
    # Clean up any stale commands from prior tests to avoid side effects
    db.query(Command).filter(Command.portfolio_id == "test-e2e").delete()
    db.commit()

    # 1. Create a Position for test-e2e
    position = Position(
        portfolio_id="test-e2e",
        ticker="AAPL",
        quantity=10.0,
        avg_entry_price=100.0,
        current_price=100.0,
        current_value=1000.0,
    )
    db.add(position)
    db.commit()

    # 2. POST /portfolios/test-e2e/liquidate
    response = client.post(
        "/portfolios/test-e2e/liquidate",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "liquidating"

    # 3. Query DB: positions for test-e2e should be empty
    positions = db.query(Position).filter(Position.portfolio_id == "test-e2e").all()
    assert len(positions) == 0

    # 4. Query DB: portfolio status is "liquidated"
    db.refresh(test_portfolio)
    assert test_portfolio.status == "liquidated"

    # 5. Query DB: trades include a SELL for AAPL
    trades = db.query(Trade).filter(Trade.portfolio_id == "test-e2e").all()
    sell_trades = [t for t in trades if t.ticker == "AAPL" and t.action == "SELL"]
    assert len(sell_trades) >= 1

    # 6. POST /portfolios/test-e2e/resume
    response = client.post(
        "/portfolios/test-e2e/resume",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"

    # 7. Query DB: portfolio status is "active"
    db.refresh(test_portfolio)
    assert test_portfolio.status == "active"

    # Cleanup commands created by this test
    db.query(Command).filter(Command.portfolio_id == "test-e2e").delete()
    db.commit()
