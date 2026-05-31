"""E2E test: Capital reserve and release flow."""

import pytest

from trading.core.models import Signal, Trade, CapitalMovement
from trading.mcp.server import reserve_capital, release_capital, get_capital_movements
from trading.strategies.simulation import SimulationStrategy


pytestmark = pytest.mark.e2e


def test_reserve_and_release_capital(db, test_portfolio):
    """Scenario: reserve capital, trade with reduced availability, then release."""
    # 1. Reserve capital
    result = reserve_capital("test-e2e", 2000.0, "E2E test reserve")
    assert result["status"] == "reserved"
    assert result["reserved_cash"] == 2000.0
    assert result["cash_available"] == 8000.0

    db.refresh(test_portfolio)
    assert test_portfolio.reserved_cash == 2000.0
    assert test_portfolio.cash_available == 8000.0

    # 2. Over-reserve should error
    err = reserve_capital("test-e2e", 9000.0)
    assert "error" in err

    # 3. Insert BUY signal and run strategy
    signal = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        price_at_signal=100.0,
        consumed=0,
    )
    db.add(signal)
    db.commit()

    strategy = SimulationStrategy("test-e2e")
    trades = strategy.run(db, {"AAPL": 100.0})

    assert len(trades) == 1
    trade = trades[0]
    # min(max_trade=500, cash_available - cash_min=8000-100=7900) = 500
    assert trade.amount == pytest.approx(500.0)

    db.refresh(test_portfolio)
    cash_after_trade = test_portfolio.cash_current
    assert cash_after_trade == pytest.approx(10000.0 - trade.amount - trade.fees)

    # 4. Release partial capital
    result_rel = release_capital("test-e2e", 1500.0)
    assert result_rel["status"] == "released"
    assert result_rel["reserved_cash"] == 500.0
    assert result_rel["cash_available"] == pytest.approx(cash_after_trade - 500.0)

    db.refresh(test_portfolio)
    assert test_portfolio.reserved_cash == 500.0
    # reserve/release themselves don't touch cash_current
    assert test_portfolio.cash_current == pytest.approx(cash_after_trade)

    # 5. Verify capital movements history
    movements = get_capital_movements("test-e2e", limit=10)
    assert len(movements) == 2
    types = {m["movement_type"] for m in movements}
    assert "reserve" in types
    assert "release" in types

    # 6. Final DB state: reserved_cash=500, cash_current unchanged by reserve/release
    assert test_portfolio.reserved_cash == 500.0
    assert test_portfolio.cash_current == pytest.approx(cash_after_trade)
