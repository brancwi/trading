"""Unit tests for StrategyBase."""

import pytest

from trading.strategies.base import StrategyBase
from trading.core.models import Portfolio, Position, Trade, PortfolioHistory


class DummyStrategy(StrategyBase):
    """Concrete subclass for testing the abstract base."""

    def run(self, db, prices):
        return []


@pytest.fixture
def strategy(test_portfolio):
    return DummyStrategy(portfolio_id=test_portfolio.id)


@pytest.mark.unit
def test_get_portfolio(strategy, db, test_portfolio):
    port = strategy.get_portfolio(db)
    assert port.id == test_portfolio.id
    assert port.cash_initial == 10000.0


@pytest.mark.unit
def test_buy_deducts_cash_and_creates_position(strategy, db, test_portfolio):
    trade = strategy.buy(db, ticker="AAPL", quantity=10, price=50)

    port = strategy.get_portfolio(db)
    assert port.cash_current == 10000 - (10 * 50 + 1.0)
    assert trade.action == "BUY"
    assert trade.amount == 500.0
    assert trade.fees == 1.0

    pos = strategy.get_position(db, "AAPL")
    assert pos is not None
    assert pos.quantity == 10
    assert pos.avg_entry_price == 50.0


@pytest.mark.unit
def test_buy_updates_avg_entry_price(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    strategy.buy(db, ticker="AAPL", quantity=10, price=60)

    pos = strategy.get_position(db, "AAPL")
    assert pos.quantity == 20
    expected_avg = (50 * 10 + 60 * 10) / 20
    assert pos.avg_entry_price == expected_avg


@pytest.mark.unit
def test_buy_raises_when_insufficient_cash(strategy, db, test_portfolio):
    with pytest.raises(ValueError, match="Cash disponible insuffisant"):
        strategy.buy(db, ticker="AAPL", quantity=10000, price=50)


@pytest.mark.unit
def test_buy_respects_reserved_cash(strategy, db, test_portfolio):
    test_portfolio.reserved_cash = 9000.0
    db.commit()

    # cash_available = 1000, trade cost = 500 + 1 = 501 → should succeed
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)

    # cash_available = 1000, trade cost = 1000 + 1 = 1001 → should fail
    with pytest.raises(ValueError, match="Cash disponible insuffisant"):
        strategy.buy(db, ticker="AAPL", quantity=20, price=50)


@pytest.mark.unit
def test_sell_adds_cash_and_calculates_pnl(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    trade = strategy.sell(db, ticker="AAPL", quantity=5, price=60)

    port = strategy.get_portfolio(db)
    expected_cash = 10000 - (10 * 50 + 1.0) + (5 * 60 - 1.0)
    assert port.cash_current == pytest.approx(expected_cash)

    expected_pnl = (60 - 50) * 5 - 1.0
    assert trade.realized_pnl == pytest.approx(expected_pnl)

    pos = strategy.get_position(db, "AAPL")
    assert pos.quantity == 5


@pytest.mark.unit
def test_sell_deletes_position_when_quantity_zero(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    strategy.sell(db, ticker="AAPL", quantity=10, price=60)

    pos = strategy.get_position(db, "AAPL")
    assert pos is None


@pytest.mark.unit
def test_sell_raises_when_position_insufficient(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=5, price=50)

    with pytest.raises(ValueError, match="Position insuffisante"):
        strategy.sell(db, ticker="AAPL", quantity=10, price=50)

    with pytest.raises(ValueError, match="Position insuffisante"):
        strategy.sell(db, ticker="UNKNOWN", quantity=1, price=50)


@pytest.mark.unit
def test_update_position_prices(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    strategy.buy(db, ticker="TSLA", quantity=5, price=200)

    strategy.update_position_prices(db, prices={"AAPL": 55.0, "TSLA": 190.0})

    aapl = strategy.get_position(db, "AAPL")
    assert aapl.current_price == 55.0
    assert aapl.current_value == 10 * 55.0
    assert aapl.unrealized_pnl == pytest.approx(10 * (55.0 - 50.0))
    assert aapl.unrealized_pnl_pct == pytest.approx((55.0 - 50.0) / 50.0 * 100)

    tsla = strategy.get_position(db, "TSLA")
    assert tsla.current_price == 190.0
    assert tsla.current_value == 5 * 190.0
    assert tsla.unrealized_pnl == pytest.approx(5 * (190.0 - 200.0))


@pytest.mark.unit
def test_snapshot_history_creates_row(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    strategy.update_position_prices(db, prices={"AAPL": 55.0})

    snap = strategy.snapshot_history(db)

    assert isinstance(snap, PortfolioHistory)
    assert snap.portfolio_id == test_portfolio.id
    expected_cash = 10000 - (10 * 50 + 1.0)
    assert snap.cash == pytest.approx(expected_cash)
    assert snap.positions_value == pytest.approx(10 * 55.0)
    expected_total = expected_cash + 10 * 55.0
    assert snap.total_value == pytest.approx(expected_total)
    assert snap.total_pnl == pytest.approx(expected_total - 10000)

    # Verify it was persisted
    rows = db.query(PortfolioHistory).filter(PortfolioHistory.portfolio_id == test_portfolio.id).all()
    assert len(rows) == 1


@pytest.mark.unit
def test_liquidate_sells_all_positions(strategy, db, test_portfolio):
    strategy.buy(db, ticker="AAPL", quantity=10, price=50)
    strategy.buy(db, ticker="TSLA", quantity=5, price=200)

    trades = strategy.liquidate(db, prices={"AAPL": 55.0, "TSLA": 190.0})

    assert len(trades) == 2
    tickers = {t.ticker for t in trades}
    assert tickers == {"AAPL", "TSLA"}

    assert strategy.get_position(db, "AAPL") is None
    assert strategy.get_position(db, "TSLA") is None

    port = strategy.get_portfolio(db)
    assert port.status == "liquidated"
