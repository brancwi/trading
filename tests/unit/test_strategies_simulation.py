"""Unit tests for SimulationStrategy."""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from trading.strategies.simulation import SimulationStrategy
from trading.core.models import Signal, Trade


@pytest.fixture
def sim_strategy(test_portfolio):
    return SimulationStrategy(portfolio_id=test_portfolio.id)


@pytest.mark.unit
def test_run_skips_when_portfolio_not_active(sim_strategy, db, test_portfolio):
    test_portfolio.status = "paused"
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 100.0})
    assert trades == []


@pytest.mark.unit
def test_run_processes_buy_signal_when_sentiment_above_threshold(sim_strategy, db, test_portfolio):
    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 100.0})

    assert len(trades) == 1
    assert trades[0].action == "BUY"
    assert trades[0].ticker == "AAPL"


@pytest.mark.unit
def test_run_respects_cash_available_not_cash_current(sim_strategy, db, test_portfolio):
    test_portfolio.reserved_cash = 9600.0
    db.commit()
    # cash_current = 10000, reserved = 9600 → cash_available = 400

    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 10.0})

    assert len(trades) == 1
    # max_trade = 500, cash_available - cash_min = 400 - 100 = 300
    # trade_amount = min(500, 300) = 300
    assert trades[0].amount == pytest.approx(300.0)
    assert trades[0].quantity == pytest.approx(30.0)


@pytest.mark.unit
def test_run_skips_when_cash_available_below_cash_min(sim_strategy, db, test_portfolio):
    test_portfolio.reserved_cash = 9950.0
    db.commit()
    # cash_available = 50, which is < cash_min (100)

    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 10.0})
    assert trades == []


@pytest.mark.unit
def test_run_skips_signal_when_ticker_not_in_prices(sim_strategy, db, test_portfolio):
    sig = Signal(
        ticker="MISSING",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 100.0})
    assert trades == []


@pytest.mark.unit
def test_run_skips_signal_when_price_not_positive(sim_strategy, db, test_portfolio):
    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades_zero = sim_strategy.run(db, prices={"AAPL": 0.0})
    trades_negative = sim_strategy.run(db, prices={"AAPL": -5.0})

    assert trades_zero == []
    assert trades_negative == []


@pytest.mark.unit
@patch("trading.strategies.simulation.SignalModel")
def test_run_ml_filter_holds_signal(mock_signal_model_cls, sim_strategy, db, test_portfolio):
    mock_model = MagicMock()
    mock_model.trained = True
    mock_model.predict.return_value = {"action": "HOLD", "confidence": 0.9}
    mock_signal_model_cls.return_value = mock_model

    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 100.0})

    assert trades == []
    mock_model.predict.assert_called_once_with(
        ticker="AAPL",
        sentiment_combined=0.8,
        sentiment_confidence=0.85,
    )


@pytest.mark.unit
def test_run_consumes_signals(sim_strategy, db, test_portfolio):
    sig = Signal(
        ticker="AAPL",
        action="BUY",
        sentiment=0.8,
        strength=0.9,
        confidence=0.85,
        consumed=0,
    )
    db.add(sig)
    db.commit()

    trades = sim_strategy.run(db, prices={"AAPL": 100.0})

    assert len(trades) == 1
    db.refresh(sig)
    assert sig.consumed == 1
