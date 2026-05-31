"""Unit tests for core Pydantic schemas and ORM model properties."""

import pytest
from pydantic import ValidationError

from trading.core.models import (
    Portfolio,
    SignalAction,
    PortfolioStatus,
    SignalCreate,
    DecisionCreate,
    PortfolioRead,
)


@pytest.mark.unit
def test_portfolio_cash_available_no_reserved():
    port = Portfolio(
        id="p1",
        name="Test",
        strategy_type="simulation",
        cash_initial=10000.0,
        cash_current=10000.0,
        reserved_cash=0.0,
    )
    assert port.cash_available == 10000.0


@pytest.mark.unit
def test_portfolio_cash_available_with_reserved():
    port = Portfolio(
        id="p2",
        name="Test",
        strategy_type="simulation",
        cash_initial=10000.0,
        cash_current=10000.0,
        reserved_cash=500.0,
    )
    assert port.cash_available == 9500.0


@pytest.mark.unit
def test_portfolio_cash_available_reserved_greater_than_current():
    port = Portfolio(
        id="p3",
        name="Test",
        strategy_type="simulation",
        cash_initial=10000.0,
        cash_current=100.0,
        reserved_cash=500.0,
    )
    assert port.cash_available == -400.0


@pytest.mark.unit
def test_signal_action_enum_values():
    assert SignalAction.BUY == "BUY"
    assert SignalAction.SELL == "SELL"
    assert SignalAction.HOLD == "HOLD"
    assert SignalAction.STRONG_BUY == "STRONG_BUY"
    assert SignalAction.STRONG_SELL == "STRONG_SELL"


@pytest.mark.unit
def test_portfolio_status_enum_values():
    assert PortfolioStatus.ACTIVE == "active"
    assert PortfolioStatus.PAUSED == "paused"
    assert PortfolioStatus.LIQUIDATING == "liquidating"
    assert PortfolioStatus.LIQUIDATED == "liquidated"


@pytest.mark.unit
def test_signal_create_sentiment_out_of_range():
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=-1.1,
            strength=0.5,
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=1.1,
            strength=0.5,
            confidence=0.5,
        )


@pytest.mark.unit
def test_signal_create_strength_out_of_range():
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=0.0,
            strength=-0.1,
            confidence=0.5,
        )
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=0.0,
            strength=1.1,
            confidence=0.5,
        )


@pytest.mark.unit
def test_signal_create_confidence_out_of_range():
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=0.0,
            strength=0.5,
            confidence=-0.1,
        )
    with pytest.raises(ValidationError):
        SignalCreate(
            ticker="AAPL",
            action=SignalAction.BUY,
            sentiment=0.0,
            strength=0.5,
            confidence=1.1,
        )


@pytest.mark.unit
def test_decision_create_confidence_out_of_range():
    with pytest.raises(ValidationError):
        DecisionCreate(
            action=SignalAction.BUY,
            ticker="AAPL",
            portfolio_id="p1",
            confidence=1.1,
        )


@pytest.mark.unit
def test_decision_create_invalid_action():
    with pytest.raises(ValidationError):
        DecisionCreate(
            action="INVALID",  # type: ignore[arg-type]
            ticker="AAPL",
            portfolio_id="p1",
            confidence=0.5,
        )


@pytest.mark.unit
def test_portfolio_read_from_orm():
    port = Portfolio(
        id="test-e2e",
        name="Test Portfolio",
        strategy_type="simulation",
        cash_initial=10000.0,
        cash_current=10000.0,
        max_trade_amount=500.0,
        fee_per_order=1.0,
        status="active",
        config_json='{"sentiment_threshold": 0.5}',
    )
    pr = PortfolioRead.model_validate(port)
    assert pr.id == port.id
    assert pr.name == port.name
    assert pr.strategy_type == port.strategy_type
    assert pr.status == PortfolioStatus.ACTIVE
    assert pr.cash_initial == port.cash_initial
    assert pr.cash_current == port.cash_current
    assert pr.max_trade_amount == port.max_trade_amount
    assert pr.fee_per_order == port.fee_per_order
    assert pr.config_json == port.config_json
