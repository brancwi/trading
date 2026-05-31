"""Tests unitaires pour BacktestEngine et métriques."""

import math

import numpy as np
import pytest

from trading.ml.backtest import (
    BacktestEngine,
    BacktestMetrics,
    _compute_expectancy,
    _compute_max_drawdown,
)


pytestmark = pytest.mark.unit


class TestBacktestEngineRun:
    def test_run_empty_trades_returns_all_zeros(self):
        engine = BacktestEngine()
        metrics = engine.run([])
        assert metrics == BacktestMetrics(
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            win_rate=0.0,
            total_trades=0,
            avg_trade_pnl=0.0,
        )

    def test_run_winning_trades_only(self):
        engine = BacktestEngine()
        trades = [{"pnl": 100}, {"pnl": 200}, {"pnl": 50}]
        metrics = engine.run(trades)
        assert metrics.total_return_pct > 0
        assert metrics.sharpe_ratio > 0
        assert metrics.win_rate == 100.0
        assert metrics.profit_factor == float("inf")

    def test_run_losing_trades_only(self):
        engine = BacktestEngine()
        trades = [{"pnl": -100}, {"pnl": -50}, {"pnl": -25}]
        metrics = engine.run(trades)
        assert metrics.total_return_pct < 0
        assert metrics.sharpe_ratio < 0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0

    def test_run_mixed_trades(self):
        engine = BacktestEngine()
        trades = [{"pnl": 100}, {"pnl": -100}, {"pnl": 50}, {"pnl": -50}]
        metrics = engine.run(trades)
        assert metrics.win_rate == 50.0
        assert math.isclose(metrics.profit_factor, 1.0, rel_tol=0.1)


class TestComputeFunctions:
    def test_compute_max_drawdown_20_percent(self):
        curve = np.array([100, 80])
        result = _compute_max_drawdown(curve)
        assert result == -20.0

    def test_compute_expectancy_equal_wins_losses(self):
        pnls = np.array([100, -100])
        result = _compute_expectancy(pnls)
        assert result == 0.0
