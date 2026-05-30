"""Backtest et métriques financières pour évaluer les stratégies.

Usage:
    from trading.ml.backtest import BacktestEngine
    bt = BacktestEngine()
    metrics = bt.run(signals, initial_cash=10000)
    print(metrics.sharpe_ratio, metrics.max_drawdown)
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Métriques financières d'une stratégie."""
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    profit_factor: float
    expectancy: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float


def _compute_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Sharpe annualisé (simplifié, pas annualisé ici car pas de fréquence fixe)."""
    excess = returns - risk_free_rate
    std = np.std(excess)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std)


def _compute_max_drawdown(equity_curve: np.ndarray) -> float:
    """Max drawdown en pourcentage."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.min(drawdown) * 100)


def _compute_profit_factor(pnls: np.ndarray) -> float:
    gains = np.sum(pnls[pnls > 0])
    losses = abs(np.sum(pnls[pnls < 0]))
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return gains / losses


def _compute_expectancy(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / len(pnls)
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
    return win_rate * avg_win - (1 - win_rate) * avg_loss


class BacktestEngine:
    """Simule une stratégie de trading et calcule les métriques."""

    def run(self, trades: list[dict], initial_cash: float = 10000.0) -> BacktestMetrics:
        """
        trades: liste de dicts avec keys:
            - 'pnl' (float): PnL réalisé sur le trade
            - 'timestamp' (str ou datetime): date du trade (optionnel)
        """
        if not trades:
            return BacktestMetrics(
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
            )

        pnls = np.array([t.get("pnl", 0.0) for t in trades])
        equity = initial_cash + np.cumsum(pnls)
        returns = pnls / initial_cash  # simplifié

        total_return = ((equity[-1] - initial_cash) / initial_cash) * 100
        sharpe = _compute_sharpe(returns)
        mdd = _compute_max_drawdown(equity)
        pf = _compute_profit_factor(pnls)
        exp = _compute_expectancy(pnls)
        win_rate = len(pnls[pnls > 0]) / len(pnls) * 100 if len(pnls) > 0 else 0.0

        return BacktestMetrics(
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 3),
            max_drawdown_pct=round(mdd, 2),
            profit_factor=round(pf, 2),
            expectancy=round(exp, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(trades),
            avg_trade_pnl=round(float(np.mean(pnls)), 2),
        )
