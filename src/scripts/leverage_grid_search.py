"""Grid search levier × take-profit × stop-loss pour ninja.

Usage:
    cd /home/brancwi/dev/projects/trading
    env TRADING_ENVIRONMENT=staging PYTHONPATH=src .venv/bin/python src/scripts/leverage_grid_search.py
"""

import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.core.models import Portfolio
from trading.ml.dataset_builder import build_dataset
from trading.ml.evaluator import backtest_strategy
from trading.ml.portfolio_training import run_for_portfolio

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Grille de paramètres
LEVERAGES = [1.0, 1.5, 2.0, 2.5, 3.0]
TAKE_PROFITS = [None, 0.05, 0.08, 0.10, 0.15, 0.20]
STOP_LOSSES = [None, 0.02, 0.03, 0.05, 0.08]

def run_grid_search():
    results = []
    
    with db_session() as db:
        port = db.query(Portfolio).filter(Portfolio.id == "staging-ninja").first()
        
        # Entraîne une fois
        print("Entraînement du modèle ninja (H=10, th=0.05)...")
        result = run_for_portfolio(db, port, horizon=10, threshold=0.05, save=False)
        
        y_pred = result["y_pred"]
        y_proba = result.get("y_proba")
        
        # Récupère le df_test
        df = build_dataset(db, horizon=10, threshold=0.05)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        df_test = df[df["date"] >= "2025-01-01"].copy()
        
        print(f"Dataset test: {len(df_test)} rows")
        print(f"Grid: {len(LEVERAGES)} × {len(TAKE_PROFITS)} × {len(STOP_LOSSES)} = {len(LEVERAGES)*len(TAKE_PROFITS)*len(STOP_LOSSES)} combinaisons\n")
        
        total = len(LEVERAGES) * len(TAKE_PROFITS) * len(STOP_LOSSES)
        count = 0
        
        for lev, tp, sl in itertools.product(LEVERAGES, TAKE_PROFITS, STOP_LOSSES):
            count += 1
            bt = backtest_strategy(
                df_test,
                y_pred,
                y_proba=y_proba,
                initial_capital=500.0,
                fee_per_order=1.0,
                base_currency="EUR",
                slippage_pct=0.001,
                confidence_threshold=0.0,
                leverage=lev,
                take_profit_pct=tp,
                stop_loss_pct=sl,
            )
            results.append({
                "leverage": lev,
                "take_profit": f"{tp*100:.0f}%" if tp else "none",
                "stop_loss": f"{sl*100:.0f}%" if sl else "none",
                "return_pct": bt["total_return_pct"],
                "sharpe": bt["sharpe_ratio"],
                "max_dd": bt["max_drawdown_pct"],
                "trades": bt["trades_executed"],
                "fee_impact": bt["fee_impact_pct"],
                "final_value": bt["final_value"],
            })
            
            if count % 10 == 0:
                print(f"  Progress: {count}/{total}...")
    
    # Affiche le top 10 par return
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("return_pct", ascending=False)
    
    print("\n" + "="*100)
    print("  TOP 15 COMBINAISONS — RENDEMENT")
    print("="*100)
    print(f"{'Lev':>4} {'TP':>6} {'SL':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Fee%':>8} {'Final':>10}")
    print("-"*100)
    for _, row in df_results.head(15).iterrows():
        print(f"{row['leverage']:>4.1f} {row['take_profit']:>6} {row['stop_loss']:>6} {row['return_pct']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>7.1f}% {row['trades']:>8d} {row['fee_impact']:>7.1f}% {row['final_value']:>9.0f}€")
    
    # Top 10 par Sharpe (avec drawdown < 30%)
    df_safe = df_results[df_results["max_dd"] > -30]
    df_safe = df_safe.sort_values("sharpe", ascending=False)
    
    print("\n" + "="*100)
    print("  TOP 10 — SHARPE (drawdown < 30%)")
    print("="*100)
    print(f"{'Lev':>4} {'TP':>6} {'SL':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Fee%':>8} {'Final':>10}")
    print("-"*100)
    for _, row in df_safe.head(10).iterrows():
        print(f"{row['leverage']:>4.1f} {row['take_profit']:>6} {row['stop_loss']:>6} {row['return_pct']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>7.1f}% {row['trades']:>8d} {row['fee_impact']:>7.1f}% {row['final_value']:>9.0f}€")
    
    # Top 10 par rendement avec drawdown < 30%
    df_safe_ret = df_results[df_results["max_dd"] > -30].sort_values("return_pct", ascending=False)
    
    print("\n" + "="*100)
    print("  TOP 10 — RETURN (drawdown < 30%)")
    print("="*100)
    print(f"{'Lev':>4} {'TP':>6} {'SL':>6} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>8} {'Fee%':>8} {'Final':>10}")
    print("-"*100)
    for _, row in df_safe_ret.head(10).iterrows():
        print(f"{row['leverage']:>4.1f} {row['take_profit']:>6} {row['stop_loss']:>6} {row['return_pct']:>+9.1f}% {row['sharpe']:>8.2f} {row['max_dd']:>7.1f}% {row['trades']:>8d} {row['fee_impact']:>7.1f}% {row['final_value']:>9.0f}€")
    
    print("\n" + "="*100)
    print(f"Total combinaisons testées: {len(df_results)}")
    print("="*100)
    
    return df_results


if __name__ == "__main__":
    run_grid_search()
