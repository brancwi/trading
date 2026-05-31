"""Évaluation des modèles — métriques classiques + backtest financier.

Calcule :
  - Accuracy, F1, confusion matrix
  - Backtest simple : simule un portefeuille qui suit les prédictions BUY/SELL
  - Sharpe ratio, max drawdown, profit cumulé
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str] = None) -> dict:
    """Évalue un classifieur multi-classe."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    label_names = label_names or ["HOLD", "BUY", "SELL"]
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def backtest_strategy(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    initial_capital: float = 10_000.0,
    transaction_cost_pct: float = 0.001,
) -> dict[str, Any]:
    """Simule un portefeuille qui achète sur BUY, vend sur SELL, hold sinon.

    Args:
        df: DataFrame avec colonnes [close, future_return, label]
        y_pred: Prédictions du modèle (0=HOLD, 1=BUY, 2=SELL)
        initial_capital: Capital initial
        transaction_cost_pct: Coût de transaction (0.1%)

    Returns:
        Dict avec métriques financières
    """
    capital = initial_capital
    position = 0.0  # nombre d'actions détenues
    portfolio_values = [initial_capital]
    trades = 0

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        pred = y_pred[i]

        if pred == 1 and capital > 0:  # BUY
            shares = capital * (1 - transaction_cost_pct) / price
            position += shares
            capital = 0.0
            trades += 1
        elif pred == 2 and position > 0:  # SELL
            capital = position * price * (1 - transaction_cost_pct)
            position = 0.0
            trades += 1

        # Valeur du portfolio = cash + positions
        current_value = capital + position * price
        portfolio_values.append(current_value)

    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital) - 1

    # Sharpe ratio (simplifié : rendement journalier / écart-type)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

    # Max drawdown
    peak = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min()

    metrics = {
        "initial_capital": round(initial_capital, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trades_executed": trades,
        "avg_trade_return_pct": round(total_return / max(trades, 1) * 100, 4),
    }
    return metrics


def print_report(result: dict[str, Any], df: pd.DataFrame) -> None:
    """Affiche un rapport complet."""
    print("\n" + "=" * 60)
    print("  RAPPORT D'ÉVALUATION — MODÈLE DE SIGNAL")
    print("=" * 60)

    # Métriques classification
    m = result["metrics"]
    print(f"\n📊 Classification:")
    print(f"   Accuracy      : {m['accuracy']:.2%}")
    print(f"   F1 (macro)    : {m['f1_macro']:.4f}")
    print(f"   F1 (weighted) : {m['f1_weighted']:.4f}")
    print(f"   Precision     : {m['precision_macro']:.4f}")
    print(f"   Recall        : {m['recall_macro']:.4f}")

    # Matrice de confusion
    print(f"\n📋 Confusion Matrix:")
    cm = np.array(m["confusion_matrix"])
    print(f"                 PRED")
    print(f"   TRUE   HOLD  BUY  SELL")
    print(f"   HOLD   {cm[0,0]:>4d} {cm[0,1]:>4d} {cm[0,2]:>4d}")
    print(f"   BUY    {cm[1,0]:>4d} {cm[1,1]:>4d} {cm[1,2]:>4d}")
    print(f"   SELL   {cm[2,0]:>4d} {cm[2,1]:>4d} {cm[2,2]:>4d}")

    # Backtest
    if "backtest" in result:
        b = result["backtest"]
        print(f"\n💰 Backtest Simulation:")
        print(f"   Capital initial : ${b['initial_capital']:,.2f}")
        print(f"   Capital final   : ${b['final_value']:,.2f}")
        print(f"   Rendement total : {b['total_return_pct']:+.2f}%")
        print(f"   Sharpe ratio    : {b['sharpe_ratio']:.4f}")
        print(f"   Max drawdown    : {b['max_drawdown_pct']:.2f}%")
        print(f"   Trades exécutés : {b['trades_executed']}")

    # Feature importance (XGBoost)
    if "feature_importance" in result:
        print(f"\n🔝 Top 10 Features (XGBoost):")
        for name, score in result["feature_importance"][:10]:
            print(f"   {name:25s} {score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Run via train_pipeline.py")
