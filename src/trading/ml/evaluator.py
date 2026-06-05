"""Évaluation des modèles — métriques classiques + backtest financier.

Calcule :
  - Accuracy, F1, confusion matrix
  - Backtest robuste : simule un portefeuille qui suit les prédictions BUY/SELL
  - Support multi-ticker (une position par ticker) ou single-position
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
    y_proba: np.ndarray | None = None,
    initial_capital: float = 10_000.0,
    fee_per_order: float = 1.0,
    base_currency: str = "USD",
    slippage_pct: float = 0.001,
    confidence_threshold: float = 0.0,
    leverage: float = 1.0,
    take_profit_pct: float | None = None,
    stop_loss_pct: float | None = None,
    position_pct: float = 1.0,  # 1.0 = all-in, 0.2 = 20% sizing
    multi_ticker: bool = True,  # True = une position par ticker
) -> dict[str, Any]:
    """Simule un portefeuille qui achète sur BUY, vend sur SELL, hold sinon.

    Args:
        df: DataFrame avec colonnes [ticker, close, future_return, label]
        y_pred: Prédictions du modèle (0=HOLD, 1=BUY, 2=SELL)
        y_proba: Probabilités du modèle (N, 3) — utilisées pour filtrer par confiance
        initial_capital: Capital initial du portfolio
        fee_per_order: Frais fixe par ordre (ex: 1.0 EUR/USD)
        base_currency: Devise du portfolio (pour l'affichage)
        confidence_threshold: Seuil de confiance minimum pour trader (0=tous, 0.6=forts)
        position_pct: Fraction du capital à investir par trade (1.0=all-in, 0.2=20%)
        multi_ticker: Si True, permet une position par ticker. Si False, une seule position globale
            qui est liquidée quand on change de ticker.

    Returns:
        Dict avec métriques financières
    """
    capital = initial_capital
    
    if multi_ticker:
        # Une position par ticker
        positions: dict[str, dict] = {}  # ticker -> {"shares": float, "entry_price": float}
    else:
        # Une seule position globale
        positions = {"_global": {"shares": 0.0, "entry_price": 0.0}}
    
    # Dernière valeur connue de chaque position (pour le tracking du portfolio)
    last_position_values: dict[str, float] = {}
    
    portfolio_values = [initial_capital]
    trades = 0
    total_fees = 0.0
    current_ticker = None

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        pred = y_pred[i]
        ticker = df.iloc[i].get("ticker", "_global")

        # Filtre par confiance : si y_proba fourni, on ignore les signaux faibles
        if y_proba is not None and confidence_threshold > 0:
            max_proba = float(y_proba[i].max())
            if max_proba < confidence_threshold:
                pred = 0  # FORCER HOLD si confiance insuffisante

        # Single-position mode : liquider si on change de ticker
        if not multi_ticker and current_ticker is not None and ticker != current_ticker:
            pos = positions["_global"]
            if pos["shares"] > 0:
                sell_price = price * (1 - slippage_pct)
                proceeds = pos["shares"] * sell_price - fee_per_order
                if proceeds > 0:
                    capital += proceeds
                    total_fees += fee_per_order
                    trades += 1
                pos["shares"] = 0.0
                pos["entry_price"] = 0.0
            current_ticker = ticker
        elif not multi_ticker:
            current_ticker = ticker

        # Récupérer la position
        if multi_ticker:
            if ticker not in positions:
                positions[ticker] = {"shares": 0.0, "entry_price": 0.0}
            pos = positions[ticker]
        else:
            pos = positions["_global"]

        # Helper : valeur d'une position avec levier
        def _position_value(shares: float, entry_price: float, current_price: float) -> float:
            if shares > 0 and entry_price > 0 and leverage != 1.0:
                return shares * entry_price + (current_price / entry_price - 1) * leverage * shares * entry_price
            return shares * current_price

        # Slippage : on achète plus cher, on vend moins cher
        buy_price = price * (1 + slippage_pct)
        sell_price = price * (1 - slippage_pct)

        # Vérification take-profit / stop-loss (sortie forcée)
        tp_sl_exit = False
        if pos["shares"] > 0 and pos["entry_price"] > 0:
            unrealized_pct = (sell_price / pos["entry_price"] - 1) * leverage
            if take_profit_pct is not None and unrealized_pct >= take_profit_pct:
                tp_sl_exit = True
            if stop_loss_pct is not None and unrealized_pct <= -stop_loss_pct:
                tp_sl_exit = True

        if tp_sl_exit and pos["shares"] > 0:
            proceeds = _position_value(pos["shares"], pos["entry_price"], sell_price) - fee_per_order
            if proceeds > 0:
                capital += proceeds
                pos["shares"] = 0.0
                pos["entry_price"] = 0.0
                total_fees += fee_per_order
                trades += 1
            continue  # skip signal pour ce jour

        if pred == 1 and capital > fee_per_order:  # BUY
            invest = (capital - fee_per_order) * position_pct
            if invest < fee_per_order:
                pass  # Pas assez pour investir
            else:
                shares = invest / buy_price
                # Mise à jour du prix d'entrée moyen (moyenne pondérée)
                if pos["shares"] > 0:
                    total_value = pos["shares"] * pos["entry_price"] + shares * buy_price
                    pos["shares"] += shares
                    pos["entry_price"] = total_value / pos["shares"]
                else:
                    pos["shares"] = shares
                    pos["entry_price"] = buy_price
                capital -= invest + fee_per_order
                total_fees += fee_per_order
                trades += 1
                
        elif pred == 2 and pos["shares"] > 0:  # SELL
            proceeds = _position_value(pos["shares"], pos["entry_price"], sell_price) - fee_per_order
            if proceeds > 0:
                capital += proceeds
                pos["shares"] = 0.0
                pos["entry_price"] = 0.0
                total_fees += fee_per_order
                trades += 1

        # Valeur du portfolio = cash + toutes les positions
        # Mettre à jour la valeur de la position courante
        if pos["shares"] > 0:
            last_position_values[ticker] = _position_value(pos["shares"], pos["entry_price"], price)
        elif ticker in last_position_values:
            del last_position_values[ticker]
        
        current_value = capital + sum(last_position_values.values())
        portfolio_values.append(current_value)

    # Liquidation finale de toutes les positions restantes
    for ticker in list(positions.keys()):
        pos = positions[ticker]
        if pos["shares"] > 0:
            # Dernier prix connu pour ce ticker
            ticker_rows = df[df["ticker"] == ticker] if "ticker" in df.columns else df
            if len(ticker_rows) > 0:
                last_price = ticker_rows.iloc[-1]["close"] * (1 - slippage_pct)
                proceeds = _position_value(pos["shares"], pos["entry_price"], last_price) - fee_per_order
                if proceeds > 0:
                    capital += proceeds
                    total_fees += fee_per_order
                    trades += 1
                pos["shares"] = 0.0
                pos["entry_price"] = 0.0

    # Recalculer la valeur finale avec liquidation
    final_value = capital
    for t, p in positions.items():
        if p["shares"] > 0:
            ticker_rows = df[df["ticker"] == t] if "ticker" in df.columns else df
            if len(ticker_rows) > 0:
                last_price = ticker_rows.iloc[-1]["close"]
                final_value += _position_value(p["shares"], p["entry_price"], last_price)

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
        "base_currency": base_currency,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "trades_executed": trades,
        "total_fees": round(total_fees, 2),
        "avg_trade_return_pct": round(total_return / max(trades, 1) * 100, 4),
        "fee_impact_pct": round(total_fees / initial_capital * 100, 2),
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
    print(f"   BUY    {cm[1,1]:>4d} {cm[1,2]:>4d} {cm[1,0]:>4d}")
    print(f"   SELL   {cm[2,2]:>4d} {cm[2,0]:>4d} {cm[2,1]:>4d}")

    # Backtest
    if "backtest" in result:
        b = result["backtest"]
        curr = b.get("base_currency", "USD")
        print(f"\n💰 Backtest Simulation ({b['initial_capital']:.0f} {curr}):")
        print(f"   Capital initial : {b['initial_capital']:,.2f} {curr}")
        print(f"   Capital final   : {b['final_value']:,.2f} {curr}")
        print(f"   Rendement total : {b['total_return_pct']:+.2f}%")
        print(f"   Sharpe ratio    : {b['sharpe_ratio']:.4f}")
        print(f"   Max drawdown    : {b['max_drawdown_pct']:.2f}%")
        print(f"   Trades exécutés : {b['trades_executed']}")
        print(f"   Frais totaux    : {b.get('total_fees', 0):.2f} {curr}")
        print(f"   Impact frais    : {b.get('fee_impact_pct', 0):.2f}%")

    # Feature importance (XGBoost)
    if "feature_importance" in result:
        print(f"\n🔝 Top 10 Features (XGBoost):")
        for name, score in result["feature_importance"][:10]:
            print(f"   {name:25s} {score:.4f}")

    print("=" * 60)


if __name__ == "__main__":
    # Exemple d'utilisation
    print("Run via train_pipeline.py")
