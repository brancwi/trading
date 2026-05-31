"""Entraînement et évaluation différenciée par portefeuille.

Chaque portefeuille a son propre capital initial, ses frais et sa devise.
Le modèle est entraîné une fois globalement, mais le backtest est simulé
avec les paramètres réels de chaque portefeuille pour voir l'impact du
capital et des coûts de trading.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from trading.core.database import db_session
from trading.core.models import Portfolio
from trading.ml.dataset_builder import build_dataset
from trading.ml.trainer import train_xgboost_walkforward, save_model, FEATURE_COLS
from trading.ml.evaluator import evaluate_classifier, backtest_strategy, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

# Hyperparamètres spécifiques par type de portefeuille
# Ninja (petit capital) : moins de trades, plus conservateur
# Simulation/Rotation (gros capital) : plus agressif
PORTFOLIO_HYPERPARAMS = {
    "ninja": {
        "n_estimators": 150,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "slippage_pct": 0.002,  # plus de slippage sur petits ordres
    },
    "rotation": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "slippage_pct": 0.001,
    },
    "simulation": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "slippage_pct": 0.001,
    },
}


def run_for_portfolio(
    db,
    portfolio: Portfolio,
    horizon: int = 5,
    threshold: float = 0.03,
    save: bool = True,
) -> dict:
    """Entraîne/évalue pour un portefeuille spécifique avec walk-forward."""
    strategy = portfolio.strategy_type
    hyper = PORTFOLIO_HYPERPARAMS.get(strategy, PORTFOLIO_HYPERPARAMS["simulation"])

    logger.info(
        "🚀 Portfolio '%s' — capital=%.0f %s | fee=%.2f | strategy=%s | hyper=%s",
        portfolio.id, portfolio.cash_initial, portfolio.base_currency,
        portfolio.fee_per_order or 1.0, strategy, hyper,
    )

    # 1) Dataset
    df = build_dataset(db, horizon=horizon, threshold=threshold)
    if df.empty or len(df) < 100:
        raise ValueError(f"Dataset too small: {len(df)} rows")

    # 2) Train walk-forward
    result = train_xgboost_walkforward(
        df,
        split_date="2025-01-01",
        n_estimators=hyper["n_estimators"],
        max_depth=hyper["max_depth"],
        learning_rate=hyper["learning_rate"],
        subsample=hyper["subsample"],
        colsample_bytree=hyper["colsample_bytree"],
    )

    # 3) Evaluate classification
    metrics = evaluate_classifier(result["y_test"], result["y_pred"])
    result["metrics"] = metrics

    # 4) Feature importance
    if hasattr(result["model"], "feature_importances_"):
        imp = list(zip(FEATURE_COLS, result["model"].feature_importances_))
        imp.sort(key=lambda x: x[1], reverse=True)
        result["feature_importance"] = imp

    # 5) Backtest avec les paramètres RÉELS du portfolio + slippage spécifique
    # On récupère le vrai subset de test (même split temporel que l'entraînement)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])
    df_test = df[df["date"] >= "2025-01-01"].copy()
    backtest = backtest_strategy(
        df_test,
        result["y_pred"],
        initial_capital=portfolio.cash_initial,
        fee_per_order=portfolio.fee_per_order or 1.0,
        base_currency=portfolio.base_currency,
        slippage_pct=hyper["slippage_pct"],
    )
    result["backtest"] = backtest
    result["portfolio_id"] = portfolio.id
    result["hyperparams"] = hyper

    # 6) Save
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"signal_{portfolio.id}_H{horizon}_th{threshold:.2f}_walkforward.pkl"
        save_model(result, model_path)

    return result


def run_all_portfolios(
    horizon: int = 5,
    threshold: float = 0.03,
    save: bool = True,
) -> list[dict]:
    """Boucle sur tous les portfolios actifs et génère un rapport comparatif."""
    results: list[dict] = []

    with db_session() as db:
        portfolios = db.query(Portfolio).filter(
            Portfolio.status == "active",
            Portfolio.id.like("staging-%"),
        ).all()

        if not portfolios:
            logger.warning("No active staging portfolios found")
            return results

        for port in portfolios:
            try:
                result = run_for_portfolio(db, port, horizon=horizon, threshold=threshold, save=save)
                results.append(result)
            except Exception as e:
                logger.exception("Failed for portfolio %s: %s", port.id, e)

    # Rapport comparatif
    print("\n" + "=" * 90)
    print("  COMPARAISON MULTI-PORTEFEUILLES — WALK-FORWARD 2025")
    print("=" * 90)
    print(f"\n{'Portfolio':<22} {'Capital':>10} {'Fee':>6} {'Slippage':>9} {'Return':>10} {'Sharpe':>8} {'Trades':>8} {'Fee Impact':>10}")
    print("-" * 90)
    for r in results:
        b = r["backtest"]
        h = r["hyperparams"]
        print(
            f"{r['portfolio_id']:<22} "
            f"{b['initial_capital']:>8.0f} {b['base_currency']:>2} "
            f"{b.get('total_fees', 0):>5.0f} "
            f"{h['slippage_pct']*100:>7.2f}% "
            f"{b['total_return_pct']:>+8.1f}% "
            f"{b['sharpe_ratio']:>6.2f} "
            f"{b['trades_executed']:>6d} "
            f"{b['fee_impact_pct']:>8.2f}%"
        )
    print("=" * 90)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Portfolio-aware training pipeline")
    parser.add_argument("--portfolio-id", type=str, default=None, help="Specific portfolio (default: all active)")
    parser.add_argument("--horizon", type=int, default=5, help="Future return horizon (days)")
    parser.add_argument("--threshold", type=float, default=0.03, help="Label threshold")
    parser.add_argument("--no-save", action="store_true", help="Skip model saving")
    args = parser.parse_args()

    try:
        if args.portfolio_id:
            with db_session() as db:
                port = db.query(Portfolio).filter(Portfolio.id == args.portfolio_id).first()
                if not port:
                    logger.error("Portfolio %s not found", args.portfolio_id)
                    return 1
                result = run_for_portfolio(db, port, horizon=args.horizon, threshold=args.threshold, save=not args.no_save)
                print_report(result, pd.DataFrame())
        else:
            run_all_portfolios(horizon=args.horizon, threshold=args.threshold, save=not args.no_save)
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
