"""Pipeline complet d'entraînement — exécute ETL + train + eval d'un seul coup.

Usage:
    python -m trading.ml.train_pipeline [--horizon 5] [--threshold 0.03]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.trainer import train_xgboost, save_model, FEATURE_COLS
from trading.ml.evaluator import evaluate_classifier, backtest_strategy, print_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


def run_pipeline(horizon: int = 5, threshold: float = 0.03, save: bool = True) -> dict:
    """Orchestre le pipeline complet."""
    logger.info("🚀 Starting training pipeline  H=%d  th=%.2f", horizon, threshold)

    with db_session() as db:
        # 1) Dataset
        df = build_dataset(db, horizon=horizon, threshold=threshold)
        if df.empty or len(df) < 100:
            raise ValueError(f"Dataset too small: {len(df)} rows")

        # 2) Train
        result = train_xgboost(df)

        # 3) Evaluate
        metrics = evaluate_classifier(result["y_test"], result["y_pred"])
        result["metrics"] = metrics

        # Feature importance
        if hasattr(result["model"], "feature_importances_"):
            imp = list(zip(FEATURE_COLS, result["model"].feature_importances_))
            imp.sort(key=lambda x: x[1], reverse=True)
            result["feature_importance"] = imp

        # 4) Backtest
        # Récupère le sous-ensemble test avec prix pour le backtest
        test_indices = df.index[-len(result["y_test"]):]
        df_test = df.iloc[-len(result["y_test"]):].copy()
        backtest = backtest_strategy(df_test, result["y_pred"])
        result["backtest"] = backtest

        # 5) Save
        if save:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODELS_DIR / f"signal_xgboost_H{horizon}_th{threshold:.2f}.pkl"
            save_model(result, model_path)

        # 6) Report
        print_report(result, df_test)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Training pipeline for trading signals")
    parser.add_argument("--horizon", type=int, default=5, help="Future return horizon (days)")
    parser.add_argument("--threshold", type=float, default=0.03, help="Label threshold")
    parser.add_argument("--no-save", action="store_true", help="Skip model saving")
    args = parser.parse_args()

    try:
        run_pipeline(horizon=args.horizon, threshold=args.threshold, save=not args.no_save)
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
