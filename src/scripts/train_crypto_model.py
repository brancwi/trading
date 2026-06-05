"""Entraîne un modèle XGBoost spécifique aux cryptos pour ninja."""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.evaluator import backtest_strategy
from trading.ml.trainer import FEATURE_COLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cryptos avec suffisamment d'historique (>= 1000 jours = 4 ans)
CRYPTO_TICKERS = ['ETC-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD', 'UNI-USD']


def train_crypto_model(horizon=5, threshold=0.03):
    with db_session() as db:
        # Build dataset complet
        df = build_dataset(db, horizon=horizon, threshold=threshold)
        
        # Filtre uniquement les cryptos
        df_crypto = df[df["ticker"].isin(CRYPTO_TICKERS)].copy()
        logger.info(f"Dataset crypto: {len(df_crypto)} rows (total: {len(df)})")
        
        if len(df_crypto) < 200:
            raise ValueError(f"Pas assez de données crypto: {len(df_crypto)}")
        
        # Split walk-forward (2025+)
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
        logger.info(f"Train: {len(df_train)} | Test: {len(df_test)}")
        
        # Features techniques uniquement
        tech_cols = [c for c in FEATURE_COLS if not c.startswith("sentiment")]
        
        X_train = df_train[tech_cols].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[tech_cols].values.astype(np.float32)
        y_test = df_test["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Hyperparams agressifs pour crypto (volatilité élevée)
        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Backtest
        bt = backtest_strategy(
            df_test,
            y_pred,
            y_proba=y_proba,
            initial_capital=500.0,
            fee_per_order=1.0,
            base_currency="EUR",
            slippage_pct=0.002,  # slippage plus élevé en crypto
            confidence_threshold=0.0,
        )
        
        # Accuracy
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*70)
        print(f"  RÉSULTATS CRYPTO — H={horizon} th={threshold}")
        print("="*70)
        print(f"  Accuracy      : {acc:.2%}")
        print(f"  Return        : {bt['total_return_pct']:+.1f}%")
        print(f"  Sharpe        : {bt['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown  : {bt['max_drawdown_pct']:.1f}%")
        print(f"  Trades        : {bt['trades_executed']}")
        print(f"  Fee Impact    : {bt['fee_impact_pct']:.1f}%")
        print(f"  Final Value   : {bt['final_value']:.0f}€")
        print("="*70)
        
        # Sauvegarde
        model_dir = Path(__file__).resolve().parents[2] / "models"
        model_dir.mkdir(exist_ok=True)
        path = model_dir / f"crypto_ninja_H{horizon}_th{threshold:.2f}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler, "feature_names": tech_cols}, f)
        logger.info(f"Modèle sauvegardé: {path}")
        
        return bt, acc


if __name__ == "__main__":
    for h, t in [(3, 0.05), (5, 0.03), (5, 0.05), (10, 0.05)]:
        try:
            train_crypto_model(horizon=h, threshold=t)
        except Exception as e:
            logger.error(f"Erreur H={h} th={t}: {e}")
