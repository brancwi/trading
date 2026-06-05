"""Test crypto avec features enrichies (crypto-natives + F&G + funding)."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.trainer import FEATURE_COLS

logging.basicConfig(level=logging.WARNING)

CRYPTO_TICKERS = ['ETC-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD']


def backtest(df_test, y_pred, y_proba, position_pct=0.2, confidence_threshold=0.0):
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    portfolio_values = [capital]
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred = y_pred[i]
        
        if y_proba is not None and confidence_threshold > 0:
            if float(y_proba[i].max()) < confidence_threshold:
                pred = 0
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        if pred == 1 and capital > 1.0:
            invest = (capital - 1.0) * position_pct
            if invest >= 1.0:
                position += invest / buy_price
                capital -= invest + 1.0
                entry_price = buy_price
        elif pred == 2 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                capital += proceeds
                position = 0.0
        
        portfolio_values.append(capital + position * price)
    
    pv = pd.Series(portfolio_values)
    peak = pv.cummax()
    dd = (pv - peak) / peak
    return {"final": pv.iloc[-1], "max_dd": dd.min(), "return": (pv.iloc[-1]/500-1)*100}


def test(horizon=5, threshold=0.05):
    with db_session() as db:
        df = build_dataset(db, horizon=horizon, threshold=threshold)
        df_crypto = df[df["ticker"].isin(CRYPTO_TICKERS)].copy()
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
        print(f"\n=== ENRICHED CRYPTO — H={horizon} th={threshold} ===")
        print(f"Train: {len(df_train)} | Test: {len(df_test)} | Features: {len(FEATURE_COLS)}")
        
        X_train = df_train[FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[FEATURE_COLS].values.astype(np.float32)
        y_test = df_test["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.10,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=4,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc*100:.1f}%")
        
        # Feature importance
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 features:")
        for _, r in imp.head(10).iterrows():
            print(f"  {r['feature']:25s} {r['importance']:.4f}")
        
        print(f"\n{'Size':>5} {'Conf':>5} {'Final':>10} {'Return':>9} {'MaxDD':>7}")
        print("-" * 42)
        for size in [0.1, 0.2, 0.3, 1.0]:
            for conf in [0.0, 0.5, 0.6]:
                r = backtest(df_test, y_pred, y_proba, position_pct=size, confidence_threshold=conf)
                print(f"{size*100:>4.0f}% {conf:>4.1f} {r['final']:>10,.0f}€ {r['return']:>+8.1f}% {r['max_dd']*100:>6.1f}%")


if __name__ == "__main__":
    test(horizon=5, threshold=0.05)
    test(horizon=10, threshold=0.05)
