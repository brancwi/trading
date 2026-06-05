"""Test crypto SANS UNI-USD (données corrompues)."""

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

# SANS UNI-USD
CRYPTO_TICKERS = ['ETC-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD']


def backtest_all_in(df_test, y_pred, y_proba, confidence_threshold=0.0):
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred = y_pred[i]
        
        if y_proba is not None and confidence_threshold > 0:
            if float(y_proba[i].max()) < confidence_threshold:
                pred = 0
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        if pred == 1 and capital > 1.0:
            invest = capital - 1.0
            shares = invest / buy_price
            position += shares
            capital = 0.0
            entry_price = buy_price
        elif pred == 2 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                capital = proceeds
                position = 0.0
    
    final_value = capital + position * df_test.iloc[-1]["close"] if position > 0 else capital
    return final_value


def test_crypto(horizon=5, threshold=0.05):
    with db_session() as db:
        df = build_dataset(db, horizon=horizon, threshold=threshold)
        df_crypto = df[df["ticker"].isin(CRYPTO_TICKERS)].copy()
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
        tech_cols = [c for c in FEATURE_COLS if not c.startswith("sentiment")]
        
        X_train = df_train[tech_cols].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[tech_cols].values.astype(np.float32)
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
        
        final = backtest_all_in(df_test, y_pred, y_proba)
        ret = (final / 500.0 - 1) * 100
        n_trades = sum(1 for p in y_pred if p != 0)
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n=== CRYPTO SANS UNI — H={horizon} th={threshold} ===")
        print(f"Train: {len(df_train)} rows | Test: {len(df_test)} rows | Tickers: {CRYPTO_TICKERS}")
        print(f"Accuracy: {acc*100:.1f}% | Trades: {n_trades}")
        print(f"Final: {final:,.0f}€  ({ret:+.1f}%)")
        
        # Aussi avec position sizing
        def backtest_sizing(df_test, y_pred, y_proba, position_pct=0.2):
            capital = 500.0
            position = 0.0
            entry_price = 0.0
            
            for i in range(len(df_test)):
                price = df_test.iloc[i]["close"]
                pred = y_pred[i]
                
                buy_price = price * 1.001
                sell_price = price * 0.999
                
                if pred == 1 and capital > 1.0:
                    invest = (capital - 1.0) * position_pct
                    if invest < 1.0:
                        continue
                    shares = invest / buy_price
                    position += shares
                    capital -= invest + 1.0
                    entry_price = buy_price
                elif pred == 2 and position > 0:
                    proceeds = position * sell_price - 1.0
                    if proceeds > 0:
                        capital += proceeds
                        position = 0.0
            
            return capital + position * df_test.iloc[-1]["close"] if position > 0 else capital
        
        for pct in [0.1, 0.2, 0.3]:
            final_s = backtest_sizing(df_test, y_pred, y_proba, position_pct=pct)
            ret_s = (final_s / 500.0 - 1) * 100
            print(f"  Sizing {pct*100:.0f}%: {final_s:>10,.0f}€  ({ret_s:>+8.1f}%)")


if __name__ == "__main__":
    test_crypto(horizon=5, threshold=0.05)
    test_crypto(horizon=10, threshold=0.05)
    test_crypto(horizon=5, threshold=0.03)
