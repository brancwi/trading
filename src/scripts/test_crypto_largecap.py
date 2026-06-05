"""Test crypto avec grandes capitalisations uniquement (BTC, ETH, SOL)."""

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

LARGE_CAPS = ['BTC-USD', 'ETH-USD', 'SOL-USD']


def backtest(df_test, y_pred, y_proba, position_pct=0.2, sl_pct=0.05, tp_pct=0.10):
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    portfolio_values = [capital]
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred = y_pred[i]
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        # SL/TP
        if position > 0 and entry_price > 0:
            pnl_pct = (sell_price / entry_price - 1)
            if pnl_pct <= -sl_pct:
                capital += max(0, position * sell_price - 1.0)
                position = 0.0; entry_price = 0.0
                portfolio_values.append(capital); continue
            if pnl_pct >= tp_pct:
                capital += max(0, position * sell_price - 1.0)
                position = 0.0; entry_price = 0.0
                portfolio_values.append(capital); continue
        
        if pred == 1 and capital > 1.0:
            invest = (capital - 1.0) * position_pct
            if invest >= 1.0:
                position += invest / buy_price
                capital -= invest + 1.0
                entry_price = buy_price
        elif pred == 2 and position > 0:
            capital += max(0, position * sell_price - 1.0)
            position = 0.0; entry_price = 0.0
        
        portfolio_values.append(capital + position * price)
    
    pv = pd.Series(portfolio_values)
    peak = pv.cummax()
    dd = (pv - peak) / peak
    return {"final": pv.iloc[-1], "max_dd": dd.min(), "return": (pv.iloc[-1]/500-1)*100}


def test():
    with db_session() as db:
        df = build_dataset(db, horizon=10, threshold=0.05)
        df_crypto = df[df["ticker"].isin(LARGE_CAPS)].copy()
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
        if len(df_test) < 100:
            print(f"Pas assez de données test: {len(df_test)} rows")
            return
        
        tech_cols = [c for c in FEATURE_COLS if not c.startswith("sentiment")]
        
        X_train = df_train[tech_cols].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[tech_cols].values.astype(np.float32)
        
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
        
        print(f"\n=== LARGE CAPS: BTC, ETH, SOL — H=10 th=0.05 ===")
        print(f"Train: {len(df_train)} | Test: {len(df_test)} rows")
        
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(
            df_test["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values, y_pred
        )
        print(f"Accuracy: {acc*100:.1f}%")
        
        print(f"\n{'SL':>4} {'TP':>4} {'Size':>5} {'Final':>10} {'Return':>9} {'MaxDD':>7}")
        print("-" * 45)
        
        for sl in [0.03, 0.05, 0.08, 0.10]:
            for tp in [0.05, 0.10, 0.15, 0.20]:
                for size in [0.2, 0.3]:
                    r = backtest(df_test, y_pred, y_proba, size, sl, tp)
                    if r["max_dd"] > -0.40:
                        print(f"{sl*100:>3.0f}% {tp*100:>3.0f}% {size*100:>4.0f}% {r['final']:>10,.0f}€ {r['return']:>+8.1f}% {r['max_dd']*100:>6.1f}%")


if __name__ == "__main__":
    test()
