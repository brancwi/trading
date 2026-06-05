"""Test crypto avec stop-loss et take-profit."""

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


def backtest_risk_mgmt(df_test, y_pred, y_proba, position_pct=0.2, 
                       sl_pct=0.05, tp_pct=0.10, confidence_threshold=0.0):
    """Backtest avec SL/TP et position sizing."""
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    invested = 0.0
    portfolio_values = [capital]
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred = y_pred[i]
        
        if y_proba is not None and confidence_threshold > 0:
            if float(y_proba[i].max()) < confidence_threshold:
                pred = 0
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        # Check SL/TP on existing position
        if position > 0 and entry_price > 0:
            pnl_pct = (sell_price / entry_price - 1)
            if pnl_pct <= -sl_pct:  # Stop loss
                proceeds = position * sell_price - 1.0
                if proceeds > 0:
                    capital += proceeds
                position = 0.0
                entry_price = 0.0
                invested = 0.0
                portfolio_values.append(capital)
                continue
            if pnl_pct >= tp_pct:  # Take profit
                proceeds = position * sell_price - 1.0
                if proceeds > 0:
                    capital += proceeds
                position = 0.0
                entry_price = 0.0
                invested = 0.0
                portfolio_values.append(capital)
                continue
        
        if pred == 1 and capital > 1.0:
            invest = (capital - 1.0) * position_pct
            if invest < 1.0:
                pass
            else:
                shares = invest / buy_price
                position += shares
                capital -= invest + 1.0
                entry_price = buy_price
                invested += invest
        
        elif pred == 2 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                capital += proceeds
                position = 0.0
                entry_price = 0.0
                invested = 0.0
        
        current = capital + position * price
        portfolio_values.append(current)
    
    pv = pd.Series(portfolio_values)
    peak = pv.cummax()
    dd = (pv - peak) / peak
    
    return {
        "final": pv.iloc[-1],
        "max_dd": dd.min(),
        "return": (pv.iloc[-1] / 500.0 - 1) * 100,
    }


def test_grid():
    with db_session() as db:
        df = build_dataset(db, horizon=5, threshold=0.05)
        df_crypto = df[df["ticker"].isin(CRYPTO_TICKERS)].copy()
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
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
        
        print("\n=== GRID SEARCH: SL / TP / SIZING ===")
        print(f"{'SL':>5} {'TP':>5} {'Size':>5} {'Final':>10} {'Return':>10} {'MaxDD':>8}")
        print("-" * 50)
        
        best = None
        for sl in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
            for tp in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                for size in [0.1, 0.2, 0.3]:
                    r = backtest_risk_mgmt(df_test, y_pred, y_proba, 
                                           position_pct=size, sl_pct=sl, tp_pct=tp)
                    if r["max_dd"] > -0.50:  # Only show if DD < 50%
                        print(f"{sl*100:>5.0f}% {tp*100:>5.0f}% {size*100:>5.0f}% {r['final']:>10,.0f}€ {r['return']:>+9.1f}% {r['max_dd']*100:>7.1f}%")
                        if best is None or (r["return"] > best["return"] and r["max_dd"] > -0.30):
                            best = r
                            best["sl"] = sl
                            best["tp"] = tp
                            best["size"] = size
        
        if best:
            print(f"\n=== BEST (DD < 30%) ===")
            print(f"SL={best['sl']*100:.0f}% TP={best['tp']*100:.0f}% Size={best['size']*100:.0f}%")
            print(f"Return: {best['return']:+.1f}% | MaxDD: {best['max_dd']*100:.1f}%")
        else:
            print("\n=== NO CONFIG WITH DD < 50% FOUND ===")


if __name__ == "__main__":
    test_grid()
