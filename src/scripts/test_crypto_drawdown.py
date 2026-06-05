"""Calcule drawdown détaillé pour crypto sans UNI."""

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


def backtest_with_tracking(df_test, y_pred, y_proba, position_pct=1.0, confidence_threshold=0.0):
    """Backtest avec suivi complet du capital pour drawdown."""
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    portfolio_values = [capital]
    trades = []
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred = y_pred[i]
        ticker = df_test.iloc[i]["ticker"]
        ts = df_test.iloc[i]["timestamp"]
        
        if y_proba is not None and confidence_threshold > 0:
            if float(y_proba[i].max()) < confidence_threshold:
                pred = 0
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        if pred == 1 and capital > 1.0:
            invest = (capital - 1.0) * position_pct
            if invest < 1.0:
                pass
            else:
                shares = invest / buy_price
                position += shares
                capital -= invest + 1.0
                entry_price = buy_price
                trades.append({"ts": ts, "ticker": ticker, "action": "BUY", "price": buy_price, "invest": invest})
        
        elif pred == 2 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                pnl = proceeds - (position * entry_price)
                capital += proceeds
                trades.append({"ts": ts, "ticker": ticker, "action": "SELL", "price": sell_price, "pnl": pnl, "capital": capital})
                position = 0.0
                entry_price = 0.0
        
        current = capital + position * price
        portfolio_values.append(current)
    
    pv = pd.Series(portfolio_values)
    peak = pv.cummax()
    dd = (pv - peak) / peak
    
    return {
        "final": pv.iloc[-1],
        "max_dd": dd.min(),
        "max_dd_idx": dd.idxmin(),
        "peak": peak.iloc[-1],
        "trades": trades,
        "pv": pv,
    }


def test(horizon=5, threshold=0.05):
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
        
        print(f"\n=== DRAWDOWN ANALYSIS — H={horizon} th={threshold} ===")
        
        for pct in [1.0, 0.3, 0.2, 0.1]:
            r = backtest_with_tracking(df_test, y_pred, y_proba, position_pct=pct)
            ret = (r["final"] / 500.0 - 1) * 100
            dd = r["max_dd"] * 100
            n_sells = len([t for t in r["trades"] if t["action"] == "SELL"])
            n_buys = len([t for t in r["trades"] if t["action"] == "BUY"])
            
            # Profits/losses
            sells = [t for t in r["trades"] if t["action"] == "SELL"]
            if sells:
                pnls = [t["pnl"] for t in sells]
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
                avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
            else:
                win_rate = avg_win = avg_loss = 0
            
            label = "All-in" if pct == 1.0 else f"Sizing {pct*100:.0f}%"
            print(f"\n{label}:")
            print(f"  Final: {r['final']:>10,.0f}€  ({ret:>+8.1f}%)")
            print(f"  Max Drawdown: {dd:>8.1f}%")
            print(f"  Trades: {n_buys} buys / {n_sells} sells")
            print(f"  Win rate: {win_rate:.1f}% | Avg win: {avg_win:+.0f}€ | Avg loss: {avg_loss:+.0f}€")
            
            # Show worst drawdown period
            if r["max_dd_idx"] < len(df_test):
                dd_ts = df_test.iloc[min(r["max_dd_idx"], len(df_test)-1)]["timestamp"]
                print(f"  Worst DD at: {dd_ts}")


if __name__ == "__main__":
    test(horizon=5, threshold=0.05)
    test(horizon=10, threshold=0.05)
