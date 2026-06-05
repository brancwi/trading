"""Analyse détaillée du backtest crypto — trade par trade."""

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

logging.basicConfig(level=logging.WARNING)

CRYPTO_TICKERS = ['ETC-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD', 'UNI-USD']


def detailed_backtest(horizon=5, threshold=0.05):
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
        
        # Backtest détaillé avec suivi du capital
        capital = 500.0
        position = 0.0
        entry_price = 0.0
        portfolio_values = [capital]
        trades_log = []
        
        for i in range(len(df_test)):
            price = df_test.iloc[i]["close"]
            pred = y_pred[i]
            ticker = df_test.iloc[i]["ticker"]
            date = df_test.iloc[i]["timestamp"]
            
            buy_price = price * 1.001
            sell_price = price * 0.999
            
            if pred == 1 and capital > 1.0:  # BUY
                invest = capital - 1.0
                shares = invest / buy_price
                position += shares
                entry_price = buy_price
                capital = 0.0
                trades_log.append({"date": date, "ticker": ticker, "action": "BUY", "price": buy_price, "capital": 0})
            elif pred == 2 and position > 0:  # SELL
                proceeds = position * sell_price - 1.0
                if proceeds > 0:
                    pnl = proceeds - (position * entry_price)
                    capital = proceeds
                    trades_log.append({"date": date, "ticker": ticker, "action": "SELL", "price": sell_price, "capital": capital, "pnl": pnl})
                    position = 0.0
                    entry_price = 0.0
            
            current_value = capital + position * price
            portfolio_values.append(current_value)
        
        # Analyse des trades
        df_trades = pd.DataFrame(trades_log)
        sells = df_trades[df_trades["action"] == "SELL"]
        
        print(f"\n=== ANALYSE DÉTAILLÉE — H={horizon} th={threshold} ===")
        print(f"Trades: {len(df_trades)} ({len(sells)} sells)")
        if len(sells) > 0:
            print(f"PnL moyen par sell: {sells['pnl'].mean():.2f}€")
            print(f"PnL médian: {sells['pnl'].median():.2f}€")
            print(f"Meilleur trade: {sells['pnl'].max():.2f}€")
            print(f"Pire trade: {sells['pnl'].min():.2f}€")
            print(f"Trades gagnants: {(sells['pnl'] > 0).sum()} / {len(sells)} ({(sells['pnl'] > 0).mean()*100:.1f}%)")
        
        # Drawdown analysis
        pv = pd.Series(portfolio_values)
        peak = pv.cummax()
        dd = (pv - peak) / peak
        max_dd_idx = dd.idxmin()
        
        print(f"\nCapital final: {pv.iloc[-1]:.0f}€")
        print(f"Max drawdown: {dd.min()*100:.1f}%")
        print(f"Date max drawdown: {df_test.iloc[min(max_dd_idx, len(df_test)-1)]['timestamp']}")
        print(f"Capital au max drawdown: {pv.iloc[max_dd_idx]:.0f}€")
        print(f"Peak avant drawdown: {peak.iloc[max_dd_idx]:.0f}€")
        
        # Top 10 meilleurs/pires trades
        if len(sells) > 0:
            print(f"\nTop 5 meilleurs trades:")
            for _, t in sells.nlargest(5, "pnl").iterrows():
                print(f"  {t['date'].date()} {t['ticker']} SELL @ {t['price']:.2f}  PnL={t['pnl']:+.0f}€")
            print(f"\nTop 5 pires trades:")
            for _, t in sells.nsmallest(5, "pnl").iterrows():
                print(f"  {t['date'].date()} {t['ticker']} SELL @ {t['price']:.2f}  PnL={t['pnl']:+.0f}€")


if __name__ == "__main__":
    detailed_backtest(horizon=5, threshold=0.05)
