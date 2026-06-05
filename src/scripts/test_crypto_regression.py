"""Test crypto avec régression du future return (pas classification)."""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.trainer import FEATURE_COLS

logging.basicConfig(level=logging.WARNING)

CRYPTO_TICKERS = ['ETC-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'ALGO-USD', 'BCH-USD']


def backtest_regression(df_test, y_pred_return, position_pct=0.2, threshold=0.05):
    """Trade uniquement quand le return prédit > threshold."""
    capital = 500.0
    position = 0.0
    entry_price = 0.0
    portfolio_values = [capital]
    n_trades = 0
    
    for i in range(len(df_test)):
        price = df_test.iloc[i]["close"]
        pred_ret = y_pred_return[i]
        
        buy_price = price * 1.001
        sell_price = price * 0.999
        
        # BUY si return prédit > threshold et pas de position
        if pred_ret > threshold and capital > 1.0 and position == 0:
            invest = (capital - 1.0) * position_pct
            if invest >= 1.0:
                position += invest / buy_price
                capital -= invest + 1.0
                entry_price = buy_price
                n_trades += 1
        
        # SELL si return prédit < -threshold ou si on a une position et le return devient négatif
        elif pred_ret < -threshold / 2 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                capital += proceeds
                position = 0.0
                n_trades += 1
        
        # Forced sell at end
        if i == len(df_test) - 1 and position > 0:
            proceeds = position * sell_price - 1.0
            if proceeds > 0:
                capital += proceeds
                position = 0.0
        
        portfolio_values.append(capital + position * price)
    
    pv = pd.Series(portfolio_values)
    peak = pv.cummax()
    dd = (pv - peak) / peak
    return {
        "final": pv.iloc[-1], 
        "max_dd": dd.min(), 
        "return": (pv.iloc[-1]/500-1)*100,
        "trades": n_trades,
    }


def test(horizon=5):
    with db_session() as db:
        df = build_dataset(db, horizon=horizon, threshold=0.05)
        df_crypto = df[df["ticker"].isin(CRYPTO_TICKERS)].copy()
        df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"]).dt.date
        df_crypto["date"] = pd.to_datetime(df_crypto["date"])
        
        df_train = df_crypto[df_crypto["date"] < "2025-01-01"].copy()
        df_test = df_crypto[df_crypto["date"] >= "2025-01-01"].copy()
        
        print(f"\n=== REGRESSION APPROACH — H={horizon} ===")
        print(f"Train: {len(df_train)} | Test: {len(df_test)} | Features: {len(FEATURE_COLS)}")
        
        X_train = df_train[FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["future_return"].values
        X_test = df_test[FEATURE_COLS].values.astype(np.float32)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.10,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=4,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(df_test["future_return"], y_pred)
        r2 = r2_score(df_test["future_return"], y_pred)
        print(f"MSE: {mse:.6f} | R²: {r2:.4f}")
        print(f"Pred return range: [{y_pred.min()*100:.2f}%, {y_pred.max()*100:.2f}%]")
        
        # Feature importance
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 features:")
        for _, r in imp.head(10).iterrows():
            print(f"  {r['feature']:25s} {r['importance']:.4f}")
        
        print(f"\n{'Size':>5} {'Thresh':>7} {'Final':>10} {'Return':>9} {'MaxDD':>7} {'Trades':>7}")
        print("-" * 55)
        for size in [0.1, 0.2, 0.3]:
            for thresh in [0.01, 0.02, 0.03, 0.05, 0.08]:
                r = backtest_regression(df_test, y_pred, position_pct=size, threshold=thresh)
                print(f"{size*100:>4.0f}% {thresh*100:>6.1f}% {r['final']:>10,.0f}€ {r['return']:>+8.1f}% {r['max_dd']*100:>6.1f}% {r['trades']:>7d}")


if __name__ == "__main__":
    test(horizon=5)
    test(horizon=10)
