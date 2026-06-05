"""Test régression future return sur stocks — prédire la magnitude."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.evaluator import backtest_strategy

CLASSIC_COLS = [
    "sma_10", "sma_20", "sma_50", "ema_10", "ema_20",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_width", "bb_pct", "atr_14",
    "momentum_10", "roc_10", "volatility_20",
    "price_sma20_ratio", "price_sma50_ratio",
    "price_change_1d", "price_change_3d", "price_change_7d",
    "volume_ratio", "volume_change_1d",
    "adx_14", "stoch_k", "stoch_d",
    "obv_ratio", "williams_r", "tr_normalized",
    "volatility_ratio",
]


def test():
    with db_session() as db:
        df = build_dataset(db, horizon=10, threshold=0.05)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        df_train = df[df["date"] < "2025-01-01"].copy()
        df_test = df[df["date"] >= "2025-01-01"].copy()
        
        X_train = df_train[CLASSIC_COLS].values.astype(np.float32)
        y_train = df_train["future_return"].values
        X_test = df_test[CLASSIC_COLS].values.astype(np.float32)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, n_jobs=4,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(df_test["future_return"], y_pred)
        r2 = r2_score(df_test["future_return"], y_pred)
        print(f"\n=== STOCKS REGRESSION ===")
        print(f"MSE: {mse:.2f} | R²: {r2:.4f}")
        print(f"Pred range: [{y_pred.min()*100:.1f}%, {y_pred.max()*100:.1f}%]")
        
        # Feature importance
        imp = pd.DataFrame({
            "feature": CLASSIC_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 features:")
        for _, r in imp.head(10).iterrows():
            print(f"  {r['feature']:25s} {r['importance']:.4f}")
        
        # Convert regression to classification for backtest
        # BUY if predicted return > threshold, SELL if < -threshold/2, else HOLD
        y_class = np.zeros(len(y_pred), dtype=int)
        for thresh in [0.01, 0.02, 0.03, 0.05]:
            y_class = np.where(y_pred > thresh, 1, np.where(y_pred < -thresh/2, 2, 0))
            print(f"\n--- Threshold {thresh*100:.1f}%: BUY={(y_class==1).sum()} SELL={(y_class==2).sum()} HOLD={(y_class==0).sum()} ---")
            bt = backtest_strategy(df_test, y_class, None, initial_capital=500, fee_per_order=1.0,
                                   slippage_pct=0.001, position_pct=0.2, multi_ticker=True)
            print(f"Return: {bt['total_return_pct']:+.1f}% | DD: {bt['max_drawdown_pct']:.1f}% | Trades: {bt['trades_executed']} | Sharpe: {bt['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    test()
