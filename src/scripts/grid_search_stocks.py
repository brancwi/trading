"""Grid search SL / TP / sizing / confidence sur le modèle stocks walk-forward."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.ml.dataset_builder import build_dataset
from trading.ml.evaluator import backtest_strategy

# Features classiques uniquement
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


def grid_search():
    with db_session() as db:
        df = build_dataset(db, horizon=10, threshold=0.05)
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        df_train = df[df["date"] < "2025-01-01"].copy()
        df_test = df[df["date"] >= "2025-01-01"].copy()
        
        X_train = df_train[CLASSIC_COLS].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[CLASSIC_COLS].values.astype(np.float32)
        y_test = df_test["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=4,
            reg_alpha=0.1, reg_lambda=1.0,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        print("\n=== GRID SEARCH STOCKS — Walk-Forward 2025 ===")
        print(f"Test set: {len(df_test)} rows")
        
        results = []
        for size in [0.1, 0.2, 0.3]:
            for sl in [None, 0.05, 0.10]:
                for tp in [None, 0.10, 0.20]:
                    for conf in [0.0, 0.5]:
                        bt = backtest_strategy(
                            df_test, y_pred, y_proba,
                            initial_capital=500, fee_per_order=1.0,
                            slippage_pct=0.001, confidence_threshold=conf,
                            position_pct=size,
                            stop_loss_pct=sl,
                            take_profit_pct=tp,
                            multi_ticker=True,
                        )
                        results.append({
                            "size": size, "sl": sl, "tp": tp, "conf": conf,
                            "return": bt["total_return_pct"],
                            "dd": bt["max_drawdown_pct"],
                            "trades": bt["trades_executed"],
                            "sharpe": bt["sharpe_ratio"],
                            "fees": bt["total_fees"],
                        })
        
        df_res = pd.DataFrame(results)
        
        # Best by return (DD < 30%)
        viable = df_res[df_res["dd"] > -30]
        if len(viable) > 0:
            best = viable.loc[viable["return"].idxmax()]
            print(f"\n🏆 BEST BY RETURN (DD < 30%):")
            print(f"  Size={best['size']*100:.0f}% SL={best['sl']} TP={best['tp']} Conf={best['conf']}")
            print(f"  Return: {best['return']:+.1f}% | DD: {best['dd']:.1f}% | Trades: {best['trades']} | Sharpe: {best['sharpe']:.2f}")
        
        # Best by Sharpe (DD < 30%)
        if len(viable) > 0:
            best_s = viable.loc[viable["sharpe"].idxmax()]
            print(f"\n🏆 BEST BY SHARPE (DD < 30%):")
            print(f"  Size={best_s['size']*100:.0f}% SL={best_s['sl']} TP={best_s['tp']} Conf={best_s['conf']}")
            print(f"  Return: {best_s['return']:+.1f}% | DD: {best_s['dd']:.1f}% | Trades: {best_s['trades']} | Sharpe: {best_s['sharpe']:.2f}")
        
        # Best by Return/DD ratio
        df_res["return_dd_ratio"] = df_res["return"] / df_res["dd"].abs()
        best_rdd = df_res.loc[df_res["return_dd_ratio"].idxmax()]
        print(f"\n🏆 BEST RETURN/DD RATIO:")
        print(f"  Size={best_rdd['size']*100:.0f}% SL={best_rdd['sl']} TP={best_rdd['tp']} Conf={best_rdd['conf']}")
        print(f"  Return: {best_rdd['return']:+.1f}% | DD: {best_rdd['dd']:.1f}% | Trades: {best_rdd['trades']} | Ratio: {best_rdd['return_dd_ratio']:.2f}")
        
        # Top 10 by return
        print(f"\n📊 TOP 10 BY RETURN:")
        print(f"{'Size':>5} {'SL':>5} {'TP':>5} {'Conf':>5} {'Return':>9} {'DD':>7} {'Trades':>7} {'Sharpe':>7}")
        print("-" * 65)
        for _, r in df_res.nlargest(10, "return").iterrows():
            print(f"{r['size']*100:>4.0f}% {str(r['sl']):>5s} {str(r['tp']):>5s} {r['conf']:>5.1f} {r['return']:>+8.1f}% {r['dd']:>6.1f}% {r['trades']:>6d} {r['sharpe']:>6.2f}")


if __name__ == "__main__":
    grid_search()
