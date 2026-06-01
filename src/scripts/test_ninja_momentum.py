"""Test nouvelle stratégie ninja : momentum court terme sur actions volatiles."""

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

# Univers volatil : meme stocks + tech volatile + crypto
MOMENTUM_TICKERS = [
    # Meme / Volatil
    "GME", "AMC", "PLTR", "SOFI", "MSTR", "HOOD", "COIN", "RBLX",
    # Tech volatil
    "TSLA", "AMD", "NVDA", "META", "NFLX", "SHOP", "CRWD", "DDOG",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD",
]

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


def test_config(horizon, threshold, position_pct, sl, tp, conf_thresh):
    with db_session() as db:
        df = build_dataset(db, horizon=horizon, threshold=threshold)
        df = df[df["ticker"].isin(MOMENTUM_TICKERS)].copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        df_train = df[df["date"] < "2025-01-01"].copy()
        df_test = df[df["date"] >= "2025-01-01"].copy()
        
        if len(df_test) < 50:
            return None
        
        X_train = df_train[CLASSIC_COLS].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[CLASSIC_COLS].values.astype(np.float32)
        
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
        
        # Frais IBKR Tiered réalistes: min 0.35$/order ≈ 0.32€/order
        # Pour un round-trip (BUY+SELL) = 0.64€/trade
        ibkr_fee_per_order = 0.32
        bt = backtest_strategy(
            df_test, y_pred, y_proba,
            initial_capital=500, fee_per_order=ibkr_fee_per_order,
            slippage_pct=0.001, confidence_threshold=conf_thresh,
            position_pct=position_pct,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            multi_ticker=True,
        )
        return bt


def grid_search():
    print("=== NINJA MOMENTUM — Grid Search ===")
    print(f"Univers: {len(MOMENTUM_TICKERS)} tickers volatiles")
    print()
    
    results = []
    for H in [2, 3, 5]:
        for th in [0.02, 0.03]:
            for size in [0.1, 0.15, 0.2]:
                for sl in [None, 0.03]:
                    for tp in [None, 0.06]:
                        for conf in [0.0, 0.5]:
                            bt = test_config(H, th, size, sl, tp, conf)
                            if bt is None:
                                continue
                            results.append({
                                "H": H, "th": th, "size": size,
                                "sl": sl, "tp": tp, "conf": conf,
                                "return": bt["total_return_pct"],
                                "dd": bt["max_drawdown_pct"],
                                "trades": bt["trades_executed"],
                                "sharpe": bt["sharpe_ratio"],
                                "fees": bt["total_fees"],
                            })
    
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("Aucun résultat")
        return
    
    # Best by return (DD < 30%)
    viable = df_res[df_res["dd"] > -30]
    if len(viable) > 0:
        best = viable.loc[viable["return"].idxmax()]
        print(f"🏆 BEST RETURN (DD < 30%):")
        print(f"  H={best['H']} th={best['th']} size={best['size']*100:.0f}% SL={best['sl']} TP={best['tp']} conf={best['conf']}")
        print(f"  Return: {best['return']:+.1f}% | DD: {best['dd']:.1f}% | Trades: {best['trades']} | Sharpe: {best['sharpe']:.2f}")
    
    # Best by Sharpe
    if len(viable) > 0:
        best_s = viable.loc[viable["sharpe"].idxmax()]
        print(f"\n🏆 BEST SHARPE (DD < 30%):")
        print(f"  H={best_s['H']} th={best_s['th']} size={best_s['size']*100:.0f}% SL={best_s['sl']} TP={best_s['tp']} conf={best_s['conf']}")
        print(f"  Return: {best_s['return']:+.1f}% | DD: {best_s['dd']:.1f}% | Trades: {best_s['trades']} | Sharpe: {best_s['sharpe']:.2f}")
    
    # Best return/DD ratio
    df_res["rdd"] = df_res["return"] / df_res["dd"].abs()
    best_rdd = df_res.loc[df_res["rdd"].idxmax()]
    print(f"\n🏆 BEST RETURN/DD:")
    print(f"  H={best_rdd['H']} th={best_rdd['th']} size={best_rdd['size']*100:.0f}% SL={best_rdd['sl']} TP={best_rdd['tp']} conf={best_rdd['conf']}")
    print(f"  Return: {best_rdd['return']:+.1f}% | DD: {best_rdd['dd']:.1f}% | Trades: {best_rdd['trades']} | Ratio: {best_rdd['rdd']:.2f}")
    
    print(f"\n📊 TOP 10 BY RETURN:")
    print(f"{'H':>3} {'th':>5} {'Size':>5} {'SL':>5} {'TP':>5} {'Conf':>5} {'Return':>9} {'DD':>7} {'Trades':>7} {'Sharpe':>7}")
    print("-" * 75)
    for _, r in df_res.nlargest(10, "return").iterrows():
        print(f"{r['H']:>3} {r['th']:>5.2f} {r['size']*100:>4.0f}% {str(r['sl']):>5s} {str(r['tp']):>5s} {r['conf']:>5.1f} {r['return']:>+8.1f}% {r['dd']:>6.1f}% {int(r['trades']):>6d} {r['sharpe']:>6.2f}")


if __name__ == "__main__":
    grid_search()
