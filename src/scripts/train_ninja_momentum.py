"""Entraînement spécialisé du modèle ninja momentum H=2 sur tickers volatils."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trading.core.database import db_session
from trading.core.models import Portfolio
from trading.ml.dataset_builder import build_dataset
from trading.ml.trainer import save_model
from trading.ml.evaluator import backtest_strategy

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

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

VOLATIL_TICKERS = [
    # Chers (performance historique +40%)
    "PLTR", "TSLA", "AMD", "NVDA", "META", "NFLX", "SHOP", "CRWD", "DDOG",
    # Moins chers (< 120$) — accessibles avec petit capital
    "LYFT",   # 14$  — rideshare
    "FSLY",   # 18$  — CDN/cloud
    "UBER",   # 70$  — rideshare
    "ZM",     # 101$ — vidéoconf
]

HYPER = {
    "horizon": 2,
    "threshold": 0.03,
    "n_estimators": 150,
    "max_depth": 4,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "slippage_pct": 0.001,
    "confidence_threshold": 0.0,
    "fee_per_order": 0.32,
    "leverage": 1.0,
    "take_profit_pct": 0.06,
    "stop_loss_pct": 0.03,
    "position_pct": 0.2,
}


def main():
    print("=" * 60)
    print("ENTRAÎNEMENT NINJA MOMENTUM H=2 — Tickers volatils")
    print("=" * 60)

    with db_session() as db:
        # 1) Dataset
        print("\n[1/4] Construction du dataset...")
        df = build_dataset(db, horizon=HYPER["horizon"], threshold=HYPER["threshold"])
        
        # Filtre uniquement les tickers volatils présents en DB
        available = set(df["ticker"].unique())
        vols = [t for t in VOLATIL_TICKERS if t in available]
        print(f"   Tickers volatils disponibles: {vols} ({len(vols)}/{len(VOLATIL_TICKERS)})")
        
        df = df[df["ticker"].isin(vols)].copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["date"] = pd.to_datetime(df["date"])
        
        # 2) Split temporel
        print("\n[2/4] Split train/test...")
        df_train = df[df["date"] < "2025-01-01"].copy()
        df_test = df[df["date"] >= "2025-01-01"].copy()
        print(f"   Train: {len(df_train)} rows | Test: {len(df_test)} rows")
        
        # 3) Entraînement
        print("\n[3/4] Entraînement XGBoost...")
        X_train = df_train[CLASSIC_COLS].values.astype(np.float32)
        y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        X_test = df_test[CLASSIC_COLS].values.astype(np.float32)
        y_test = df_test["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
        
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=HYPER["n_estimators"],
            max_depth=HYPER["max_depth"],
            learning_rate=HYPER["learning_rate"],
            subsample=HYPER["subsample"],
            colsample_bytree=HYPER["colsample_bytree"],
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=4,
            reg_alpha=0.1,
            reg_lambda=1.0,
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # 4) Backtest
        print("\n[4/4] Backtest...")
        bt = backtest_strategy(
            df_test, y_pred, y_proba,
            initial_capital=500,
            fee_per_order=HYPER["fee_per_order"],
            slippage_pct=HYPER["slippage_pct"],
            confidence_threshold=HYPER["confidence_threshold"],
            leverage=HYPER["leverage"],
            take_profit_pct=HYPER["take_profit_pct"],
            stop_loss_pct=HYPER["stop_loss_pct"],
            position_pct=HYPER["position_pct"],
            multi_ticker=True,
        )
        
        print(f"\n{'='*60}")
        print("RÉSULTATS BACKTEST")
        print(f"{'='*60}")
        print(f"  Return:        {bt['total_return_pct']:+.1f}%")
        print(f"  Max Drawdown:  {bt['max_drawdown_pct']:.1f}%")
        print(f"  Trades:        {bt['trades_executed']}")
        print(f"  Sharpe:        {bt['sharpe_ratio']:.2f}")
        print(f"  Frais totaux:  {bt['total_fees']:.2f}€")
        print(f"  Valeur finale: {bt['final_value']:.2f}€")
        print(f"{'='*60}")
        
        # 5) Metrics classification
        from sklearn.metrics import accuracy_score, f1_score
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "backtest": bt,
        }
        
        # 6) Feature importance
        feat_imp = list(zip(CLASSIC_COLS, model.feature_importances_))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        print("\nTop 10 features:")
        for name, imp in feat_imp[:10]:
            print(f"  {name}: {imp:.4f}")
        
        # 7) Sauvegarde
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        hyper = dict(HYPER)
        hyper["tickers_trained"] = vols
        result = {
            "model": model,
            "scaler": scaler,
            "feature_names": CLASSIC_COLS,
            "hyperparams": hyper,
            "portfolio_id": "staging-ninja",
            "metrics": metrics,
        }
        model_path = MODELS_DIR / f"signal_staging-ninja_H{HYPER['horizon']}_th{HYPER['threshold']:.2f}_walkforward.pkl"
        save_model(result, model_path)
        print(f"\n💾 Modèle sauvegardé: {model_path}")
        
        # 7) Mise à jour du portfolio en DB
        port = db.query(Portfolio).filter(Portfolio.id == "staging-ninja").first()
        if port:
            port.fee_per_order = HYPER["fee_per_order"]
            db.commit()
            print(f"📝 Portfolio staging-ninja mis à jour: fee={HYPER['fee_per_order']}€/order")
        
        return 0


if __name__ == "__main__":
    sys.exit(main())
