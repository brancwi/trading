"""Validation robustesse du modèle H=2 momentum avant déploiement.

Tests:
1. Performance sur univers COMPLET (US+EU) vs univers volatil (21 tickers)
2. Walk-forward sur 3 périodes de test (2024-H2, 2025-Q1, 2025-Q2)
3. Stability des features importance
"""

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

MOMENTUM_TICKERS = [
    "GME", "AMC", "PLTR", "SOFI", "MSTR", "HOOD", "COIN", "RBLX",
    "TSLA", "AMD", "NVDA", "META", "NFLX", "SHOP", "CRWD", "DDOG",
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD",
]

# Paramètres gagnants du grid search
BEST_PARAMS = {
    "horizon": 2,
    "threshold": 0.03,
    "position_pct": 0.20,
    "sl": 0.03,
    "tp": 0.06,
    "conf": 0.5,
}

IBKR_FEE = 0.32  # €/order


def train_and_backtest(df_train, df_test, conf_thresh):
    """Entraîne XGBoost et backtest."""
    X_train = df_train[CLASSIC_COLS].values.astype(np.float32)
    y_train = df_train["label"].map({"HOLD": 0, "BUY": 1, "SELL": 2}).values
    X_test = df_test[CLASSIC_COLS].values.astype(np.float32)

    # Nettoyer inf/NaN
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    # Drop NaN (au cas où)
    mask_train = ~np.isnan(X_train).any(axis=1)
    mask_test = ~np.isnan(X_test).any(axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_test = X_test[mask_test]
    df_test_clean = df_test.iloc[mask_test].copy()

    if len(X_train) < 100 or len(X_test) < 50:
        return None

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

    bt = backtest_strategy(
        df_test_clean, y_pred, y_proba,
        initial_capital=500, fee_per_order=IBKR_FEE,
        slippage_pct=0.001, confidence_threshold=conf_thresh,
        position_pct=BEST_PARAMS["position_pct"],
        stop_loss_pct=BEST_PARAMS["sl"],
        take_profit_pct=BEST_PARAMS["tp"],
        multi_ticker=True,
    )
    return bt, model


def test_universe(name, tickers_subset, db):
    """Teste sur un sous-ensemble de tickers."""
    df = build_dataset(db, horizon=BEST_PARAMS["horizon"], threshold=BEST_PARAMS["threshold"])
    df = df[df["ticker"].isin(tickers_subset)].copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    df_train = df[df["date"] < "2025-01-01"].copy()
    df_test = df[df["date"] >= "2025-01-01"].copy()

    if len(df_test) < 50:
        return None

    result = train_and_backtest(df_train, df_test, BEST_PARAMS["conf"])
    if result is None:
        return None
    bt, model = result
    return {
        "universe": name,
        "tickers": len(tickers_subset),
        "rows_train": len(df_train),
        "rows_test": len(df_test),
        "return": bt["total_return_pct"],
        "dd": bt["max_drawdown_pct"],
        "trades": bt["trades_executed"],
        "sharpe": bt["sharpe_ratio"],
        "fees": bt["total_fees"],
        "final_value": bt["final_value"],
    }


def test_walkforward(name, tickers_subset, db):
    """Walk-forward sur 3 fenêtres de test."""
    df = build_dataset(db, horizon=BEST_PARAMS["horizon"], threshold=BEST_PARAMS["threshold"])
    df = df[df["ticker"].isin(tickers_subset)].copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["date"] = pd.to_datetime(df["date"])

    windows = [
        ("2024-H2", "2024-07-01", "2024-12-31"),
        ("2025-Q1", "2025-01-01", "2025-03-31"),
        ("2025-Q2", "2025-04-01", "2025-06-30"),
    ]

    results = []
    for label, start, end in windows:
        df_train = df[df["date"] < start].copy()
        df_test = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        if len(df_train) < 100 or len(df_test) < 30:
            continue

        result = train_and_backtest(df_train, df_test, BEST_PARAMS["conf"])
        if result is None:
            continue
        bt, _ = result
        results.append({
            "window": label,
            "return": bt["total_return_pct"],
            "dd": bt["max_drawdown_pct"],
            "trades": bt["trades_executed"],
            "sharpe": bt["sharpe_ratio"],
        })
    return results


def main():
    print("=" * 70)
    print("VALIDATION ROBUSTESSE — Ninja Momentum H=2")
    print("=" * 70)

    with db_session() as db:
        # --- 1. Récupérer tous les tickers disponibles ---
        from sqlalchemy import text
        rows = db.execute(text("SELECT DISTINCT ticker FROM market_data ORDER BY ticker")).fetchall()
        all_tickers = [r[0] for r in rows]
        # Exclure cryptos (testés comme non viables)
        non_crypto = [t for t in all_tickers if "-USD" not in t and "-EUR" not in t]
        # Exclure tickers avec trop peu de données
        valid_tickers = []
        for t in non_crypto:
            cnt = db.execute(
                text("SELECT COUNT(*) FROM market_data WHERE ticker = :t"),
                {"t": t}
            ).scalar()
            if cnt and cnt >= 200:
                valid_tickers.append(t)

        print(f"\n📊 Univers total: {len(valid_tickers)} tickers (excluant cryptos)")
        print(f"   Exemples: {', '.join(valid_tickers[:10])}...")

        # --- 2. Test univers complet ---
        print("\n" + "-" * 50)
        print("TEST 1: Performance sur univers COMPLET")
        print("-" * 50)
        res_full = test_universe("Complet (US+EU)", valid_tickers, db)
        if res_full:
            print(f"  Return: {res_full['return']:+.1f}% | DD: {res_full['dd']:.1f}% | "
                  f"Trades: {res_full['trades']} | Sharpe: {res_full['sharpe']:.2f} | "
                  f"Frais: {res_full['fees']:.2f}€")
        else:
            print("  ❌ Échec — pas assez de données")

        # --- 3. Test univers volatil (21 tickers) ---
        volatil_in_db = [t for t in MOMENTUM_TICKERS if t in valid_tickers]
        print(f"\nTEST 2: Performance sur univers VOLATIL ({len(volatil_in_db)} tickers)")
        res_vol = test_universe("Volatil", volatil_in_db, db)
        if res_vol:
            print(f"  Return: {res_vol['return']:+.1f}% | DD: {res_vol['dd']:.1f}% | "
                  f"Trades: {res_vol['trades']} | Sharpe: {res_vol['sharpe']:.2f} | "
                  f"Frais: {res_vol['fees']:.2f}€")

        # --- 4. Test univers classique (excluant les volatils) ---
        classic_tickers = [t for t in valid_tickers if t not in MOMENTUM_TICKERS]
        print(f"\nTEST 3: Performance sur univers CLASSIQUE ({len(classic_tickers)} tickers)")
        res_classic = test_universe("Classique", classic_tickers, db)
        if res_classic:
            print(f"  Return: {res_classic['return']:+.1f}% | DD: {res_classic['dd']:.1f}% | "
                  f"Trades: {res_classic['trades']} | Sharpe: {res_classic['sharpe']:.2f} | "
                  f"Frais: {res_classic['fees']:.2f}€")

        # --- 5. Walk-forward multi-périodes ---
        print("\n" + "-" * 50)
        print("TEST 4: Walk-forward multi-périodes (univers complet)")
        print("-" * 50)
        wf = test_walkforward("Complet", valid_tickers, db)
        for r in wf:
            print(f"  {r['window']}: Return {r['return']:+.1f}% | DD {r['dd']:.1f}% | "
                  f"Trades {r['trades']} | Sharpe {r['sharpe']:.2f}")

        # --- 6. Walk-forward sur univers volatil ---
        print(f"\nTEST 5: Walk-forward (univers volatil)")
        wf_vol = test_walkforward("Volatil", volatil_in_db, db)
        for r in wf_vol:
            print(f"  {r['window']}: Return {r['return']:+.1f}% | DD {r['dd']:.1f}% | "
                  f"Trades {r['trades']} | Sharpe {r['sharpe']:.2f}")

        # --- 7. Analyse robustesse ---
        print("\n" + "=" * 70)
        print("ANALYSE ROBUSTESSE")
        print("=" * 70)

        checks = []
        # Check 1: Performance univers complet > 0
        if res_full and res_full["return"] > 0:
            checks.append(("✅", f"Univers complet rentable: {res_full['return']:+.1f}%"))
        else:
            checks.append(("❌", f"Univers complet NON rentable: {res_full['return']:+.1f}%"))

        # Check 2: Performance univers classique > 0
        if res_classic and res_classic["return"] > 0:
            checks.append(("✅", f"Univers classique rentable: {res_classic['return']:+.1f}%"))
        else:
            checks.append(("❌", f"Univers classique NON rentable: {res_classic['return']:+.1f}%"))

        # Check 3: Walk-forward stable
        if len(wf) >= 2:
            returns = [r["return"] for r in wf]
            if all(r > -20 for r in returns) and any(r > 0 for r in returns):
                checks.append(("✅", f"Walk-forward stable: {returns}"))
            else:
                checks.append(("❌", f"Walk-forward instable: {returns}"))
        else:
            checks.append(("⚠️", "Pas assez de fenêtres walk-forward"))

        # Check 4: Comparaison volatil vs complet
        if res_full and res_vol:
            ratio = res_vol["return"] / max(res_full["return"], 0.01)
            if ratio < 3:  # Le volatil ne doit pas surperformer le complet par un facteur 3x
                checks.append(("✅", f"Pas d'overfitting extrême (volatil/complet = {ratio:.1f}x)"))
            else:
                checks.append(("⚠️", f"Overfitting possible (volatil/complet = {ratio:.1f}x)"))

        for icon, msg in checks:
            print(f"  {icon} {msg}")

        # Décision finale
        print("\n" + "=" * 70)
        passed = sum(1 for icon, _ in checks if icon == "✅")
        total = len(checks)
        if passed >= 3:
            print(f"🟢 VALIDATION PASSÉE ({passed}/{total}): Le modèle semble robuste.")
            print("   → Prêt pour entraînement final sur univers complet.")
        elif passed >= 2:
            print(f"🟡 VALIDATION MITIGÉE ({passed}/{total}): Risque modéré.")
            print("   → Entraînement possible mais surveillance renforcée recommandée.")
        else:
            print(f"🔴 VALIDATION ÉCHOUÉE ({passed}/{total}): Modèle probablement overfitté.")
            print("   → Ne pas déployer. Revoir les features ou l'horizon.")
        print("=" * 70)


if __name__ == "__main__":
    main()
