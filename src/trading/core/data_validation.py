"""Validation et nettoyage des données de marché.

Détecte les anomalies courantes :
  - Prix négatifs ou nuls
  - Gaps de prix extrêmes (>50% d'un jour à l'autre)
  - Volumes nuls ou aberrants
  - Données manquantes (OHLC incomplètes)
  - Prix hors range historique (split/reverse split non déclaré)

Usage:
    from trading.core.data_validation import validate_market_data, clean_market_data
    issues = validate_market_data(db)
    clean_market_data(db, dry_run=False)
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text

from trading.core.models import MarketData

logger = logging.getLogger(__name__)


# Seuils de validation
MAX_PRICE_GAP_PCT = 0.50       # 50% de gap d'un jour à l'autre = suspect
MAX_PRICE_DROP_PCT = 0.90      # 90% de baisse en un jour = très suspect
MIN_VALID_PRICE = 0.001        # Prix minimum acceptable
MAX_VOLUME_ZERO_DAYS = 3       # Max 3 jours de volume nul consécutifs
MIN_VOLUME_FOR_PRICE = 1       # Volume minimum pour un prix valide


def _get_market_data_df(db: Session, tickers: list[str] | None = None) -> pd.DataFrame:
    """Charge les données de marché en DataFrame."""
    query = db.query(MarketData).order_by(MarketData.ticker, MarketData.timestamp.asc())
    if tickers:
        query = query.filter(MarketData.ticker.in_(tickers))
    
    rows = query.all()
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame([
        {
            "id": r.id,
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "open": r.open_price,
            "high": r.high,
            "low": r.low,
            "close": r.price,
            "volume": r.volume,
            "source": r.source,
        }
        for r in rows
    ])
    return df


def validate_market_data(db: Session, tickers: list[str] | None = None) -> dict:
    """Valide les données de marché et retourne un rapport d'anomalies.
    
    Returns:
        Dict avec les anomalies détectées par catégorie.
    """
    df = _get_market_data_df(db, tickers)
    if df.empty:
        return {"status": "no_data", "issues": []}
    
    issues = []
    
    # 1. Prix négatifs ou nuls
    invalid_price = df[(df["close"] <= 0) | (df["close"].isna())]
    for _, row in invalid_price.iterrows():
        issues.append({
            "type": "INVALID_PRICE",
            "severity": "CRITICAL",
            "ticker": row["ticker"],
            "date": row["timestamp"],
            "value": row["close"],
            "message": f"Prix invalide: {row['close']}",
        })
    
    # 2. OHLC incomplet
    incomplete = df[(df["open"].isna()) | (df["high"].isna()) | (df["low"].isna())]
    for _, row in incomplete.iterrows():
        issues.append({
            "type": "INCOMPLETE_OHLC",
            "severity": "WARNING",
            "ticker": row["ticker"],
            "date": row["timestamp"],
            "message": "Données OHLC incomplètes",
        })
    
    # 3. Gaps de prix extrêmes (par ticker)
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("timestamp").copy()
        g["prev_close"] = g["close"].shift(1)
        g["price_change"] = (g["close"] / g["prev_close"] - 1).abs()
        
        extreme_gaps = g[g["price_change"] > MAX_PRICE_GAP_PCT]
        for _, row in extreme_gaps.iterrows():
            issues.append({
                "type": "EXTREME_GAP",
                "severity": "ERROR",
                "ticker": ticker,
                "date": row["timestamp"],
                "value": row["price_change"],
                "message": f"Gap de prix extrême: {row['price_change']*100:.1f}% (close={row['close']}, prev={row['prev_close']})",
            })
        
        # 4. Prix très bas (< 0.01) pour des actifs qui ne devraient pas l'être
        very_low = g[g["close"] < MIN_VALID_PRICE]
        for _, row in very_low.iterrows():
            issues.append({
                "type": "VERY_LOW_PRICE",
                "severity": "ERROR",
                "ticker": ticker,
                "date": row["timestamp"],
                "value": row["close"],
                "message": f"Prix anormalement bas: {row['close']}",
            })
        
        # 5. Volume nul sur plusieurs jours consécutifs
        g["volume_zero"] = (g["volume"] == 0) | (g["volume"].isna())
        g["zero_streak"] = g["volume_zero"].astype(int).groupby((~g["volume_zero"]).cumsum()).cumsum()
        long_streaks = g[g["zero_streak"] > MAX_VOLUME_ZERO_DAYS]
        for _, row in long_streaks.iterrows():
            issues.append({
                "type": "ZERO_VOLUME_STREAK",
                "severity": "WARNING",
                "ticker": ticker,
                "date": row["timestamp"],
                "value": row["zero_streak"],
                "message": f"Volume nul pendant {row['zero_streak']} jours consécutifs",
            })
        
        # 6. High < Low ou High < Close ou Low > Close
        illogical = g[
            (g["high"] < g["low"]) |
            (g["high"] < g["close"]) |
            (g["low"] > g["close"]) |
            (g["high"] < g["open"]) |
            (g["low"] > g["open"])
        ]
        for _, row in illogical.iterrows():
            issues.append({
                "type": "ILLOGICAL_OHLC",
                "severity": "ERROR",
                "ticker": ticker,
                "date": row["timestamp"],
                "message": f"OHLC illogique: O={row['open']} H={row['high']} L={row['low']} C={row['close']}",
            })
    
    # Résumé
    critical = len([i for i in issues if i["severity"] == "CRITICAL"])
    errors = len([i for i in issues if i["severity"] == "ERROR"])
    warnings = len([i for i in issues if i["severity"] == "WARNING"])
    
    logger.info("[DataValidation] %d issues found: %d critical, %d errors, %d warnings",
                len(issues), critical, errors, warnings)
    
    return {
        "status": "ok" if len(issues) == 0 else "issues_found",
        "total_issues": len(issues),
        "critical": critical,
        "errors": errors,
        "warnings": warnings,
        "issues": issues,
    }


def clean_market_data(db: Session, dry_run: bool = True, tickers: list[str] | None = None) -> dict:
    """Nettoie les données de marché en supprimant les anomalies.
    
    Args:
        db: Session SQLAlchemy
        dry_run: Si True, ne supprime rien — retourne juste ce qui serait supprimé
        tickers: Liste de tickers spécifiques à nettoyer (None = tous)
    
    Returns:
        Rapport du nettoyage
    """
    report = validate_market_data(db, tickers)
    deleted = 0
    
    if dry_run:
        logger.info("[DataValidation] DRY RUN — %d records would be deleted", report["critical"] + report["errors"])
        return {**report, "dry_run": True, "deleted": 0}
    
    # Supprimer les anomalies CRITICAL et ERROR
    ids_to_delete = []
    for issue in report["issues"]:
        if issue["severity"] in ("CRITICAL", "ERROR") and "id" in issue:
            ids_to_delete.append(issue["id"])
    
    # Also delete by querying directly for invalid prices
    from sqlalchemy import delete
    
    # Prix négatifs ou nuls
    stmt = delete(MarketData).where(MarketData.price <= 0)
    if tickers:
        stmt = stmt.where(MarketData.ticker.in_(tickers))
    result = db.execute(stmt)
    deleted += result.rowcount
    
    # Prix très bas (< 0.001) — à vérifier manuellement car certains penny stocks sont légitimes
    # On ne supprime pas automatiquement, on les signale juste
    
    db.commit()
    
    logger.info("[DataValidation] Deleted %d invalid records", deleted)
    return {**report, "dry_run": False, "deleted": deleted}


if __name__ == "__main__":
    from trading.core.database import db_session
    with db_session() as db:
        report = validate_market_data(db)
        print(f"Status: {report['status']}")
        print(f"Total issues: {report['total_issues']} (C:{report['critical']} E:{report['errors']} W:{report['warnings']})")
        for issue in report["issues"][:20]:
            print(f"  [{issue['severity']}] {issue['ticker']} @ {issue['date']}: {issue['message']}")
