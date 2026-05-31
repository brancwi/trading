"""Génération de labels d'entraînement par future-return.

Pour chaque point de prix historique (ticker, date, close), on calcule le
rendement futur sur un horizon H (ex: 5 jours) :

    future_return = (Close_{t+H} / Close_t) - 1

Puis on assigne un label :
    BUY  si future_return >  +threshold
    SELL si future_return <  -threshold
    HOLD sinon

Les labels sont stockés dans `training_labels` pour l'entraînement de
modèles supervisés (XGBoost, LSTM, etc.).
"""

import logging
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from trading.core.database import db_session
from trading.core.models import MarketData, TrainingLabel

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = [1, 5, 10, 20]
DEFAULT_THRESHOLDS = [0.01, 0.03, 0.05]


def _label_from_return(ret: float, threshold: float) -> str:
    if ret > threshold:
        return "BUY"
    if ret < -threshold:
        return "SELL"
    return "HOLD"


def generate_labels(
    db: Session,
    tickers: list[str] | None = None,
    horizons: list[int] = None,
    thresholds: list[float] = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Génère les labels future-return pour tous les prix historiques.

    Args:
        db: Session SQLAlchemy
        tickers: Liste de tickers (None = tous ceux présents en DB)
        horizons: Liste d'horizons en jours (défaut: [1, 5, 10, 20])
        thresholds: Liste de seuils (défaut: [0.01, 0.03, 0.05])
        dry_run: Si True, ne persiste pas en base

    Returns:
        Dict {"BUY": N, "SELL": M, "HOLD": K, "total": T}
    """
    horizons = horizons or DEFAULT_HORIZONS
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # Liste des tickers
    if tickers is None:
        rows = db.execute(text("SELECT DISTINCT ticker FROM market_data ORDER BY ticker")).fetchall()
        tickers = [r[0] for r in rows]

    stats: dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0, "total": 0}

    for ticker in tickers:
        logger.info("[Labeller] Processing %s", ticker)

        # Récupérer les prix chronologiques pour ce ticker
        prices = (
            db.query(MarketData)
            .filter(MarketData.ticker == ticker)
            .order_by(MarketData.timestamp.asc())
            .all()
        )
        if len(prices) < max(horizons) + 1:
            logger.warning("[Labeller] %s: only %d rows, skipping", ticker, len(prices))
            continue

        # Index rapide : position → prix
        closes = [p.price for p in prices]
        timestamps = [p.timestamp for p in prices]

        for horizon in horizons:
            for threshold in thresholds:
                for i in range(len(prices) - horizon):
                    close_t = closes[i]
                    close_future = closes[i + horizon]
                    ret = (close_future / close_t) - 1.0
                    label = _label_from_return(ret, threshold)

                    if not dry_run:
                        tl = TrainingLabel(
                            timestamp=timestamps[i],
                            ticker=ticker,
                            horizon_days=horizon,
                            close_at_signal=round(close_t, 4),
                            future_close=round(close_future, 4),
                            future_return=round(ret * 100, 4),  # en %
                            label=label,
                            threshold_used=round(threshold, 4),
                        )
                        db.add(tl)

                    stats[label] += 1
                    stats["total"] += 1

        if not dry_run:
            db.commit()
            logger.info("[Labeller] %s committed", ticker)

    logger.info(
        "[Labeller] Done — total=%d  BUY=%d  SELL=%d  HOLD=%d",
        stats["total"], stats["BUY"], stats["SELL"], stats["HOLD"],
    )
    return stats


def label_summary(db: Session) -> dict:
    """Retourne un résumé des labels existants."""
    rows = db.execute(text("""
        SELECT
            horizon_days,
            threshold_used,
            label,
            COUNT(*) as cnt,
            AVG(future_return) as avg_ret,
            MIN(future_return) as min_ret,
            MAX(future_return) as max_ret
        FROM training_labels
        GROUP BY horizon_days, threshold_used, label
        ORDER BY horizon_days, threshold_used, label
    """)).fetchall()

    result: dict = {}
    for r in rows:
        key = f"H{r.horizon_days}_th{r.threshold_used}"
        if key not in result:
            result[key] = {}
        result[key][r.label] = {
            "count": r.cnt,
            "avg_return_pct": round(r.avg_ret, 2) if r.avg_ret else None,
            "min_return_pct": round(r.min_ret, 2) if r.min_ret else None,
            "max_return_pct": round(r.max_ret, 2) if r.max_ret else None,
        }
    return result


if __name__ == "__main__":
    import sys
    dry = "--dry-run" in sys.argv
    with db_session() as db:
        stats = generate_labels(db, dry_run=dry)
        print("\n=== Label Stats ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        if not dry:
            print("\n=== Summary by horizon/threshold ===")
            summary = label_summary(db)
            for key, data in summary.items():
                print(f"\n{key}:")
                for label, info in data.items():
                    print(f"  {label}: {info['count']} (avg ret: {info['avg_return_pct']}%)")
