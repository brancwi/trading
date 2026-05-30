#!/usr/bin/env python3
"""Valide automatiquement les sentiment_scores à posteriori via le prix du marché.

Pour chaque sentiment_score non validé, compare le prix du ticker à T=analyse
et à T+window (ex: 4h, 24h). Déduit le label validé :
  - price_change > +threshold%  → "positive"
  - price_change < -threshold%  → "negative"
  - sinon                       → "neutral"

Usage:
    uv run python scripts/validate_sentiment_labels.py
    uv run python scripts/validate_sentiment_labels.py --window-hours 24 --threshold 2.0
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text

from trading.core.database import db_session, engine
from trading.core.models import SentimentScore


def get_price_at(ticker: str, timestamp: datetime) -> float | None:
    """Récupère le prix le plus proche d'un ticker à un moment donné."""
    with engine.connect() as conn:
        # Cherche le prix juste avant ou au timestamp
        row = conn.execute(
            text("""
                SELECT price FROM market_data
                WHERE ticker = :ticker AND timestamp <= :ts
                ORDER BY timestamp DESC
                LIMIT 1
            """),
            {"ticker": ticker, "ts": timestamp},
        ).fetchone()
        if row:
            return float(row[0])
        # Fallback : premier prix après le timestamp
        row = conn.execute(
            text("""
                SELECT price FROM market_data
                WHERE ticker = :ticker AND timestamp >= :ts
                ORDER BY timestamp ASC
                LIMIT 1
            """),
            {"ticker": ticker, "ts": timestamp},
        ).fetchone()
        if row:
            return float(row[0])
    return None


def validate_score(score: SentimentScore, window_hours: int, threshold_pct: float) -> bool:
    """Valide un sentiment_score. Retourne True si mis à jour."""
    if score.validated_label is not None:
        return False  # Déjà validé

    price_at_analysis = get_price_at(score.ticker, score.timestamp)
    if price_at_analysis is None:
        return False  # Pas de prix de référence

    validation_ts = score.timestamp + timedelta(hours=window_hours)
    price_at_validation = get_price_at(score.ticker, validation_ts)
    if price_at_validation is None:
        return False  # Pas assez d'historique de prix encore

    change_pct = ((price_at_validation - price_at_analysis) / price_at_analysis) * 100

    if change_pct > threshold_pct:
        validated_label = "positive"
    elif change_pct < -threshold_pct:
        validated_label = "negative"
    else:
        validated_label = "neutral"

    with db_session() as db:
        s = db.query(SentimentScore).filter_by(id=score.id).first()
        if s:
            s.validated_label = validated_label
            s.validated_at = datetime.utcnow()
            s.validation_method = "market_price"
            s.price_at_analysis = price_at_analysis
            s.price_at_validation = price_at_validation
            s.price_change_pct = round(change_pct, 4)
            db.commit()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation auto des sentiment scores")
    parser.add_argument("--window-hours", type=int, default=4, help="Fenêtre de validation en heures (défaut: 4)")
    parser.add_argument("--threshold", type=float, default=1.0, help="Seuil de variation en %% (défaut: 1.0)")
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans écrire en DB")
    args = parser.parse_args()

    with db_session() as db:
        scores = db.query(SentimentScore).filter(SentimentScore.validated_label.is_(None)).all()

    updated = 0
    skipped = 0
    for score in scores:
        if args.dry_run:
            price_a = get_price_at(score.ticker, score.timestamp)
            price_v = get_price_at(score.ticker, score.timestamp + timedelta(hours=args.window_hours))
            if price_a and price_v:
                change = ((price_v - price_a) / price_a) * 100
                label = "positive" if change > args.threshold else "negative" if change < -args.threshold else "neutral"
                print(f"[DRY-RUN] id={score.id} {score.ticker} change={change:.2f}% → {label}")
                updated += 1
            else:
                skipped += 1
        else:
            if validate_score(score, args.window_hours, args.threshold):
                updated += 1
            else:
                skipped += 1

    print(f"✅ {updated} scores validés, {skipped} ignorés (pas assez d'historique ou déjà validés)")
    print(f"   Fenêtre: T+{args.window_hours}h, Seuil: ±{args.threshold}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
