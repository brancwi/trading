"""Flow indépendant — validation a posteriori des labels sentiment."""

import logging
from datetime import datetime

from prefect import flow, task

from trading.core.database import db_session
from trading.core.models import SentimentScore, MarketData

logger = logging.getLogger(__name__)

# Seuil de variation pour considérer une validation positive/negative
_VALIDATION_THRESHOLD_PCT = 1.0


def _change_to_label(change_pct: float) -> str:
    if change_pct > _VALIDATION_THRESHOLD_PCT:
        return "positive"
    if change_pct < -_VALIDATION_THRESHOLD_PCT:
        return "negative"
    return "neutral"


@task(retries=1, retry_delay_seconds=30)
def validate_pending_scores() -> dict:
    """Valide les scores non encore validés en comparant le prix actuel."""
    validated_count = 0
    skipped_count = 0

    with db_session() as db:
        pending = (
            db.query(SentimentScore)
            .filter(
                SentimentScore.validated_label.is_(None),
                SentimentScore.price_at_analysis.isnot(None),
            )
            .all()
        )

        for score in pending:
            latest = (
                db.query(MarketData)
                .filter(MarketData.ticker == score.ticker)
                .order_by(MarketData.timestamp.desc())
                .first()
            )

            if latest is None or latest.price is None:
                skipped_count += 1
                continue

            price_before = score.price_at_analysis
            price_now = latest.price
            change_pct = ((price_now - price_before) / price_before) * 100.0
            validated_label = _change_to_label(change_pct)

            score.price_at_validation = price_now
            score.price_change_pct = round(change_pct, 4)
            score.validated_label = validated_label
            score.validated_at = datetime.utcnow()
            score.validation_method = "market_price"
            validated_count += 1

        db.commit()

    logger.info(
        f"Validation: {validated_count} validés, {skipped_count} ignorés "
        f"(sur {len(pending)} pending)"
    )
    return {"validated": validated_count, "skipped": skipped_count}


@flow(name="validation_flow", log_prints=True)
def validation_flow() -> dict:
    """Flow de validation a posteriori — déclenché toutes les 4 heures."""
    result = validate_pending_scores()
    logger.info(f"Validation flow terminé: {result}")
    return result


if __name__ == "__main__":
    validation_flow()
