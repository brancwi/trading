"""Flow Prefect — Génération de signaux ML par portfolio.

Déclenché périodiquement (ex: toutes les heures) pour générer des signaux
BUY/SELL à partir des modèles XGBoost entraînés.
"""

import logging
from typing import Any

from prefect import flow, task

from trading.core.database import db_session
from trading.core.models import Portfolio
from trading.ml.signal_generator import MLSignalGenerator

logger = logging.getLogger(__name__)


@task
def generate_ml_signals_for_portfolio(portfolio_id: str) -> dict[str, Any]:
    """Génère des signaux ML pour un portfolio donné."""
    with db_session() as db:
        gen = MLSignalGenerator(portfolio_id)
        signals = gen.generate_signals(db)
        return {
            "portfolio_id": portfolio_id,
            "signals_count": len(signals),
            "tickers": [s.ticker for s in signals],
            "actions": [s.action for s in signals],
        }


@flow(name="ml_signal_generation_flow", log_prints=True)
def ml_signal_generation_flow(portfolio_id: str | None = None) -> dict[str, Any]:
    """Flow de génération de signaux ML.

    Args:
        portfolio_id: Si fourni, génère uniquement pour ce portfolio.
                     Sinon, génère pour tous les portfolios actifs.
    """
    if portfolio_id:
        result = generate_ml_signals_for_portfolio(portfolio_id)
        logger.info("ML signals for %s: %d generated", portfolio_id, result["signals_count"])
        return {"results": [result]}

    # Mode multi-portfolio
    with db_session() as db:
        portfolios = db.query(Portfolio).filter(Portfolio.status == "active").all()
        portfolio_ids = [p.id for p in portfolios]

    results = []
    for pid in portfolio_ids:
        try:
            result = generate_ml_signals_for_portfolio(pid)
            results.append(result)
            logger.info("ML signals for %s: %d generated", pid, result["signals_count"])
        except Exception as e:
            logger.exception("ML signal generation failed for %s: %s", pid, e)
            results.append({"portfolio_id": pid, "signals_count": 0, "error": str(e)})

    return {"results": results}


if __name__ == "__main__":
    ml_signal_generation_flow()
