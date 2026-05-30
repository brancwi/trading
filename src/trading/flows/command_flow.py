"""Flow indépendant — traitement des commandes Hermes."""

import logging

from prefect import flow, task

from trading.core.database import db_session
from trading.execution.commands import CommandProcessor
from trading.core.models import Portfolio
from trading.events.emitters import emit_portfolio_updated

logger = logging.getLogger(__name__)


@task
def process_pending_commands() -> int:
    """Traite toutes les commandes en attente."""
    with db_session() as db:
        processor = CommandProcessor(db)
        n = processor.process_pending()
        # Émettre events pour les portfolios modifiés
        for port in db.query(Portfolio).all():
            positions_value = sum(
                p.current_value or 0 for p in port.positions
            )
            total = port.cash_current + positions_value
            pnl = total - port.cash_initial
            emit_portfolio_updated(port.id, total, pnl)
        return n


@flow(name="command_processing_flow", log_prints=True)
def command_processing_flow() -> dict:
    """Flow de traitement des commandes — déclenché par hermes.command.received."""
    count = process_pending_commands()
    logger.info(f"Commands processed: {count}")
    return {"commands_processed": count}


if __name__ == "__main__":
    command_processing_flow()
