"""Flow indépendant — envoi des notifications Telegram."""

import logging
from datetime import datetime

from prefect import flow, task

from trading.core.database import db_session
from trading.notifier.telegram import TelegramNotifier
from trading.core.models import Portfolio

logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=5)
def send_daily_summary() -> bool:
    """Envoie le résumé quotidien si heure >= 20h."""
    now = datetime.now()
    if now.hour < 20:
        return False
    notifier = TelegramNotifier()
    with db_session() as db:
        values = {}
        for port in db.query(Portfolio).all():
            positions_value = sum(p.current_value or 0 for p in port.positions)
            values[port.id] = port.cash_current + positions_value
        return notifier.notify_summary(values, db=db)


@task(retries=3, retry_delay_seconds=5)
def notify_trade_event(portfolio_id: str, ticker: str, action: str, amount: float) -> bool:
    """Notifie un trade spécifique (utilisé par automation ou manuel)."""
    notifier = TelegramNotifier()
    with db_session() as db:
        return notifier.notify_trade(
            portfolio=portfolio_id,
            action=action,
            ticker=ticker,
            qty=0,  # qty non passée dans ce flow simplifié
            price=0,
            amount=amount,
            db=db,
        )


@flow(name="notifications_flow", log_prints=True)
def notifications_flow(notify_summary: bool = False) -> dict:
    """Flow de notifications — déclenché par event ou schedule."""
    sent = False
    if notify_summary:
        sent = send_daily_summary()
    return {"summary_sent": sent}


if __name__ == "__main__":
    notifications_flow(notify_summary=True)
