"""Notifier Telegram - envoi de messages."""

import logging
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.models import Alert

logger = logging.getLogger(__name__)
settings = get_settings()


class TelegramNotifier:
    """Client minimal Telegram via HTTP (sans dépendance lourde)."""

    def __init__(self) -> None:
        self.token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.http = httpx.Client(timeout=15.0)

    def send(self, message: str, db: Session | None = None) -> bool:
        """Envoie un message Markdown simple."""
        if not self.token or not self.chat_id:
            logger.warning("Telegram non configuré")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }
        try:
            resp = self.http.post(url, json=payload)
            resp.raise_for_status()
            if db:
                alert = Alert(alert_type="info", message=message)
                db.add(alert)
                db.commit()
            return True
        except Exception as e:
            logger.error(f"Telegram échec: {e}")
            if db:
                alert = Alert(alert_type="error", message=message, error=str(e))
                db.add(alert)
                db.commit()
            return False

    def notify_signal(self, ticker: str, action: str, sentiment: float, strength: float, db: Session | None = None) -> bool:
        emoji = "🟢" if "BUY" in action else "🔴" if "SELL" in action else "⚪"
        msg = (
            f"{emoji} *SIGNAL {action}*\n"
            f"Ticker: `{ticker}`\n"
            f"Sentiment: {sentiment:+.2f}\n"
            f"Force: {strength:.2f}/1.0"
        )
        return self.send(msg, db)

    def notify_trade(self, portfolio: str, action: str, ticker: str, qty: float, price: float, amount: float, pnl: float | None = None, db: Session | None = None) -> bool:
        emoji = "✅" if action == "BUY" else "💰"
        msg = (
            f"{emoji} *TRADE {action}* | `{portfolio}`\n"
            f"Ticker: `{ticker}`\n"
            f"Qty: {qty:.3f}\n"
            f"Prix: ${price:.2f}\n"
            f"Montant: ${amount:.2f}"
        )
        if pnl is not None:
            msg += f"\nPnL: ${pnl:+.2f}"
        return self.send(msg, db)

    def notify_summary(self, portfolio_values: dict[str, float], db: Session | None = None) -> bool:
        lines = ["📊 *RÉSUMÉ JOURNALIER*"]
        total = 0.0
        for name, value in portfolio_values.items():
            lines.append(f"{name}: ${value:,.2f}")
            total += value
        lines.append(f"*TOTAL: ${total:,.2f}*")
        return self.send("\n".join(lines), db)
