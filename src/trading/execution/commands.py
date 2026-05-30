"""Command Bus - traite les commandes en file d'attente."""

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from trading.core.models import Command, Portfolio, Trade
from trading.strategies.base import StrategyBase
from trading.strategies.simulation import SimulationStrategy
from trading.strategies.rotation import RotationStrategy
from trading.strategies.ninja import NinjaStrategy

logger = logging.getLogger(__name__)

STRATEGY_MAP = {
    "simulation": SimulationStrategy,
    "rotation": RotationStrategy,
    "ninja": NinjaStrategy,
}


class CommandProcessor:
    """Lit la table commands et exécute les actions demandées."""

    def __init__(self, db: Session):
        self.db = db

    def process_pending(self) -> int:
        """Traite toutes les commandes pending. Retourne le nombre traité."""
        cmds = self.db.query(Command).filter(Command.status == "pending").order_by(Command.created_at).all()
        count = 0
        for cmd in cmds:
            try:
                cmd.status = "processing"
                self.db.commit()
                result = self._execute(cmd)
                cmd.status = "completed"
                cmd.result = json.dumps(result)
                count += 1
            except Exception as e:
                logger.exception(f"Commande {cmd.id} échouée: {e}")
                cmd.status = "failed"
                cmd.result = str(e)
            finally:
                cmd.processed_at = datetime.now()
                self.db.commit()
        return count

    def _execute(self, cmd: Command) -> dict[str, Any]:
        if cmd.command_type == "LIQUIDATE":
            return self._cmd_liquidate(cmd)
        elif cmd.command_type == "PAUSE":
            return self._cmd_pause(cmd)
        elif cmd.command_type == "RESUME":
            return self._cmd_resume(cmd)
        elif cmd.command_type == "CONFIG_UPDATE":
            return self._cmd_config(cmd)
        elif cmd.command_type in ("BUY", "SELL"):
            return self._cmd_trade(cmd)
        elif cmd.command_type == "DEPOSIT":
            return self._cmd_deposit(cmd)
        elif cmd.command_type == "WITHDRAW":
            return self._cmd_withdraw(cmd)
        else:
            return {"error": f"Commande {cmd.command_type} non implémentée"}

    def _cmd_liquidate(self, cmd: Command) -> dict[str, Any]:
        port = self._get_port(cmd.portfolio_id)
        strategy_cls = STRATEGY_MAP.get(port.strategy_type, StrategyBase)
        strategy = strategy_cls(port.id)
        # On récupère les derniers prix connus
        from trading.core.models import MarketData
        latest = (
            self.db.query(MarketData.ticker, MarketData.price)
            .distinct(MarketData.ticker)
            .order_by(MarketData.ticker, MarketData.timestamp.desc())
            .all()
        )
        prices = {row.ticker: row.price for row in latest}
        trades = strategy.liquidate(self.db, prices)
        return {"liquidated": True, "trades": len(trades), "cash": port.cash_current}

    def _cmd_pause(self, cmd: Command) -> dict[str, Any]:
        port = self._get_port(cmd.portfolio_id)
        port.status = "paused"
        self.db.commit()
        return {"status": port.status}

    def _cmd_resume(self, cmd: Command) -> dict[str, Any]:
        port = self._get_port(cmd.portfolio_id)
        port.status = "active"
        self.db.commit()
        return {"status": port.status}

    def _cmd_config(self, cmd: Command) -> dict[str, Any]:
        port = self._get_port(cmd.portfolio_id)
        payload = json.loads(cmd.payload or "{}")
        current = json.loads(port.config_json or "{}")
        current.update(payload)
        port.config_json = json.dumps(current)
        self.db.commit()
        return {"config": current}

    def _cmd_trade(self, cmd: Command) -> dict[str, Any]:
        payload = json.loads(cmd.payload or "{}")
        ticker = payload["ticker"]
        qty = payload["quantity"]
        price = payload.get("price", 0)
        port = self._get_port(cmd.portfolio_id)
        strategy_cls = STRATEGY_MAP.get(port.strategy_type, StrategyBase)
        strategy = strategy_cls(port.id)
        if cmd.command_type == "BUY":
            trade = strategy.buy(self.db, ticker, qty, price)
        else:
            trade = strategy.sell(self.db, ticker, qty, price)
        return {"trade_id": trade.id, "action": trade.action}

    def _cmd_deposit(self, cmd: Command) -> dict[str, Any]:
        payload = json.loads(cmd.payload or "{}")
        amount = payload.get("amount", 0)
        port = self._get_port(cmd.portfolio_id)
        port.cash_current += amount
        port.cash_initial += amount
        self.db.commit()
        return {"deposit": amount, "cash": port.cash_current}

    def _cmd_withdraw(self, cmd: Command) -> dict[str, Any]:
        payload = json.loads(cmd.payload or "{}")
        amount = payload.get("amount", 0)
        port = self._get_port(cmd.portfolio_id)
        if port.cash_current < amount:
            raise ValueError("Cash insuffisant")
        port.cash_current -= amount
        self.db.commit()
        return {"withdraw": amount, "cash": port.cash_current}

    def _get_port(self, portfolio_id: str | None) -> Portfolio:
        if not portfolio_id:
            raise ValueError("portfolio_id requis")
        port = self.db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
        if not port:
            raise ValueError(f"Portefeuille {portfolio_id} introuvable")
        return port
