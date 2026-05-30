"""Routes /decisions - injection de décisions manuelles par Hermes."""

import json
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from trading.api.dependencies import DbDep, AuthDep
from trading.core.models import Signal, Command, Trade, DecisionCreate
from trading.execution.commands import CommandProcessor

router = APIRouter(prefix="/decisions", tags=["decisions"])


@router.post("")
def inject_decision(decision: DecisionCreate, db: Session = DbDep, _: str = AuthDep):
    """
    Hermes injecte une décision manuelle ou un override IA.
    Crée un signal + une commande de trade en file.
    """
    # 1. Créer le signal
    signal = Signal(
        ticker=decision.ticker,
        action=decision.action.value,
        sentiment=decision.confidence * (1 if "BUY" in decision.action.value else -1),
        strength=decision.confidence,
        confidence=decision.confidence,
        source="hermes_decision",
        price_at_signal=decision.amount,
        expires_at=datetime.utcnow() + timedelta(hours=1),
    )
    db.add(signal)
    db.commit()
    db.refresh(signal)

    # 2. Créer la commande de trade
    cmd = Command(
        command_type=decision.action.value,
        portfolio_id=decision.portfolio_id,
        payload=json.dumps({
            "ticker": decision.ticker,
            "quantity": decision.amount or 0,
            "price": 0,  # sera résolu au moment de l'exécution
            "reason": decision.reason,
            "signal_id": signal.id,
        }),
        requested_by="hermes",
    )
    db.add(cmd)
    db.commit()

    return {
        "signal_id": signal.id,
        "command_id": cmd.id,
        "status": "queued",
    }


@router.get("/pending")
def list_pending_decisions(db: Session = DbDep, _: str = AuthDep):
    """Liste les commandes de trade en attente."""
    return (
        db.query(Command)
        .filter(Command.command_type.in_(["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]))
        .order_by(Command.created_at.desc())
        .limit(50)
        .all()
    )
