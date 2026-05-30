"""Routes /strategies - contrôle des stratégies et paramètres."""

import json
from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from trading.api.dependencies import DbDep, AuthDep
from trading.core.models import Portfolio, Command
from trading.execution.commands import CommandProcessor

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("")
def list_strategies(db: Session = DbDep, _: str = AuthDep):
    """Liste les stratégies actives avec leur config."""
    ports = db.query(Portfolio).all()
    return [
        {
            "portfolio_id": p.id,
            "strategy_type": p.strategy_type,
            "status": p.status,
            "config": json.loads(p.config_json or "{}"),
        }
        for p in ports
    ]


@router.post("/{portfolio_id}/config")
def update_config(portfolio_id: str, payload: dict, db: Session = DbDep, _: str = AuthDep):
    """Met à jour la configuration dynamique d'une stratégie."""
    port = db.query(Portfolio).filter(Portfolio.id == portfolio_id).first()
    if not port:
        raise HTTPException(status_code=404, detail="Portefeuille introuvable")
    cmd = Command(
        command_type="CONFIG_UPDATE",
        portfolio_id=portfolio_id,
        payload=json.dumps(payload),
        requested_by="hermes",
    )
    db.add(cmd)
    db.commit()
    processor = CommandProcessor(db)
    processor.process_pending()
    return {"status": "updated", "config": payload}
