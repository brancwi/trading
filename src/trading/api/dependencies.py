"""Dépendances FastAPI : auth, DB, etc."""

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from trading.core.config import get_settings
from trading.core.database import get_db

settings = get_settings()


async def verify_api_key(x_api_key: str = Header(...)) -> str:
    """Vérifie la clé API interne."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Clé API invalide")
    return x_api_key


# Alias pour injection propre
DbDep = Depends(get_db)
AuthDep = Depends(verify_api_key)
