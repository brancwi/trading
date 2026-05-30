"""Point d'entrée FastAPI - Trading Engine API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trading.core.config import get_settings
from trading.core.database import init_db
from trading.api.routes import status, portfolios, strategies, decisions

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage."""
    init_db()
    yield


app = FastAPI(
    title="Trading Engine API",
    description="Système de trading modulaire V4 - Event-driven + Command Bus",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(status.router)
app.include_router(portfolios.router)
app.include_router(strategies.router)
app.include_router(decisions.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "4.0.0"}
