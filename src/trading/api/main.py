"""Point d'entrée FastAPI - Trading Engine API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trading.core.config import get_settings
from trading.core.database import init_db
from trading.monitoring.database import init_monitoring_db
from trading.monitoring.message_logger import MessageLogger
from trading.api.routes import status, portfolios, strategies, decisions, monitoring

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisation au démarrage."""
    init_db()
    init_monitoring_db()
    yield


app = FastAPI(
    title="Trading Engine API",
    description="Système de trading modulaire V4 - Event-driven + Command Bus",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request, call_next):
    """Log every incoming API request to the monitoring DB."""
    from time import perf_counter
    start = perf_counter()
    response = await call_next(request)
    duration_ms = round((perf_counter() - start) * 1000, 2)
    MessageLogger.log(
        channel="api_rest",
        source=f"{request.method} {request.url.path}",
        metadata={
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client_host": request.client.host if request.client else None,
        },
    )
    return response

app.include_router(status.router)
app.include_router(portfolios.router)
app.include_router(strategies.router)
app.include_router(decisions.router)
app.include_router(monitoring.router)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
