"""Fixtures pytest globales — DB in-memory, rollback auto, portfolio test."""

import pytest
import tempfile
from pathlib import Path

# ── 0. Isoler le SignalModel pour les tests (pas de modèle entraîné persistant) ──
import trading.ml.signal_model as _sm

_tmp_dir = tempfile.mkdtemp()
_sm._MODEL_PATH = Path(_tmp_dir) / "signal_model.joblib"
_sm._META_PATH = Path(_tmp_dir) / "signal_model.meta.json"

# ── 1. Configure SQLite in-memory BEFORE any model or route import ──
from trading.core import database as db_module

db_module.configure_database("sqlite:///:memory:")
db_module.init_db()

# Also init monitoring DB in-memory for tests
from trading.monitoring import database as monitoring_db_module
monitoring_db_module.configure_database("sqlite:///:memory:")
monitoring_db_module.init_monitoring_db()

# ── 2. Safe imports (engine is now in-memory) ──
from fastapi.testclient import TestClient
from trading.api.main import app
from trading.core.models import Portfolio, Position, Trade, Signal, CapitalMovement, PortfolioHistory


@pytest.fixture(scope="session")
def client():
    """TestClient FastAPI avec la DB in-memory partagée."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def db():
    """Session SQLAlchemy avec rollback automatique après chaque test."""
    session = db_module.SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_portfolio(db):
    """Portfolio dédié aux tests — jamais simulation/rotation/ninja."""
    port = Portfolio(
        id="test-e2e",
        name="Test E2E Portfolio",
        strategy_type="simulation",
        base_currency="USD",
        cash_initial=10000.0,
        cash_current=10000.0,
        reserved_cash=0.0,
        max_trade_amount=500.0,
        fee_per_order=1.0,
        status="active",
        config_json='{"sentiment_threshold": 0.5, "cash_min": 100}',
    )
    db.add(port)
    db.commit()
    db.refresh(port)
    yield port
    # Cleanup explicite pour éviter les fuites entre tests
    db.query(Trade).filter(Trade.portfolio_id == port.id).delete()
    db.query(Position).filter(Position.portfolio_id == port.id).delete()
    db.query(CapitalMovement).filter(CapitalMovement.portfolio_id == port.id).delete()
    db.query(PortfolioHistory).filter(PortfolioHistory.portfolio_id == port.id).delete()
    db.query(Signal).delete()
    db.delete(port)
    db.commit()
