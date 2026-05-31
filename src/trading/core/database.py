"""Couche d'accès base de données - SQLAlchemy + utilitaires."""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

from trading.core.config import get_settings

settings = get_settings()

# Engine adaptatif SQLite / PostgreSQL
# En dev  → SQLite (local, rapide, isolated)
# En staging / prod → PostgreSQL (résilient, partagé)
_db_url = settings.resolved_database_url
_is_sqlite = _db_url.startswith("sqlite")

logger = logging.getLogger(__name__)
logger.info("[DB] Environment=%s | URL=%s | Type=%s",
            settings.environment, _db_url.split("@")[-1] if "@" in _db_url else _db_url.split("/")[-1],
            "sqlite" if _is_sqlite else "postgresql")

_engine_kwargs = {"echo": False, "pool_pre_ping": True}
if _is_sqlite:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}


def _create_engine_and_session(db_url: str):
    """Crée un engine et sessionmaker pour une URL donnée."""
    is_sqlite = db_url.startswith("sqlite")
    is_memory = is_sqlite and ":memory:" in db_url
    kwargs = {"echo": False, "pool_pre_ping": True}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    if is_memory:
        kwargs["poolclass"] = StaticPool
    eng = create_engine(db_url, **kwargs)
    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    sess = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return eng, sess


_engine, _Session = _create_engine_and_session(_db_url)
engine = _engine
SessionLocal = _Session
Base = declarative_base()


def configure_database(db_url: str) -> None:
    """Reconfigure l'engine et SessionLocal (utile pour les tests)."""
    global engine, SessionLocal
    engine, SessionLocal = _create_engine_and_session(db_url)


def init_db() -> None:
    """Crée toutes les tables définies dans les modèles ORM."""
    # Import explicite pour s'assurer que tous les modèles sont enregistrés dans Base.metadata
    import trading.core.models  # noqa: F401
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dépendance FastAPI pour obtenir une session DB."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager pour les transactions hors FastAPI."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
