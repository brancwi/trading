"""Database layer for the monitoring subsystem (TimescaleDB or SQLite fallback).

Supports:
  - TimescaleDB (production) — hypertables, compression, retention
  - SQLite (testing / local fallback) — zero-config, file-based
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, declarative_base

from trading.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

# En dev  → SQLite (local, rapide, isolated)
# En staging / prod → PostgreSQL TimescaleDB (résilient)
MONITORING_DB_URL = getattr(settings, "monitoring_db_url", "")
if not MONITORING_DB_URL:
    MONITORING_DB_URL = settings.resolved_monitoring_db_url

logger.info("[MonitoringDB] Environment=%s | URL=%s",
            settings.environment, MONITORING_DB_URL.split("/")[-1] if "/" in MONITORING_DB_URL else MONITORING_DB_URL)

_IS_SQLITE = MONITORING_DB_URL.startswith("sqlite")


_engine_kwargs = {"pool_pre_ping": True}
if _IS_SQLITE:
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    if ":memory:" in MONITORING_DB_URL:
        from sqlalchemy.pool import StaticPool
        _engine_kwargs["poolclass"] = StaticPool

engine = create_engine(
    MONITORING_DB_URL,
    **_engine_kwargs,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
MonitoringBase = declarative_base()


def configure_database(db_url: str) -> None:
    """Reconfigure l'engine et SessionLocal (utile pour les tests).
    
    NOTE: MonitoringBase n'est PAS recréé pour préserver les modèles déjà importés.
    """
    global engine, SessionLocal, MONITORING_DB_URL, _IS_SQLITE, IS_TIMESCALEDB
    MONITORING_DB_URL = db_url
    _IS_SQLITE = db_url.startswith("sqlite")
    kwargs = {"pool_pre_ping": True}
    if _IS_SQLITE:
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in db_url:
            from sqlalchemy.pool import StaticPool
            kwargs["poolclass"] = StaticPool
    engine = create_engine(db_url, **kwargs)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    IS_TIMESCALEDB = _is_timescaledb()


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite monitoring DB."""
    if _IS_SQLITE:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ---------------------------------------------------------------------------
# Hypertable helper
# ---------------------------------------------------------------------------


def _is_timescaledb() -> bool:
    """Detect whether the connected DB is actually TimescaleDB."""
    if _IS_SQLITE:
        return False
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='timescaledb'"))
            return result.scalar() == 1
    except Exception:
        return False


IS_TIMESCALEDB = _is_timescaledb()


def create_hypertable(table_name: str, time_column: str = "timestamp") -> None:
    """Convert a regular table into a TimescaleDB hypertable (no-op on SQLite)."""
    if _IS_SQLITE or not IS_TIMESCALEDB:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', "
                    f"if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
            conn.commit()
            logger.info("[Monitoring] Hypertable created: %s", table_name)
    except Exception as e:
        logger.warning("[Monitoring] Hypertable creation skipped for %s: %s", table_name, e)


def setup_compression(table_name: str, after_interval: str = "7 days", segmentby: str = "") -> None:
    """Enable compression on a hypertable (no-op on SQLite).

    *segmentby* is a comma-separated list of columns to use for segment-by
    compression (empty = no segment-by).
    """
    if _IS_SQLITE or not IS_TIMESCALEDB:
        return
    try:
        with engine.connect() as conn:
            if segmentby:
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} SET (timescaledb.compress, "
                        f"timescaledb.compress_segmentby = '{segmentby}')"
                    )
                )
            else:
                conn.execute(
                    text(f"ALTER TABLE {table_name} SET (timescaledb.compress)")
                )
            conn.execute(
                text(
                    f"SELECT add_compression_policy('{table_name}', INTERVAL '{after_interval}')"
                )
            )
            conn.commit()
    except Exception as e:
        logger.warning("[Monitoring] Compression setup skipped for %s: %s", table_name, e)


def setup_retention(table_name: str, drop_after: str = "30 days") -> None:
    """Enable data retention (no-op on SQLite)."""
    if _IS_SQLITE or not IS_TIMESCALEDB:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"SELECT add_retention_policy('{table_name}', INTERVAL '{drop_after}')"
                )
            )
            conn.commit()
    except Exception as e:
        logger.warning("[Monitoring] Retention setup skipped for %s: %s", table_name, e)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@contextmanager
def monitoring_session() -> Generator:
    """Auto-commit session for monitoring DB."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


def _drop_pk_for_hypertable(table_name: str) -> None:
    """Drop the primary key constraint so TimescaleDB can create a hypertable.

    TimescaleDB requires that the time partitioning column be included in
    any primary key / unique index.  Since our ORM models use a simple
    ``id`` PK for SQLite compatibility, we drop it before converting the
    table to a hypertable (no-op on SQLite).
    """
    if _IS_SQLITE or not IS_TIMESCALEDB:
        return
    try:
        with engine.connect() as conn:
            conn.execute(
                text(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey")
            )
            conn.commit()
            logger.info("[Monitoring] Dropped PK constraint on %s", table_name)
    except Exception as e:
        logger.warning("[Monitoring] Could not drop PK on %s: %s", table_name, e)


def init_monitoring_db() -> None:
    """Create tables + hypertables + policies."""
    from trading.monitoring.models import LLMCallLog, MessageLog, PerformanceSnapshot

    MonitoringBase.metadata.create_all(bind=engine)

    if IS_TIMESCALEDB:
        # Remove simple id-only PKs so TimescaleDB can partition by timestamp
        _drop_pk_for_hypertable("llm_call_log")
        _drop_pk_for_hypertable("message_log")
        _drop_pk_for_hypertable("performance_snapshot")

        create_hypertable("llm_call_log", "timestamp")
        create_hypertable("message_log", "timestamp")
        create_hypertable("performance_snapshot", "timestamp")

        setup_compression("llm_call_log", "7 days", segmentby="model,provider")
        setup_compression("message_log", "3 days", segmentby="channel,source")
        setup_compression("performance_snapshot", "1 day", segmentby="metric_name")

        setup_retention("llm_call_log", "30 days")
        setup_retention("message_log", "14 days")
        setup_retention("performance_snapshot", "7 days")

        logger.info("[Monitoring] TimescaleDB initialized with hypertables")
    else:
        logger.info("[Monitoring] SQLite initialized (no hypertables)")
