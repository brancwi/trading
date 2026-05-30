"""Migration des données SQLite existantes vers PostgreSQL.

Usage:
    export DATABASE_URL=postgresql://trading:changeme@localhost:5432/trading
    python scripts/migrate_sqlite_to_postgres.py
"""

import os
import shutil
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# SQLite source
SQLITE_PATH = Path(__file__).resolve().parent.parent / "data" / "trading.db"
SQLITE_URL = f"sqlite:///{SQLITE_PATH}"

# PostgreSQL cible (depuis env ou défaut)
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://trading:changeme@localhost:5432/trading")


def migrate():
    print(f"Source SQLite : {SQLITE_URL}")
    print(f"Cible PostgreSQL : {POSTGRES_URL}")

    # Engines
    sqlite_engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
    pg_engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

    SQLiteSession = sessionmaker(bind=sqlite_engine)
    PostgresSession = sessionmaker(bind=pg_engine)

    # Liste des tables à migrer (ordre pour respecter les FK)
    tables = [
        "portfolios",
        "news",
        "market_data",
        "sentiment_scores",
        "signals",
        "positions",
        "trades",
        "portfolio_history",
        "commands",
        "alerts",
        "monitoring_metrics",
        "audit_log",
        "token_usage_log",
        "capital_movements",
    ]

    # Créer les tables dans Postgres (si elles n'existent pas)
    print("\nCréation du schéma PostgreSQL...")
    from trading.core.database import init_db, engine as _pg_engine

    # Forcer l'import des modèles pour que Base.metadata connaisse toutes les tables
    import trading.core.models  # noqa: F401

    init_db()
    print("Schéma créé.")

    # Vider les tables existantes pour éviter les doublons
    print("Nettoyage des tables PostgreSQL...")
    with PostgresSession() as dst:
        for table in reversed(tables):
            try:
                dst.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            except Exception:
                pass  # table inexistante, ignorer
        dst.commit()
    print("Tables nettoyées.")

    counts = {}
    with SQLiteSession() as src, PostgresSession() as dst:
        for table in tables:
            # Vérifier si la table existe dans SQLite
            result = src.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
                {"name": table},
            )
            if result.scalar() is None:
                print(f"  {table}: table inexistante dans SQLite — ignoré")
                counts[table] = {"source": 0, "target": 0}
                continue

            # Compter source
            result = src.execute(text(f"SELECT COUNT(*) FROM {table}"))
            src_count = result.scalar()
            counts[table] = {"source": src_count, "target": 0}

            if src_count == 0:
                print(f"  {table}: 0 ligne — ignoré")
                continue

            # Lire depuis SQLite
            rows = src.execute(text(f"SELECT * FROM {table}")).mappings().all()
            if not rows:
                print(f"  {table}: 0 ligne — ignoré")
                continue

            # Construire INSERT
            columns = list(rows[0].keys())
            col_str = ", ".join(columns)
            placeholders = ", ".join([f":{c}" for c in columns])

            # Batch insert
            batch_size = 1000
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                dst.execute(
                    text(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"),
                    [dict(r) for r in batch],
                )
            dst.commit()

            # Vérifier compte cible
            result = dst.execute(text(f"SELECT COUNT(*) FROM {table}"))
            tgt_count = result.scalar()
            counts[table]["target"] = tgt_count

            status = "✅" if src_count == tgt_count else "❌"
            print(f"  {table}: {src_count} → {tgt_count} {status}")

    # Résumé
    total_src = sum(c["source"] for c in counts.values())
    total_tgt = sum(c["target"] for c in counts.values())
    print(f"\n{'='*50}")
    print(f"Total lignes source : {total_src}")
    print(f"Total lignes cible  : {total_tgt}")
    print(f"Migration {'✅ réussie' if total_src == total_tgt else '❌ ÉCHEC'}")

    # Backup SQLite
    backup_path = SQLITE_PATH.with_suffix(".db.bak")
    shutil.move(SQLITE_PATH, backup_path)
    print(f"\nSQLite déplacé vers : {backup_path}")


if __name__ == "__main__":
    migrate()
