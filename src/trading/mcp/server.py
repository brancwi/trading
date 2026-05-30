"""MCP Server unique — Trading Engine Data Access (SSE transport).

Expose tous les tools pour que Hermes interroge les données du trading engine
via le protocole MCP (Model Context Protocol).

Usage:
    python scripts/run_mcp_server.py
    # Hermes se connecte à http://localhost:8001/sse
"""

import json
import os
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from trading.core.config import get_settings

settings = get_settings()

# Connexion PostgreSQL (ou SQLite fallback)
_engine = create_engine(settings.database_url, pool_pre_ping=True)
_Session = sessionmaker(bind=_engine)

# ------------------------------------------------------------------
# Serveur MCP
# ------------------------------------------------------------------

mcp = FastMCP("trading-engine")


def _to_dict(rows):
    """Convertit des lignes SQLAlchemy rowproxy en list[dict]."""
    return [dict(r._mapping) for r in rows]


def _json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# ------------------------------------------------------------------
# Tools — Portfolios & Positions
# ------------------------------------------------------------------

@mcp.tool()
def list_portfolios() -> list[dict]:
    """Liste tous les portefeuilles avec leurs soldes et statuts."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT id, name, strategy_type, base_currency,
                       cash_initial, cash_current, reserved_cash,
                       status, created_at
                FROM portfolios
                ORDER BY created_at DESC
            """)
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


@mcp.tool()
def get_positions(portfolio_id: str) -> list[dict]:
    """Récupère les positions actuelles d'un portefeuille."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT ticker, quantity, avg_entry_price, current_price,
                       current_value, unrealized_pnl, unrealized_pnl_pct,
                       sector, opened_at
                FROM positions
                WHERE portfolio_id = :pid
                ORDER BY current_value DESC
            """),
            {"pid": portfolio_id},
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


@mcp.tool()
def get_portfolio_details(portfolio_id: str) -> dict:
    """Détails complets d'un portefeuille : info + positions + dernières trades."""
    with _Session() as db:
        portfolio = db.execute(
            text("SELECT * FROM portfolios WHERE id = :pid"),
            {"pid": portfolio_id},
        ).mappings().first()
        if portfolio is None:
            return {"error": "Portfolio not found"}

        positions = db.execute(
            text("SELECT * FROM positions WHERE portfolio_id = :pid"),
            {"pid": portfolio_id},
        ).mappings().all()

        trades = db.execute(
            text("SELECT * FROM trades WHERE portfolio_id = :pid ORDER BY executed_at DESC LIMIT 10"),
            {"pid": portfolio_id},
        ).mappings().all()

        return {
            "portfolio": dict(portfolio._mapping),
            "positions": _to_dict(positions),
            "recent_trades": _to_dict(trades),
        }


# ------------------------------------------------------------------
# Tools — Trades & History
# ------------------------------------------------------------------

@mcp.tool()
def get_trade_history(portfolio_id: str, limit: int = 50) -> list[dict]:
    """Historique des trades exécutés pour un portefeuille."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT ticker, action, quantity, price, amount, fees,
                       realized_pnl, executed_at
                FROM trades
                WHERE portfolio_id = :pid
                ORDER BY executed_at DESC
                LIMIT :limit
            """),
            {"pid": portfolio_id, "limit": limit},
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


@mcp.tool()
def get_balance_history(portfolio_id: str, limit: int = 50) -> list[dict]:
    """Historique du solde total d'un portefeuille (time-series)."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT timestamp, cash, positions_value, total_value,
                       total_pnl, total_pnl_pct, drawdown_pct
                FROM portfolio_history
                WHERE portfolio_id = :pid
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"pid": portfolio_id, "limit": limit},
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


# ------------------------------------------------------------------
# Tools — Signals & Sentiment
# ------------------------------------------------------------------

@mcp.tool()
def get_signals(consumed: bool | None = None, limit: int = 50) -> list[dict]:
    """Signaux générés par le sentiment engine."""
    with _Session() as db:
        query = """
            SELECT id, timestamp, ticker, action, sentiment, strength,
                   confidence, source, consumed
            FROM signals
        """
        params: dict = {"limit": limit}
        if consumed is not None:
            query += " WHERE consumed = :consumed"
            params["consumed"] = 1 if consumed else 0
        query += " ORDER BY timestamp DESC LIMIT :limit"

        rows = db.execute(text(query), params).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


@mcp.tool()
def get_sentiment_scores(ticker: str | None = None, limit: int = 50) -> list[dict]:
    """Scores sentiment multi-modèles (FinBERT, ModernFinBERT, Qwen, Cloud)."""
    with _Session() as db:
        query = """
            SELECT id, timestamp, ticker, finbert_score, roberta_score,
                   modern_score, qwen_score, cloud_score, lexical_score,
                   combined_score, confidence, divergence, anomaly_flag,
                   qwen_arbitrated, cloud_fallback_used,
                   input_tokens, output_tokens, estimated_cost_usd
            FROM sentiment_scores
        """
        params: dict = {"limit": limit}
        if ticker:
            query += " WHERE ticker = :ticker"
            params["ticker"] = ticker
        query += " ORDER BY timestamp DESC LIMIT :limit"

        rows = db.execute(text(query), params).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


# ------------------------------------------------------------------
# Tools — Market Data & News
# ------------------------------------------------------------------

@mcp.tool()
def get_market_data(ticker: str, limit: int = 50) -> list[dict]:
    """Données de marché OHLCV pour un ticker."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT timestamp, price, open_price, high, low,
                       change_pct, volume
                FROM market_data
                WHERE ticker = :ticker
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"ticker": ticker, "limit": limit},
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


@mcp.tool()
def get_news(ticker: str | None = None, limit: int = 50) -> list[dict]:
    """News financières récentes."""
    with _Session() as db:
        query = """
            SELECT id, timestamp, source, ticker, title, description, url, processed
            FROM news
        """
        params: dict = {"limit": limit}
        if ticker:
            query += " WHERE ticker = :ticker"
            params["ticker"] = ticker
        query += " ORDER BY timestamp DESC LIMIT :limit"

        rows = db.execute(text(query), params).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


# ------------------------------------------------------------------
# Tools — SQL Ad-hoc (read-only)
# ------------------------------------------------------------------

@mcp.tool()
def execute_sql_query(query: str) -> dict:
    """Exécute une requête SQL read-only (SELECT uniquement).

    Sécurité : les requêtes contenant DELETE, DROP, INSERT, UPDATE, ALTER
    sont rejetées.
    """
    forbidden = ["delete", "drop", "insert", "update", "alter", "truncate", "create"]
    lowered = query.lower()
    for kw in forbidden:
        if kw in lowered:
            return {"error": f"Requête interdite : mot-clé '{kw}' détecté. SELECT uniquement."}

    with _Session() as db:
        try:
            rows = db.execute(text(query)).mappings().all()
            return {
                "columns": list(rows[0].keys()) if rows else [],
                "rows": json.loads(json.dumps(_to_dict(rows), default=_json_serial)),
                "count": len(rows),
            }
        except Exception as e:
            return {"error": str(e)}


# ------------------------------------------------------------------
# Tools — Capital Movements
# ------------------------------------------------------------------

@mcp.tool()
def reserve_capital(portfolio_id: str, amount: float, reason: str = "") -> dict:
    """Met de côté du capital sur un portefeuille (soustrait du cash disponible)."""
    with _Session() as db:
        portfolio = db.execute(
            text("SELECT cash_current, reserved_cash FROM portfolios WHERE id = :pid"),
            {"pid": portfolio_id},
        ).mappings().first()
        if portfolio is None:
            return {"error": "Portfolio not found"}

        cash_current = portfolio["cash_current"]
        reserved = portfolio["reserved_cash"]
        available = cash_current - reserved

        if amount > available:
            return {
                "error": f"Montant trop élevé. Disponible: {available:.2f}, demandé: {amount:.2f}"
            }

        new_reserved = reserved + amount
        db.execute(
            text("UPDATE portfolios SET reserved_cash = :r WHERE id = :pid"),
            {"r": new_reserved, "pid": portfolio_id},
        )
        db.execute(
            text("""
                INSERT INTO capital_movements
                (portfolio_id, movement_type, amount, balance_after, reason, actor)
                VALUES (:pid, 'reserve', :amount, :balance, :reason, 'hermes')
            """),
            {
                "pid": portfolio_id,
                "amount": amount,
                "balance": cash_current - new_reserved,
                "reason": reason or "Mise de côté via MCP",
            },
        )
        db.commit()
        return {
            "status": "reserved",
            "portfolio_id": portfolio_id,
            "amount": amount,
            "reserved_cash": new_reserved,
            "cash_available": cash_current - new_reserved,
        }


@mcp.tool()
def release_capital(portfolio_id: str, amount: float) -> dict:
    """Libère du capital précédemment mis de côté."""
    with _Session() as db:
        portfolio = db.execute(
            text("SELECT cash_current, reserved_cash FROM portfolios WHERE id = :pid"),
            {"pid": portfolio_id},
        ).mappings().first()
        if portfolio is None:
            return {"error": "Portfolio not found"}

        reserved = portfolio["reserved_cash"]
        cash_current = portfolio["cash_current"]

        if amount > reserved:
            return {
                "error": f"Montant trop élevé. Réservé: {reserved:.2f}, demandé: {amount:.2f}"
            }

        new_reserved = reserved - amount
        db.execute(
            text("UPDATE portfolios SET reserved_cash = :r WHERE id = :pid"),
            {"r": new_reserved, "pid": portfolio_id},
        )
        db.execute(
            text("""
                INSERT INTO capital_movements
                (portfolio_id, movement_type, amount, balance_after, reason, actor)
                VALUES (:pid, 'release', :amount, :balance, 'Libération via MCP', 'hermes')
            """),
            {
                "pid": portfolio_id,
                "amount": amount,
                "balance": cash_current - new_reserved,
            },
        )
        db.commit()
        return {
            "status": "released",
            "portfolio_id": portfolio_id,
            "amount": amount,
            "reserved_cash": new_reserved,
            "cash_available": cash_current - new_reserved,
        }


@mcp.tool()
def get_capital_movements(portfolio_id: str, limit: int = 50) -> list[dict]:
    """Historique des mouvements de capital (réserve, libération, retrait)."""
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT timestamp, movement_type, amount, balance_after,
                       reason, actor
                FROM capital_movements
                WHERE portfolio_id = :pid
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            {"pid": portfolio_id, "limit": limit},
        ).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))


# ------------------------------------------------------------------
# Tools — Monitoring & Audit
# ------------------------------------------------------------------

@mcp.tool()
def get_token_usage(hours: int = 24) -> dict:
    """Consommation tokens et coût estimé sur les N dernières heures."""
    since = datetime.utcnow() - timedelta(hours=hours)
    with _Session() as db:
        rows = db.execute(
            text("""
                SELECT provider, model,
                       COUNT(*) as calls,
                       SUM(input_tokens) as total_input,
                       SUM(output_tokens) as total_output,
                       SUM(cost_usd) as total_cost
                FROM token_usage_log
                WHERE timestamp >= :since
                GROUP BY provider, model
            """),
            {"since": since},
        ).mappings().all()
        return {
            "period_hours": hours,
            "since": since.isoformat(),
            "breakdown": json.loads(json.dumps(_to_dict(rows), default=_json_serial)),
        }


@mcp.tool()
def get_audit_log(hours: int = 24, event_type: str | None = None) -> list[dict]:
    """Journal d'audit des événements système."""
    since = datetime.utcnow() - timedelta(hours=hours)
    with _Session() as db:
        query = """
            SELECT timestamp, event_type, entity_type, entity_id,
                   actor, severity, details_json
            FROM audit_log
            WHERE timestamp >= :since
        """
        params: dict = {"since": since}
        if event_type:
            query += " AND event_type = :event_type"
            params["event_type"] = event_type
        query += " ORDER BY timestamp DESC LIMIT 100"

        rows = db.execute(text(query), params).mappings().all()
        return json.loads(json.dumps(_to_dict(rows), default=_json_serial))
