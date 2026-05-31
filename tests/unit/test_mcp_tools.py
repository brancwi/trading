"""Tests unitaires pour les outils MCP."""

import pytest

from trading.core import database as db_module
from trading.core.models import Portfolio
from trading.mcp.server import (
    execute_sql_query,
    get_capital_movements,
    get_portfolio_details,
    get_positions,
    list_portfolios,
    release_capital,
    reserve_capital,
)


pytestmark = pytest.mark.unit


class TestMCPTools:
    def test_list_portfolios(self, test_portfolio):
        portfolios = list_portfolios()
        ids = [p["id"] for p in portfolios]
        assert "test-e2e" in ids

    def test_get_portfolio_details(self, test_portfolio):
        details = get_portfolio_details("test-e2e")
        assert "portfolio" in details
        assert details["portfolio"]["id"] == "test-e2e"

    def test_get_positions_empty(self, test_portfolio):
        positions = get_positions("test-e2e")
        assert positions == []

    def test_reserve_capital(self, test_portfolio):
        result = reserve_capital("test-e2e", 500.0)
        assert result["status"] == "reserved"
        assert result["reserved_cash"] == 500.0
        assert result["cash_available"] == 9500.0

        with db_module.SessionLocal() as db:
            portfolio = db.query(Portfolio).filter_by(id="test-e2e").first()
            assert portfolio.reserved_cash == 500.0
            assert portfolio.cash_available == 9500.0

    def test_reserve_capital_error_amount_too_high(self, test_portfolio):
        result = reserve_capital("test-e2e", 999999.0)
        assert "error" in result

    def test_release_capital(self, test_portfolio):
        reserve_capital("test-e2e", 500.0)
        result = release_capital("test-e2e", 200.0)
        assert result["status"] == "released"
        assert result["reserved_cash"] == 300.0
        assert result["cash_available"] == 9700.0

        with db_module.SessionLocal() as db:
            portfolio = db.query(Portfolio).filter_by(id="test-e2e").first()
            assert portfolio.reserved_cash == 300.0
            assert portfolio.cash_available == 9700.0

    def test_get_capital_movements(self, test_portfolio):
        reserve_capital("test-e2e", 500.0)
        release_capital("test-e2e", 200.0)
        movements = get_capital_movements("test-e2e", limit=10)
        assert len(movements) == 2
        types = {m["movement_type"] for m in movements}
        assert "reserve" in types
        assert "release" in types

    def test_execute_sql_query_select(self, test_portfolio):
        result = execute_sql_query("SELECT * FROM portfolios")
        assert "error" not in result
        assert "rows" in result
        assert result["count"] >= 1

    def test_execute_sql_query_delete_forbidden(self):
        result = execute_sql_query("DELETE FROM portfolios")
        assert "error" in result

    def test_execute_sql_query_drop_forbidden(self):
        result = execute_sql_query("DROP TABLE portfolios")
        assert "error" in result
"""Unit tests for MCP tools."""

import pytest

from trading.mcp.server import (
    list_portfolios,
    get_portfolio_details,
    get_positions,
    reserve_capital,
    release_capital,
    get_capital_movements,
    execute_sql_query,
    get_system_status,
)

pytestmark = pytest.mark.unit


class TestMCPTools:
    def test_list_portfolios(self, test_portfolio):
        portfolios = list_portfolios()
        ids = [p["id"] for p in portfolios]
        assert "test-e2e" in ids

    def test_get_portfolio_details(self, test_portfolio):
        details = get_portfolio_details("test-e2e")
        assert "portfolio" in details
        assert details["portfolio"]["id"] == "test-e2e"
        assert details["positions"] == []
        assert details["recent_trades"] == []

    def test_get_positions_empty(self, test_portfolio):
        positions = get_positions("test-e2e")
        assert positions == []

    def test_reserve_capital(self, test_portfolio):
        result = reserve_capital("test-e2e", 500.0, "Test reserve")
        assert result["status"] == "reserved"
        assert result["reserved_cash"] == 500.0
        assert result["cash_available"] == 9500.0

    def test_reserve_capital_excessive(self, test_portfolio):
        result = reserve_capital("test-e2e", 999999.0)
        assert "error" in result

    def test_release_capital(self, test_portfolio):
        reserve_capital("test-e2e", 500.0)
        result = release_capital("test-e2e", 200.0)
        assert result["status"] == "released"
        assert result["reserved_cash"] == 300.0
        assert result["cash_available"] == 9700.0

    def test_get_capital_movements(self, test_portfolio):
        reserve_capital("test-e2e", 500.0, "Test reserve")
        release_capital("test-e2e", 200.0)
        movements = get_capital_movements("test-e2e", limit=10)
        assert len(movements) == 2
        types = {m["movement_type"] for m in movements}
        assert "reserve" in types
        assert "release" in types

    def test_execute_sql_query_select(self, test_portfolio):
        result = execute_sql_query("SELECT * FROM portfolios")
        assert "error" not in result
        assert "rows" in result
        assert result["count"] >= 1

    def test_execute_sql_query_delete_forbidden(self):
        result = execute_sql_query("DELETE FROM portfolios")
        assert "error" in result

    def test_execute_sql_query_drop_forbidden(self):
        result = execute_sql_query("DROP TABLE portfolios")
        assert "error" in result


class TestMCPSystemStatus:
    def test_get_system_status_structure(self):
        status = get_system_status()
        assert "environment" in status
        assert "database" in status
        assert "type" in status["database"]
        assert status["database"]["type"] in ("postgresql", "sqlite")
        assert "url_masked" in status["database"]
        assert "services" in status
        assert "api" in status["services"]
        assert "config_summary" in status
        assert "recent_errors_last_1h" in status
        assert "generated_at" in status

    def test_get_system_status_db_version_present(self):
        status = get_system_status()
        assert status["database"]["version"] is not None
        assert len(str(status["database"]["version"])) > 0

    def test_get_system_status_url_masked(self):
        status = get_system_status()
        url = status["database"]["url_masked"]
        assert "sqlite" in url or "***" in url or "postgresql" in url
