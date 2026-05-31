"""Tests d'intégration — routes FastAPI avec TestClient."""

import pytest

pytestmark = pytest.mark.integration

API_KEY = "dev-secret-change-me"
AUTH_HEADERS = {"X-API-Key": API_KEY}


class TestHealth:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "1.0.0"}


class TestPortfolios:
    def test_list_portfolios_with_auth(self, client, test_portfolio):
        response = client.get("/portfolios", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        ids = [p["id"] for p in data]
        assert "test-e2e" in ids

    def test_get_portfolio_with_auth(self, client, test_portfolio):
        response = client.get("/portfolios/test-e2e", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-e2e"
        assert data["status"] == "active"

    def test_get_portfolio_invalid_key(self, client, test_portfolio):
        response = client.get(
            "/portfolios/test-e2e", headers={"X-API-Key": "bad-key"}
        )
        assert response.status_code == 403

    def test_get_portfolios_summary(self, client, test_portfolio):
        response = client.get("/portfolios/summary", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        item = next((p for p in data if p["id"] == "test-e2e"), None)
        assert item is not None
        assert "cash_current" in item
        assert "positions_value" in item
        assert "total_value" in item


class TestDecisions:
    def test_post_decision_creates_signal_and_command(self, client, test_portfolio):
        payload = {
            "action": "BUY",
            "ticker": "AAPL",
            "portfolio_id": "test-e2e",
            "confidence": 0.9,
        }
        response = client.post("/decisions", json=payload, headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "signal_id" in data
        assert "command_id" in data
        assert data["status"] == "queued"

    def test_get_decisions_pending(self, client, test_portfolio):
        # Créer une décision pour s'assurer qu'au moins une commande de type trade existe
        payload = {
            "action": "BUY",
            "ticker": "AAPL",
            "portfolio_id": "test-e2e",
            "confidence": 0.9,
        }
        client.post("/decisions", json=payload, headers=AUTH_HEADERS)

        response = client.get("/decisions/pending", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        actions = [cmd["command_type"] for cmd in data]
        assert "BUY" in actions


class TestPortfolioLifecycle:
    def test_pause_portfolio(self, client, test_portfolio):
        response = client.post("/portfolios/test-e2e/pause", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

        response = client.get("/portfolios/test-e2e", headers=AUTH_HEADERS)
        assert response.json()["status"] == "paused"

    def test_resume_portfolio(self, client, test_portfolio):
        # Mettre d'abord en pause
        client.post("/portfolios/test-e2e/pause", headers=AUTH_HEADERS)

        response = client.post("/portfolios/test-e2e/resume", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"

        response = client.get("/portfolios/test-e2e", headers=AUTH_HEADERS)
        assert response.json()["status"] == "active"


class TestStatus:
    def test_get_status(self, client, test_portfolio):
        response = client.get("/status", headers=AUTH_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline"] == "running"
        assert "portfolios" in data
        assert "test-e2e" in data["portfolios"]
        assert "pending_commands" in data
        assert "unread_signals" in data
