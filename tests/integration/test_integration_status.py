"""Tests d'intégration — status et diagnostic."""

import pytest

pytestmark = pytest.mark.integration


class TestStatus:
    def test_get_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_get_status(self, client, test_portfolio):
        resp = client.get("/status", headers={"X-API-Key": "dev-secret-change-me"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline"] == "running"
        assert "test-e2e" in data["portfolios"]

    def test_get_diagnostic(self, client, test_portfolio):
        resp = client.get(
            "/status/diagnostic", headers={"X-API-Key": "dev-secret-change-me"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "environment" in data
        assert "database" in data
        assert data["database"]["type"] in ("postgresql", "sqlite")
        assert "url_masked" in data["database"]
        assert "***" in data["database"]["url_masked"] or "sqlite" in data["database"]["url_masked"]
        assert "services" in data
        assert "api" in data["services"]
        assert "config_summary" in data
        assert "ml_device" in data["config_summary"]
        assert "recent_errors_last_1h" in data
        assert "generated_at" in data
