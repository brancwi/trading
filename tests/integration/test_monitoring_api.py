"""Integration tests for the new monitoring API endpoints."""

import pytest
from fastapi.testclient import TestClient

from trading.api.main import app

client = TestClient(app)

API_KEY = "dev-secret-change-me"
HEADERS = {"X-API-Key": API_KEY}


class TestMonitoringEndpoints:
    def test_llm_calls_endpoint(self):
        resp = client.get("/monitoring/llm-calls?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "llm_calls" in data
        assert isinstance(data["llm_calls"], list)

    def test_llm_calls_summary_endpoint(self):
        resp = client.get("/monitoring/llm-calls/summary?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "calls" in data
        assert "cost_usd" in data

    def test_llm_calls_timeseries_endpoint(self):
        resp = client.get("/monitoring/llm-calls/timeseries?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "interval" in data
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_messages_endpoint(self):
        resp = client.get("/monitoring/messages?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "messages" in data
        assert isinstance(data["messages"], list)

    def test_messages_channels_endpoint(self):
        resp = client.get("/monitoring/messages/channels", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "channels" in data
        assert isinstance(data["channels"], list)

    def test_performance_endpoint(self):
        resp = client.get("/monitoring/performance?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "period_hours" in data
        assert "metrics" in data

    def test_existing_token_usage_endpoint(self):
        resp = client.get("/monitoring/token-usage?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "period_hours" in data
        assert "usage" in data

    def test_existing_summary_endpoint(self):
        resp = client.get("/monitoring/summary?hours=1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "period_hours" in data
        assert "token_usage" in data
