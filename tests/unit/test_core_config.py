"""Unit tests for core configuration."""

import pytest

from trading.core.config import Settings, get_settings


@pytest.mark.unit
def test_get_settings_returns_settings_instance():
    settings = get_settings()
    assert isinstance(settings, Settings)


@pytest.mark.unit
def test_settings_environment_defaults_to_development(monkeypatch):
    monkeypatch.delenv("TRADING_ENVIRONMENT", raising=False)
    settings = Settings()
    assert settings.environment == "development"


@pytest.mark.unit
def test_settings_market_days_list():
    settings = Settings()
    assert settings.market_days_list == [1, 2, 3, 4, 5]


@pytest.mark.unit
def test_settings_env_var_override_api_port(monkeypatch):
    monkeypatch.setenv("API_PORT", "9999")
    settings = Settings()
    assert settings.api_port == 9999


@pytest.mark.unit
def test_settings_env_var_override_database_url(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///custom.db")
    settings = Settings()
    assert settings.database_url == "sqlite:///custom.db"
