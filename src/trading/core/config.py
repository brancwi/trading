"""Configuration centralisée du trading engine."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Settings chargés depuis .env et/ou variables d'environnement.

    Environnments:
      - development : SQLite, modèles locaux, tests rapides
      - staging     : PostgreSQL, DeepSeek cloud, portefeuilles virtuels permanents
      - production  : PostgreSQL, DeepSeek cloud, argent réel
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Environment — doit être 'development', 'staging', ou 'production'
    # Peut être surchargé via TRADING_ENVIRONMENT ou ENVIRONMENT
    environment: str = Field(default="development", validation_alias="trading_environment")

    # Database
    database_url: str = f"sqlite:///{DATA_DIR}/trading.db"
    postgres_password: str = "changeme"
    # Staging / Production DB URL (override database_url when env != dev)
    staging_database_url: str = "postgresql://trading:changeme@localhost:5432/trading_staging"
    production_database_url: str = "postgresql://trading:changeme@localhost:5432/trading"
    # Dev override — force SQLite regardless of env var
    dev_database_url: str = f"sqlite:///{DATA_DIR}/trading_dev.db"

    # API Keys
    finnhub_api_key: str = ""
    alpha_vantage_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # API interne
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "dev-secret-change-me"

    # MCP Server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001

    # Prefect Server (port local du UI/API Prefect)
    prefect_port: int = 4200

    # FX — taux de change par défaut (fallback) si le service temps réel échoue
    fx_eur_usd_default: float = 1.08  # 1 EUR = 1.08 USD

    # Ports par environnement (créneau 10xxx pour staging/prod)
    # Dev utilise le créneau 8xxx par défaut (pas de conflit avec autres projets)
    staging_base_port: int = 10000
    production_base_port: int = 10010

    @property
    def resolved_api_port(self) -> int:
        if self.is_staging:
            return self.staging_base_port
        if self.is_production:
            return self.production_base_port
        return self.api_port

    @property
    def resolved_mcp_port(self) -> int:
        if self.is_staging:
            return self.staging_base_port + 1
        if self.is_production:
            return self.production_base_port + 1
        return self.mcp_port

    @property
    def resolved_prefect_port(self) -> int:
        if self.is_staging:
            return self.staging_base_port + 2
        if self.is_production:
            return self.production_base_port + 2
        return self.prefect_port

    @property
    def resolved_timescaledb_port(self) -> int:
        if self.is_staging:
            return self.staging_base_port + 3
        if self.is_production:
            return self.production_base_port + 3
        return 10003  # fallback

    @property
    def resolved_grafana_port(self) -> int:
        if self.is_staging:
            return self.staging_base_port + 4
        if self.is_production:
            return self.production_base_port + 4
        return 3002

    # ML / Sentiment Engine
    ml_device: str = "cuda"
    ml_model_finbert: str = "ProsusAI/finbert"
    ml_model_roberta: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
    ml_model_modern: str = "tabularisai/ModernFinBERT"
    ml_model_qwen: str = "NOSIBLE/financial-sentiment-v1.1-base"
    ml_enable_qwen: bool = True
    ml_enable_cloud_fallback: bool = False
    ml_cloud_provider: str = "openai"
    ml_cloud_model: str = "gpt-4o-mini"
    ml_lexical_override: bool = True
    ml_divergence_threshold: float = 0.3
    ml_signal_threshold: float = 0.5
    ml_confidence_threshold: float = 0.6

    # Decision LLM
    enable_decision_llm: bool = True  # Activé par défaut partout
    decision_llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    decision_llm_use_cloud: bool = False  # Déduit automatiquement de l'environnement
    deepseek_api_key: str = ""

    # Sentiment cloud fallback
    ml_enable_cloud_fallback: bool = False  # Déduit automatiquement de l'environnement

    @property
    def is_dev(self) -> bool:
        return self.environment == "development"

    @property
    def is_staging(self) -> bool:
        return self.environment == "staging"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def use_local_models(self) -> bool:
        """True en dev (modèles locaux), False en staging/prod (cloud)."""
        return self.is_dev

    @property
    def use_cloud_llm(self) -> bool:
        """True en staging/prod (DeepSeek), False en dev (local)."""
        return self.is_staging or self.is_production

    @property
    def resolved_database_url(self) -> str:
        """DB URL selon l'environnement."""
        if self.environment == "testing":
            return "sqlite:///:memory:"
        if self.is_dev:
            return self.dev_database_url
        if self.is_staging:
            return self.staging_database_url
        return self.production_database_url

    @property
    def resolved_monitoring_db_url(self) -> str:
        """Monitoring DB URL selon l'environnement."""
        if self.environment == "testing":
            return "sqlite:///:memory:"
        if self.is_dev:
            return f"sqlite:///{DATA_DIR}/monitoring_dev.db"
        if self.is_staging:
            return "postgresql://monitoring:monitoring_pass@localhost:10003/trading_monitoring_staging"
        return "postgresql://monitoring:monitoring_pass@localhost:10003/trading_monitoring"

    # Monitoring DB (TimescaleDB or SQLite fallback)
    monitoring_db_url: str = ""
    monitoring_log_llm_calls: bool = True
    monitoring_log_messages: bool = True
    monitoring_alert_cost_daily_usd: float = 5.0
    monitoring_alert_latency_ms: int = 5000

    # Pipeline
    pipeline_interval_minutes: int = 5
    market_open_hour: int = 6
    market_close_hour: int = 22
    market_days: str = "1,2,3,4,5"  # lundi=1 ... vendredi=5

    # Logging
    log_level: str = "INFO"

    @property
    def market_days_list(self) -> list[int]:
        return [int(d.strip()) for d in self.market_days.split(",")]


@lru_cache
def get_settings() -> Settings:
    return Settings()
