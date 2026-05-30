"""Configuration centralisée du trading engine."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class Settings(BaseSettings):
    """Settings chargés depuis .env et/ou variables d'environnement."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = f"sqlite:///{DATA_DIR}/trading.db"

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

    # ML / Sentiment Engine
    ml_device: str = "cuda"
    ml_model_finbert: str = "ProsusAI/finbert"
    ml_model_roberta: str = "mrm8488/distilroberta-finetuned-financial-news"
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
