"""Modèles SQLAlchemy (tables) et Pydantic (validation/API)."""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, CheckConstraint
from sqlalchemy.orm import relationship

from trading.core.database import Base
from pydantic import BaseModel, Field, ConfigDict


# =====================================================================
# SQLAlchemy ORM
# =====================================================================

class News(Base):
    __tablename__ = "news"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String, nullable=False)
    ticker = Column(String, nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    url = Column(String)
    fetched_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Integer, default=0)


class MarketData(Base):
    __tablename__ = "market_data"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ticker = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    open_price = Column(Float)
    high = Column(Float)
    low = Column(Float)
    change_pct = Column(Float)
    volume = Column(Integer)
    source = Column(String, default="finnhub")


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    news_id = Column(Integer, ForeignKey("news.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ticker = Column(String, nullable=False, index=True)
    # Legacy — gardé pour rétro-compatibilité
    finbert_score = Column(Float)
    roberta_score = Column(Float)
    # Nouveaux scores multi-modèles v2
    modern_score = Column(Float)
    qwen_score = Column(Float)
    cloud_score = Column(Float)
    lexical_score = Column(Float)
    lexical_rule = Column(String)
    # Score final et métadonnées
    combined_score = Column(Float, nullable=False)
    confidence = Column(Float)
    divergence = Column(Float)
    keywords = Column(Text)
    anomaly_flag = Column(Integer, default=0)
    qwen_arbitrated = Column(Integer, default=0)
    cloud_fallback_used = Column(Integer, default=0)


class Signal(Base):
    __tablename__ = "signals"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ticker = Column(String, nullable=False, index=True)
    action = Column(String, CheckConstraint("action IN ('BUY','SELL','HOLD','STRONG_BUY','STRONG_SELL')"))
    sentiment = Column(Float, nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    source = Column(String, default="ml_pipeline")
    price_at_signal = Column(Float)
    expires_at = Column(DateTime)
    consumed = Column(Integer, default=0)


class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    strategy_type = Column(String, nullable=False)
    base_currency = Column(String, default="USD")
    cash_initial = Column(Float, nullable=False)
    cash_current = Column(Float, nullable=False)
    max_trade_amount = Column(Float)
    fee_per_order = Column(Float, default=1.0)
    status = Column(String, default="active")
    config_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    positions = relationship("Position", back_populates="portfolio", cascade="all, delete-orphan")
    trades = relationship("Trade", back_populates="portfolio")


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=False, index=True)
    quantity = Column(Float, nullable=False)
    avg_entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    current_value = Column(Float)
    unrealized_pnl = Column(Float, default=0)
    unrealized_pnl_pct = Column(Float, default=0)
    sector = Column(String)
    opened_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")


class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=False)
    action = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    fees = Column(Float, nullable=False)
    realized_pnl = Column(Float)
    signal_id = Column(Integer, ForeignKey("signals.id"))
    strategy_type = Column(String)
    executed_at = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="trades")


class PortfolioHistory(Base):
    __tablename__ = "portfolio_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cash = Column(Float, nullable=False)
    positions_value = Column(Float, nullable=False)
    total_value = Column(Float, nullable=False)
    total_pnl = Column(Float)
    total_pnl_pct = Column(Float)
    drawdown_pct = Column(Float)


class Command(Base):
    __tablename__ = "commands"
    id = Column(Integer, primary_key=True, autoincrement=True)
    command_type = Column(String, nullable=False)
    portfolio_id = Column(String)
    payload = Column(Text)
    status = Column(String, default="pending")
    result = Column(Text)
    requested_by = Column(String, default="hermes")
    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime)


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_type = Column(String, nullable=False)
    portfolio_id = Column(String)
    ticker = Column(String)
    message = Column(Text, nullable=False)
    channel = Column(String, default="telegram")
    sent_at = Column(DateTime, default=datetime.utcnow)
    error = Column(Text)


# =====================================================================
# Pydantic Schemas (API)
# =====================================================================

class SignalAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    STRONG_BUY = "STRONG_BUY"
    STRONG_SELL = "STRONG_SELL"


class SignalCreate(BaseModel):
    ticker: str
    action: SignalAction
    sentiment: float = Field(..., ge=-1, le=1)
    strength: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    price_at_signal: float | None = None
    source: str = "api"


class SignalRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    timestamp: datetime
    ticker: str
    action: str
    sentiment: float
    strength: float
    confidence: float
    source: str
    price_at_signal: float | None
    consumed: int


class PortfolioStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    LIQUIDATING = "liquidating"
    LIQUIDATED = "liquidated"


class PortfolioRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    strategy_type: str
    status: PortfolioStatus
    cash_initial: float
    cash_current: float
    max_trade_amount: float | None
    fee_per_order: float
    config_json: str | None


class PortfolioSummary(BaseModel):
    id: str
    name: str
    strategy_type: str
    status: str
    cash_current: float
    positions_value: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float
    nb_positions: int


class TradeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    portfolio_id: str
    ticker: str
    action: str
    quantity: float
    price: float
    amount: float
    fees: float
    realized_pnl: float | None
    executed_at: datetime


class CommandType(str, Enum):
    LIQUIDATE = "LIQUIDATE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    CONFIG_UPDATE = "CONFIG_UPDATE"
    BUY = "BUY"
    SELL = "SELL"
    REBALANCE = "REBALANCE"
    WITHDRAW = "WITHDRAW"
    DEPOSIT = "DEPOSIT"


class CommandCreate(BaseModel):
    command_type: CommandType
    portfolio_id: str | None = None
    payload: dict | None = None
    requested_by: str = "hermes"


class CommandRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    command_type: str
    portfolio_id: str | None
    payload: str | None
    status: str
    result: str | None
    requested_by: str
    created_at: datetime
    processed_at: datetime | None


class DecisionCreate(BaseModel):
    action: SignalAction
    ticker: str
    portfolio_id: str
    confidence: float = Field(..., ge=0, le=1)
    amount: float | None = None
    reason: str | None = None


class StatusRead(BaseModel):
    pipeline: str
    last_run: datetime | None
    portfolios: dict[str, float]
    pending_commands: int
    unread_signals: int
