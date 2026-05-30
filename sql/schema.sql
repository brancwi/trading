-- ============================================================
-- Schéma V4 - Système de Trading Modulaire
-- SQLite → Postgres-ready (syntaxe standard SQL)
-- ============================================================

-- 1. NEWS BRUTES (ingestion)
CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    source TEXT NOT NULL,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    url TEXT,
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_news_ticker ON news(ticker);
CREATE INDEX IF NOT EXISTS idx_news_processed ON news(processed);
CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news(timestamp);

-- 2. MARKET DATA (prix temps réel)
CREATE TABLE IF NOT EXISTS market_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    price REAL NOT NULL,
    open_price REAL,
    high REAL,
    low REAL,
    change_pct REAL,
    volume INTEGER,
    source TEXT DEFAULT 'finnhub'
);

CREATE INDEX IF NOT EXISTS idx_md_ticker ON market_data(ticker);
CREATE INDEX IF NOT EXISTS idx_md_timestamp ON market_data(timestamp);

-- 3. SENTIMENT SCORES (output du Sentiment Engine)
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    finbert_score REAL,
    roberta_score REAL,
    combined_score REAL NOT NULL,
    confidence REAL,
    keywords TEXT,
    anomaly_flag INTEGER DEFAULT 0,
    FOREIGN KEY (news_id) REFERENCES news(id)
);

CREATE INDEX IF NOT EXISTS idx_sent_ticker ON sentiment_scores(ticker);
CREATE INDEX IF NOT EXISTS idx_sent_timestamp ON sentiment_scores(timestamp);

-- 4. SIGNALS (couche critique - intermédiaire entre sentiment et stratégies)
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    ticker TEXT NOT NULL,
    action TEXT CHECK(action IN ('BUY', 'SELL', 'HOLD', 'STRONG_BUY', 'STRONG_SELL')),
    sentiment REAL NOT NULL,
    strength REAL NOT NULL,       -- 0.0 à 1.0
    confidence REAL NOT NULL,     -- 0.0 à 1.0
    source TEXT DEFAULT 'ml_pipeline',
    price_at_signal REAL,
    expires_at DATETIME,          -- validité du signal
    consumed INTEGER DEFAULT 0    -- consommé par une stratégie ?
);

CREATE INDEX IF NOT EXISTS idx_sig_ticker ON signals(ticker);
CREATE INDEX IF NOT EXISTS idx_sig_action ON signals(action);
CREATE INDEX IF NOT EXISTS idx_sig_consumed ON signals(consumed);
CREATE INDEX IF NOT EXISTS idx_sig_timestamp ON signals(timestamp);

-- 5. PORTEFEUILLES (définition)
CREATE TABLE IF NOT EXISTS portfolios (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    strategy_type TEXT NOT NULL CHECK(strategy_type IN ('simulation', 'rotation', 'ninja')),
    base_currency TEXT DEFAULT 'USD',
    cash_initial REAL NOT NULL,
    cash_current REAL NOT NULL,
    max_trade_amount REAL,
    fee_per_order REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'paused', 'liquidating', 'liquidated')),
    config_json TEXT,             -- paramètres dynamiques JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 6. POSITIONS (état courant)
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_entry_price REAL NOT NULL,
    current_price REAL,
    current_value REAL,
    unrealized_pnl REAL DEFAULT 0,
    unrealized_pnl_pct REAL DEFAULT 0,
    sector TEXT,
    opened_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
);

CREATE INDEX IF NOT EXISTS idx_pos_portfolio ON positions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_pos_ticker ON positions(ticker);

-- 7. TRADES (historique immuable)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('BUY', 'SELL')),
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    amount REAL NOT NULL,
    fees REAL NOT NULL,
    realized_pnl REAL,
    signal_id INTEGER,
    strategy_type TEXT,
    executed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);

CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_trades_executed ON trades(executed_at);

-- 8. PORTFOLIO HISTORY (snapshots pour Metabase)
CREATE TABLE IF NOT EXISTS portfolio_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    cash REAL NOT NULL,
    positions_value REAL NOT NULL,
    total_value REAL NOT NULL,
    total_pnl REAL,
    total_pnl_pct REAL,
    drawdown_pct REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id)
);

CREATE INDEX IF NOT EXISTS idx_ph_portfolio ON portfolio_history(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_ph_timestamp ON portfolio_history(timestamp);

-- 9. COMMAND QUEUE (Hermes → Trading Engine)
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    command_type TEXT NOT NULL CHECK(command_type IN (
        'LIQUIDATE', 'PAUSE', 'RESUME', 'CONFIG_UPDATE',
        'BUY', 'SELL', 'REBALANCE', 'WITHDRAW', 'DEPOSIT'
    )),
    portfolio_id TEXT,
    payload TEXT,                 -- JSON
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    result TEXT,
    requested_by TEXT DEFAULT 'hermes',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    processed_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_cmd_status ON commands(status);
CREATE INDEX IF NOT EXISTS idx_cmd_portfolio ON commands(portfolio_id);

-- 10. ALERTS / NOTIFICATIONS (log de ce qui a été envoyé)
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_type TEXT NOT NULL,
    portfolio_id TEXT,
    ticker TEXT,
    message TEXT NOT NULL,
    channel TEXT DEFAULT 'telegram',
    sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_portfolio ON alerts(portfolio_id);

-- ============================================================
-- VUES POUR METABASE / ANALYSE
-- ============================================================

-- Vue synthèse portefeuilles
CREATE VIEW IF NOT EXISTS v_portfolio_summary AS
SELECT
    p.id,
    p.name,
    p.strategy_type,
    p.status,
    p.cash_current,
    COALESCE(SUM(pos.current_value), 0) AS positions_value,
    p.cash_current + COALESCE(SUM(pos.current_value), 0) AS total_value,
    (p.cash_current + COALESCE(SUM(pos.current_value), 0)) - p.cash_initial AS total_pnl,
    ROUND(((p.cash_current + COALESCE(SUM(pos.current_value), 0)) - p.cash_initial) / p.cash_initial * 100, 2) AS total_pnl_pct,
    COUNT(DISTINCT pos.ticker) AS nb_positions
FROM portfolios p
LEFT JOIN positions pos ON pos.portfolio_id = p.id
GROUP BY p.id;

-- Vue performance quotidienne
CREATE VIEW IF NOT EXISTS v_daily_performance AS
SELECT
    portfolio_id,
    DATE(timestamp) AS date,
    MAX(total_value) AS high,
    MIN(total_value) AS low,
    total_value AS close_value,
    total_pnl_pct
FROM portfolio_history
GROUP BY portfolio_id, DATE(timestamp)
ORDER BY date DESC;

-- ============================================================
-- DONNÉES INITIALES
-- ============================================================

INSERT OR IGNORE INTO portfolios (id, name, strategy_type, base_currency, cash_initial, cash_current, max_trade_amount, fee_per_order, config_json)
VALUES
('simulation', 'Simulation Day Trading', 'simulation', 'USD', 3000, 3000, 500, 1.0, '{"sentiment_threshold": 0.5, "cash_min": 100}'),
('rotation', 'Rotation Sectorielle', 'rotation', 'USD', 3000, 3000, 600, 1.0, '{"stop_loss_pct": -12, "take_profit_pct": 20, "take_profit_sell_pct": 50}'),
('ninja', 'Ninja Opportuniste', 'ninja', 'EUR', 500, 500, 150, 1.0, '{"cash_min": 50, "min_sectors": 3}');
