# AGENTS.md - Trading Engine V4

## Contexte

Refonte complète du système de trading V3 (monolithique) vers V4 (modulaire, event-driven).
L'objectif principal : externaliser l'exécution du "cerveau" Hermes tout en gardant le contrôle via API.

## Architecture V4

```
Hermes (IA)
    ↓ REST API (FastAPI)
Command Queue (SQLite table `commands`)
    ↓
Trading Engine (Python)
├── ingestion/       → fetch_news(), fetch_prices()
├── sentiment/       → analyze_unprocessed_news() → table `signals`
├── strategies/      → simulation, rotation, ninja (indépendantes)
├── execution/       → engine.run_all(prices) + commands processor
└── notifier/        → Telegram
    ↓
SQLite (signals, trades, portfolios, positions, portfolio_history, commands)
    ↓
Metabase (dashboard)
```

## Structure des répertoires

```
trading/
├── src/trading/
│   ├── core/         → config, database, models (SQLAlchemy + Pydantic)
│   ├── api/          → FastAPI + routes (status, portfolios, strategies, decisions)
│   ├── ingestion/    → collector de données marché
│   ├── sentiment/    → FinancialBERT + RoBERTa (lazy load GPU)
│   ├── strategies/   → base.py + simulation.py + rotation.py + ninja.py
│   ├── execution/    → engine.py + commands.py (Command Bus)
│   ├── notifier/     → telegram.py
│   └── flows/        → Prefect flows (trading_flow.py)
├── scripts/
│   ├── init_db.py    → initialise SQLite + portefeuilles
│   ├── run_api.py    → uvicorn FastAPI
│   └── run_pipeline.py → pipeline CLI (sans Prefect)
├── sql/schema.sql    → schéma complet SQL
├── pyproject.toml
├── requirements.txt
└── .env.example
```

## Conventions de code

- Python 3.11+, type hints obligatoires
- SQLAlchemy 2.0 (style moderne)
- Pydantic v2 pour les API
- Pas de JSON pour les portefeuilles → tout en SQLite
- Lazy loading des modèles ML (ne pas charger au démarrage API)
- `db_session()` context manager pour les transactions hors FastAPI
- `get_db()` dependency pour FastAPI

## Points d'extension

- **Nouvelle stratégie** : hériter de `StrategyBase`, l'ajouter à `STRATEGY_MAP`
- **Nouvelle source de données** : étendre `MarketDataCollector`
- **Nouveau notifier** : créer un module parallel à `telegram.py`
- **Nouvelle commande Hermes** : ajouter au `CommandProcessor._execute()`

## Tests rapides

```bash
# 1. Init DB
python scripts/init_db.py

# 2. Lancer API
python scripts/run_api.py
# → http://localhost:8000/docs

# 3. Lancer pipeline
python scripts/run_pipeline.py --tickers AAPL NVDA

# 4. Prefect flow
python -m trading.flows.trading_flow
```

## Variables d'environnement critiques

- `DATABASE_URL` → SQLite par défaut
- `FINNHUB_API_KEY` / `ALPHA_VANTAGE_KEY`
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID`
- `API_KEY` → clé secrète pour les routes protégées

## Migration depuis V3

1. Récupérer les `.env` existants
2. Exporter JSON → SQL (portefeuilles)
3. Pointer `DATABASE_URL` vers le nouveau `data/trading.db`
4. Adapter cron pour appeler `run_pipeline.py` ou Prefect
