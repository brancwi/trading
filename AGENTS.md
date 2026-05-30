# AGENTS.md — Trading Engine V4.2

## Contexte & Objectif

Système de **simulation de trading algorithmique multi-stratégies** avec analyse de sentiment ML multi-tier (DistilRoBERTa + ModernFinBERT + Qwen3-0.6B + fallback cloud), orchestré par **Prefect v3** en architecture **event-driven**.

Le système est conçu pour être piloté par **Hermes** (assistant IA) via une **API FastAPI** + **protocole MCP** (SSE), sans que Hermes n'exécute directement le code métier.

---

## Stack Technique

| Couche | Technologie |
|--------|-------------|
| Langage | Python 3.11+ |
| Orchestration | Prefect v3 (flows, events, automations) |
| API | FastAPI + Pydantic v2 |
| Base de données | PostgreSQL 15 (Docker) — migrée depuis SQLite |
| ML / GPU | PyTorch + Transformers (lazy load) |
| Sources données | Finnhub REST + WebSocket, Alpha Vantage |
| Notifications | Telegram Bot API |
| Data Access MCP | MCP SDK 1.27.2 — SSE transport port 8001 |
| Monitoring | MonitoringService + tables audit, token usage, metrics |
| Dashboard | Metabase (branché sur PostgreSQL) |

---

## Structure du Projet

```
trading/
├── .env                          # Clés API (copié depuis ~/.hermes/.env)
├── pyproject.toml
├── requirements.txt
├── docker-compose.yml            # PostgreSQL + API + Listener + MCP + Prefect + Metabase
├── sql/
│   └── schema.sql                # Schéma complet (14+ tables + vues)
├── src/trading/
│   ├── core/                     # Config, DB (SQLAlchemy 2.0), Models
│   │   ├── config.py
│   │   ├── database.py
│   │   └── models.py             # ORM + Pydantic schemas
│   ├── api/                      # FastAPI
│   │   ├── main.py               # App + lifespan
│   │   ├── dependencies.py       # Auth API key
│   │   └── routes/
│   │       ├── status.py         # GET /status, /health
│   │       ├── portfolios.py     # CRUD + liquidate/pause/resume
│   │       ├── strategies.py     # Config dynamique
│   │       ├── decisions.py      # Injection manuelle Hermes
│   │       └── monitoring.py     # GET /monitoring/*, /monitoring/audit
│   ├── mcp/                      # MCP Server (SSE transport)
│   │   └── server.py             # FastMCP — tools de données pour Hermes
│   ├── events/                   # Event Layer (Prefect Events)
│   │   ├── listener.py           # Service asyncio (websocket + polling)
│   │   └── emitters.py           # Helpers emit_event()
│   ├── ingestion/
│   │   └── collector.py          # Finnhub REST fetcher
│   ├── sentiment/
│   │   ├── analyzer.py           # SentimentAnalyzerV2 — 4 tiers
│   │   ├── lexical_rules.py      # Override par mots-clés financiers
│   │   ├── cloud_fallback.py     # Fallback GPT-4/Claude API
│   │   └── token_tracker.py      # TokenUsageLog + coût estimé
│   ├── strategies/
│   │   ├── base.py               # StrategyBase ABC
│   │   ├── simulation.py         # Day-trading sur signaux
│   │   ├── rotation.py           # Stop-loss / Take-profit sectoriel
│   │   └── ninja.py              # Opportuniste diversification
│   ├── execution/
│   │   ├── engine.py             # Route les signaux vers stratégies
│   │   └── commands.py           # Command Bus (Hermes → Engine)
│   ├── notifier/
│   │   └── telegram.py           # Notifications Telegram
│   ├── monitoring/
│   │   └── service.py            # MonitorService — metrics, audit, token usage
│   └── flows/                    # Flows Prefect indépendants
│       ├── ingestion_flow.py     # Fetch données (schedule 2min)
│       ├── sentiment_flow.py     # Analyse ML (event-driven)
│       ├── strategy_flow.py      # Exécution trades (event-driven)
│       ├── command_flow.py       # Traite ordres Hermes
│       ├── metrics_flow.py       # Snapshots PnL (schedule 1h)
│       ├── notifications_flow.py # Telegram (schedule 20h)
│       └── deploy.py             # Crée deployments + automations
├── scripts/
│   ├── init_db.py                # Initialise DB + portfolios
│   ├── run_api.py                # Lance FastAPI (uvicorn)
│   ├── run_listener.py           # Lance EventListener asyncio
│   ├── run_mcp_server.py         # Lance MCP Server SSE (port 8001)
│   └── migrate_sqlite_to_postgres.py  # Migration SQLite → PostgreSQL
├── .kimi/skills/
│   └── prefect-trading/
│       └── SKILL.md              # Connaissance Prefect du projet
└── AGENTS.md                     # Ce fichier
```

---

## Démarrage Rapide (Debug)

### Prérequis

```bash
cd /home/brancwi/dev/projects/trading
source .venv/bin/activate
```

### Option A — Docker Compose (Recommandé)

```bash
# 1. Lancer tous les services
docker-compose up -d

# 2. Vérifier que tout est up
docker-compose ps

# 3. Accéder aux services
# API FastAPI  → http://localhost:8000/docs
# MCP Server   → http://localhost:8001/sse  (Hermes data access)
# Prefect UI   → http://localhost:4200
# Metabase     → http://localhost:3000

# 4. Voir les logs
docker-compose logs -f trading-api
docker-compose logs -f trading-listener
docker-compose logs -f trading-mcp-server
docker-compose logs -f prefect-server

# 5. Arrêter
docker-compose down

# 6. Reset complet (DB + volumes)
docker-compose down -v
```

### Option B — Manuel (Développement) — Dev Manager

Un script de gestion centralisée gère tous les services en local :

```bash
# Lancer tous les services
./scripts/dev_manager.sh start

# Voir l'état
./scripts/dev_manager.sh status

# Logs temps réel (tous)
./scripts/dev_manager.sh logs

# Logs d'un service spécifique
./scripts/dev_manager.sh logs mcp-server
./scripts/dev_manager.sh logs api-server
./scripts/dev_manager.sh logs listener

# Redémarrer un service après modification
./scripts/dev_manager.sh restart mcp-server
./scripts/dev_manager.sh restart api-server
./scripts/dev_manager.sh restart listener

# Arrêter tout
./scripts/dev_manager.sh stop
```

#### Démarrage manuel (sans le manager)

```bash
# 1. PostgreSQL (Docker)
docker compose up -d postgres

# 2. MCP Server
python scripts/run_mcp_server.py
# → http://localhost:8001/sse

# 3. API FastAPI
python scripts/run_api.py
# → http://localhost:8000/docs

# 4. Listener
python scripts/run_listener.py

# 5. Prefect Server (optionnel)
prefect server start
# → http://localhost:4200
```

### Option C — Docker uniquement (sans Compose)

```bash
# Build
docker build -t trading-engine .

# Run API
docker run -d -p 8000:8000 -v trading-data:/app/data --env-file .env trading-engine

# Run MCP Server
docker run -d -p 8001:8001 -v trading-data:/app/data --env-file .env trading-engine python scripts/run_mcp_server.py

# Run Listener
docker run -d -v trading-data:/app/data --env-file .env trading-engine python scripts/run_listener.py

# Run Prefect Server
docker run -d -p 4200:4200 prefecthq/prefect:3-latest prefect server start --host 0.0.0.0

# Run Metabase
docker run -d -p 3000:3000 -v metabase-data:/metabase-data metabase/metabase
```

---

## Variables d'Environnement Critiques

Définies dans `.env` (récupérées depuis `~/.hermes/.env`):

```bash
FINNHUB_API_KEY=xxx
TELEGRAM_BOT_TOKEN=xxx
TELEGRAM_CHAT_ID=xxx
API_KEY=dev-secret-change-me
DATABASE_URL=postgresql://trading:changeme@localhost:5432/trading
PREFECT_API_URL=http://localhost:4200/api
```

---

## Architecture Event-Driven

### Comment ça marche

1. **EventListener** (`run_listener.py`) écoute les sources et écrit en DB
2. Quand il y a du nouveau, il appelle `emit_event()` (Prefect)
3. **Prefect Automations** détectent l'event et déclenchent le flow associé
4. Le flow lit l'état depuis PostgreSQL, exécute sa logique, écrit le résultat
5. Si un portfolio est modifié, un event `portfolio.updated` est émis
6. Ce event déclenche `metrics_flow` + `notifications_flow`

### Events du système

| Event | Émis par | Déclenche |
|-------|----------|-----------|
| `market.price.updated` | EventListener | `strategy_execution_flow` |
| `news.batch.available` | EventListener | `sentiment_analysis_flow` |
| `signal.generated` | SentimentAnalyzerV2 | `strategy_execution_flow` |
| `hermes.command.received` | API routes | `command_processing_flow` |
| `portfolio.updated` | StrategyBase | `metrics_flow`, `notifications_flow` |
| `trade.executed` | StrategyBase | — (log only) |

---

## API FastAPI (Hermes Control Plane)

Toutes les routes protégées par `x-api-key`.

### Endpoints clés

```bash
# Status global
curl -H "x-api-key: dev-secret-change-me" http://localhost:8000/status

# Synthèse PnL
curl -H "x-api-key: dev-secret-change-me" http://localhost:8000/portfolios/summary

# Monitoring — audit, tokens, metrics
curl -H "x-api-key: dev-secret-change-me" http://localhost:8000/monitoring/audit
curl -H "x-api-key: dev-secret-change-me" http://localhost:8000/monitoring/token-usage
curl -H "x-api-key: dev-secret-change-me" http://localhost:8000/monitoring/metrics

# Injecter une décision manuelle
curl -X POST -H "x-api-key: dev-secret-change-me" \
  -H "Content-Type: application/json" \
  -d '{"action":"BUY","ticker":"NVDA","portfolio_id":"simulation","confidence":0.85}' \
  http://localhost:8000/decisions

# Liquidation
curl -X POST -H "x-api-key: dev-secret-change-me" \
  http://localhost:8000/portfolios/simulation/liquidate

# Pause / Resume
curl -X POST -H "x-api-key: dev-secret-change-me" \
  http://localhost:8000/portfolios/simulation/pause

# Modifier config stratégie
curl -X POST -H "x-api-key: dev-secret-change-me" \
  -H "Content-Type: application/json" \
  -d '{"sentiment_threshold":0.7}' \
  http://localhost:8000/strategies/simulation/config
```

---

## MCP Server — Hermes Data Access

Transport **SSE** sur le port **8001**. Hermes se connecte à `http://localhost:8001/sse`.

### Tools exposés

| Tool | Description |
|------|-------------|
| `list_portfolios` | Liste tous les portefeuilles avec soldes |
| `get_positions(portfolio_id)` | Positions actuelles d'un portefeuille |
| `get_portfolio_details(portfolio_id)` | Détails complets (info + positions + trades) |
| `get_trade_history(portfolio_id, limit=50)` | Historique des trades |
| `get_balance_history(portfolio_id, limit=50)` | Historique solde time-series |
| `get_signals(consumed=None, limit=50)` | Signaux générés par le sentiment engine |
| `get_sentiment_scores(ticker=None, limit=50)` | Scores multi-modèles (FinBERT, Qwen, Cloud) |
| `get_market_data(ticker, limit=50)` | Données OHLCV |
| `get_news(ticker=None, limit=50)` | News financières |
| `execute_sql_query(query)` | SQL read-only (SELECT uniquement) — sécurisé |
| `reserve_capital(portfolio_id, amount, reason)` | Réserve du capital (soustrait du cash disponible) |
| `release_capital(portfolio_id, amount)` | Libère du capital réservé |
| `get_capital_movements(portfolio_id, limit=50)` | Historique des mouvements de capital |
| `get_token_usage(hours=24)` | Consommation tokens et coût estimé |
| `get_audit_log(hours=24, event_type=None)` | Journal d'audit |

### Capital Movements

- `Portfolio.cash_available = cash_current - reserved_cash`
- Les stratégies `buy()` doivent vérifier `cash_available` avant d'acheter
- `reserve_capital` et `release_capital` écrivent dans `capital_movements` table

---

## Stratégies Implémentées

| Stratégie | Capital | Type | Règles clés |
|-----------|---------|------|-------------|
| **simulation** | $3000 USD | day-trading | Achat si sentiment > 0.5, max $500/trade, pas de stop-loss |
| **rotation** | $3000 USD | sectorielle | Stop-loss -12%, take-profit +20% (vente 50%), rééquilibrage |
| **ninja** | €500 EUR | opportuniste | Max €150/trade, diversification min 3 secteurs |

---

## Debugging

### Voir les logs Prefect

```bash
prefect flow-run logs <FLOW_RUN_ID>
prefect flow-run ls
```

### Relancer un flow spécifique

```bash
python -m trading.flows.strategy_flow --portfolio rotation
```

### Vérifier la base PostgreSQL

```bash
psql -h localhost -U trading -d trading -c "\dt"
psql -h localhost -U trading -d trading -c "SELECT * FROM portfolios;"
psql -h localhost -U trading -d trading -c "SELECT * FROM signals WHERE consumed=0;"
```

### Tester l'API localement

```bash
curl http://localhost:8000/health
```

### Vérifier le listener

```bash
# Doit afficher "WebSocket connecté — 10 tickers"
python scripts/run_listener.py
```

### Vérifier le MCP Server

```bash
# Test basique (le serveur doit répondre sur /sse)
curl http://localhost:8001/sse
```

### Test end-to-end

```bash
# Valide MCP, capital movements, stratégies cash_available
python scripts/test_e2e.py
```

---

## Conventions de Code

- **Python 3.11+**, type hints obligatoires
- **SQLAlchemy 2.0** (style moderne, pas l'ancien Query)
- **Pydantic v2** pour les API
- **PostgreSQL** — tout en base, pas de JSON files
- **Lazy loading** des modèles ML (ne pas charger au démarrage API)
- **`db_session()`** context manager pour transactions hors FastAPI
- **`get_db()`** dependency pour FastAPI
- **Thread-safe** : `SentimentAnalyzerV2` utilise un `threading.Lock()` pour le chargement GPU
- **Sentiment Engine v2** : 4 tiers — lexical override → DistilRoBERTa + ModernFinBERT (séquentiel) → Qwen arbitre (si divergence > 0.3) → cloud fallback (si Qwen incertain)
- **Token Tracking** : comptage input/output tokens pour chaque appel Qwen / cloud, avec estimation de coût ($/call) sur GPT-4o-mini, Claude Haiku, etc. Stocké en DB pour analyse cumulée
- **Capital Movements** : `reserved_cash` sur `Portfolio`, `cash_available = cash_current - reserved_cash`, tracked dans `capital_movements` table
- **8GB VRAM** : tous les modèles chargés simultanément (~1.9GB total), pas de batching (réactivité), pas de quantization (précision)
- **MCP SDK 1.27.2** : FastMCP avec `run(transport="sse", port=8001)`

---

## Points d'Extension

### Ajouter une nouvelle stratégie

1. Créer `src/trading/strategies/ma_strategie.py` (hérite de `StrategyBase`)
2. Ajouter au `STRATEGY_MAP` dans `execution/engine.py` et `flows/strategy_flow.py`
3. Créer un portfolio dans `scripts/init_db.py`
4. Créer un deployment Prefect si besoin de scheduling spécifique

### Ajouter une source de données

1. Étendre `MarketDataCollector` dans `ingestion/collector.py`
2. Ajouter un emitter dans `events/emitters.py`
3. Brancher dans `events/listener.py`

### Ajouter un nouveau flow Prefect

1. Créer `src/trading/flows/mon_flow.py` avec `@flow`
2. Ajouter au skill `.kimi/skills/prefect-trading/SKILL.md`
3. Créer un deployment via `deploy.py` ou `serve()`

### Ajouter un tool MCP

1. Ajouter une fonction `@mcp.tool()` dans `src/trading/mcp/server.py`
2. Utiliser `_Session()` pour les requêtes SQL
3. Utiliser `json.dumps(..., default=_json_serial)` pour sérialiser les datetimes
4. Restart le service `docker-compose restart trading-mcp-server`

---

## Git

```bash
git log --oneline
```

Historique:
- `77f4f02` feat(v4.1): architecture event-driven Prefect avec flows indépendants
- `45fdac4` feat(v4): architecture modulaire event-driven + API FastAPI

---

## Contact & Support

- **Projet** : `/home/brancwi/dev/projects/trading`
- **Prefect UI** : http://localhost:4200 (quand `prefect server start`)
- **API Docs** : http://localhost:8000/docs (quand `python scripts/run_api.py`)
- **MCP Server** : http://localhost:8001/sse (quand `python scripts/run_mcp_server.py`)
- **Skills** :
  - `.kimi/skills/prefect-mastery/SKILL.md` — Référence complète Prefect v3 (pur, générique)
  - `.kimi/skills/prefect-trading/SKILL.md` — Connaissance Prefect spécifique au projet
