# AGENTS.md — Trading Engine V4.1

## Contexte & Objectif

Système de **simulation de trading algorithmique multi-stratégies** avec analyse de sentiment ML (FinancialBERT + RoBERTa), orchestré par **Prefect v3** en architecture **event-driven**.

Le système est conçu pour être piloté par **Hermes** (assistant IA) via une **API FastAPI**, sans que Hermes n'exécute directement le code métier.

---

## Stack Technique

| Couche | Technologie |
|--------|-------------|
| Langage | Python 3.11+ |
| Orchestration | Prefect v3 (flows, events, automations) |
| API | FastAPI + Pydantic v2 |
| Base de données | SQLite (→ Postgres plus tard) |
| ML / GPU | PyTorch + Transformers (lazy load) |
| Sources données | Finnhub REST + WebSocket, Alpha Vantage |
| Notifications | Telegram Bot API |
| Dashboard | Metabase (branché sur SQLite) |

---

## Structure du Projet

```
trading/
├── .env                          # Clés API (copié depuis ~/.hermes/.env)
├── pyproject.toml
├── requirements.txt
├── sql/
│   └── schema.sql                # Schéma complet (10 tables + vues)
├── src/trading/
│   ├── core/                     # Config, DB (SQLAlchemy 2.0), Models
│   │   ├── config.py
│   │   ├── database.py
│   │   └── models.py             # ORM + Pydantic schemas
│   ├── api/                      # FastAPI
│   │   ├── main.py               # App + lifespan
│   │   ├── dependencies.py       # Auth API key
│   │   └── routes/
│   │       ├── status.py         # GET /status
│   │       ├── portfolios.py     # CRUD + liquidate/pause/resume
│   │       ├── strategies.py     # Config dynamique
│   │       └── decisions.py      # Injection manuelle Hermes
│   ├── events/                   # Event Layer (Prefect Events)
│   │   ├── listener.py           # Service asyncio (websocket + polling)
│   │   └── emitters.py           # Helpers emit_event()
│   ├── ingestion/
│   │   └── collector.py          # Finnhub REST fetcher
│   ├── sentiment/
│   │   └── analyzer.py           # FinBERT + RoBERTa (thread-safe)
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
│   └── flows/                    # Flows Prefect indépendants
│       ├── ingestion_flow.py     # Fetch données (schedule 2min)
│       ├── sentiment_flow.py     # Analyse ML (event-driven)
│       ├── strategy_flow.py      # Exécution trades (event-driven)
│       ├── command_flow.py       # Traite ordres Hermes
│       ├── metrics_flow.py       # Snapshots PnL (schedule 1h)
│       ├── notifications_flow.py # Telegram (schedule 20h)
│       └── deploy.py             # Crée deployments + automations
├── scripts/
│   ├── init_db.py                # Initialise SQLite + portfolios
│   ├── run_api.py                # Lance FastAPI (uvicorn)
│   └── run_listener.py           # Lance EventListener asyncio
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

### 1. Initialiser la base

```bash
python scripts/init_db.py
```

Crée `data/trading.db` + 3 portefeuilles (simulation: $3000, rotation: $3000, ninja: €500).

### 2. Démarrer Prefect Server (UI)

```bash
prefect server start
```
→ UI accessible sur http://localhost:4200

### 3. Démarrer l'API FastAPI

```bash
python scripts/run_api.py
```
→ API sur http://localhost:8000/docs (Swagger)
→ Clé API par défaut: `dev-secret-change-me` (header `x-api-key`)

### 4. Démarrer le Listener (WebSocket + Polling)

```bash
python scripts/run_listener.py
```
→ Écoute Finnhub WebSocket (prix temps réel)
→ Poll les news toutes les minutes
→ Poll les commandes Hermes toutes les 30s
→ Émet des Prefect Events

### 5. Créer les Deployments Prefect

```bash
python -m trading.flows.deploy
```

### 6. Tester un flow manuellement

```bash
python -m trading.flows.ingestion_flow
python -m trading.flows.sentiment_flow
python -m trading.flows.strategy_flow --portfolio simulation
```

---

## Variables d'Environnement Critiques

Définies dans `.env` (récupérées depuis `~/.hermes/.env`):

```bash
FINNHUB_API_KEY=d8b0219r01qk20sp3cqgd8b0219r01qk20sp3cr0
TELEGRAM_BOT_TOKEN=8590713252:AAErFmDWzsB-xD44l1lMgIgFbJcKy7Orp_s
TELEGRAM_CHAT_ID=7139818351
API_KEY=dev-secret-change-me
DATABASE_URL=sqlite:///data/trading.db
```

---

## Architecture Event-Driven

### Comment ça marche

1. **EventListener** (`run_listener.py`) écoute les sources et écrit en DB
2. Quand il y a du nouveau, il appelle `emit_event()` (Prefect)
3. **Prefect Automations** détectent l'event et déclenchent le flow associé
4. Le flow lit l'état depuis SQLite, exécute sa logique, écrit le résultat
5. Si un portfolio est modifié, un event `portfolio.updated` est émis
6. Ce event déclenche `metrics_flow` + `notifications_flow`

### Events du système

| Event | Émis par | Déclenche |
|-------|----------|-----------|
| `market.price.updated` | EventListener | `strategy_execution_flow` |
| `news.batch.available` | EventListener | `sentiment_analysis_flow` |
| `signal.generated` | SentimentAnalyzer | `strategy_execution_flow` |
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

### Vérifier la base SQLite

```bash
sqlite3 data/trading.db ".tables"
sqlite3 data/trading.db "SELECT * FROM portfolios;"
sqlite3 data/trading.db "SELECT * FROM signals WHERE consumed=0;"
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

---

## Conventions de Code

- **Python 3.11+**, type hints obligatoires
- **SQLAlchemy 2.0** (style moderne, pas l'ancien Query)
- **Pydantic v2** pour les API
- **Pas de JSON** pour les portefeuilles → tout en SQLite
- **Lazy loading** des modèles ML (ne pas charger au démarrage API)
- **`db_session()`** context manager pour transactions hors FastAPI
- **`get_db()`** dependency pour FastAPI
- **Thread-safe** : `SentimentAnalyzer` utilise un `threading.Lock()` pour le chargement GPU

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
- **Skills** : `.kimi/skills/prefect-trading/SKILL.md`
