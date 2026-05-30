---
name: prefect-trading
description: >
  Connaissance spécifique au projet Trading Engine V4.1 (orchestré par Prefect v3).

  ACTIVE CE SKILL immédiatement quand l'utilisateur travaille dans ce répertoire
  et mentionne ou sous-entend :
  - trading, simulation, rotation, ninja, day trading, sector rotation
  - portfolio, portefeuille, position, trade, PnL, drawdown
  - strategy, stratégie, backtest, signal, sentiment analysis
  - FinBERT, RoBERTa, FinancialBERT, NLP, sentiment score
  - Finnhub, market data, prix temps réel, websocket, news ingestion
  - Hermes, command bus, commande, liquidation, pause, resume
  - event-driven trading, trading flow, trading pipeline
  - Telegram, notification de trade, alerte de signal
  - ajouter une stratégie, modifier un flow, nouveau portefeuille
  - DB trading, SQLite, table signals, table trades
  - Metabase, dashboard trading, métriques de performance

  ACTIVE aussi quand l'utilisateur :
  - veut tester/debug un flow du système de trading
  - demande comment relancer le listener ou l'API
  - parle de simulation de trading algorithmique
  - mentionne "mon système de trading" ou "mon bot de trading"
  - veut configurer les clés API (Finnhub, Telegram)

  NE PAS activer si :
  - l'utilisateur parle de Prefect en général sans lien avec ce projet
    (utiliser prefect-mastery à la place)
  - c'est une question purement financière sans lien au code
  - le contexte est un autre projet de trading (pas celui-ci)
  - c'est une question sur FastAPI/Metabase général sans lien Prefect/trading
---

# Skill: Prefect Trading Orchestration

## Contexte

Ce skill documente l'utilisation de **Prefect v3** dans le projet Trading Engine. Il est destiné à être utilisé par n'importe quelle session IA pour comprendre, modifier ou déboguer l'orchestration sans avoir à relire toute la doc Prefect.

## Architecture Event-Driven

Le système utilise **6 flows indépendants** orchestrés par des **Prefect Events** et **Automations**.

```
EventListener (asyncio)  →  emit_event()  →  Automation  →  Flow
```

### Flows

| Flow | Fichier | Déclencheur | Rôle |
|------|---------|-------------|------|
| `ingestion_flow` | `src/trading/flows/ingestion_flow.py` | Schedule (2 min) | Fetch news + prix |
| `sentiment_analysis_flow` | `src/trading/flows/sentiment_flow.py` | Event `news.batch.available` | Analyse FinBERT/RoBERTa |
| `strategy_execution_flow` | `src/trading/flows/strategy_flow.py` | Event `signal.generated` | Exécute trades |
| `command_processing_flow` | `src/trading/flows/command_flow.py` | Event `hermes.command.received` | Traite ordres Hermes |
| `metrics_flow` | `src/trading/flows/metrics_flow.py` | Schedule (1h) + Event | Snapshots PnL |
| `notifications_flow` | `src/trading/flows/notifications_flow.py` | Schedule (20h) + Event | Telegram |

### Events émis par le système

```python
from trading.events.emitters import (
    emit_market_price_updated,      # "market.price.updated"
    emit_news_batch_available,      # "news.batch.available"
    emit_signal_generated,          # "signal.generated"
    emit_hermes_command_received,   # "hermes.command.received"
    emit_portfolio_updated,         # "portfolio.updated"
    emit_trade_executed,            # "trade.executed"
)
```

## Commandes essentielles

```bash
# Démarrer le serveur Prefect (UI + API)
prefect server start

# Lancer un flow manuellement
python -m trading.flows.ingestion_flow
python -m trading.flows.sentiment_flow
python -m trading.flows.strategy_flow --portfolio simulation

# Créer les deployments
python -m trading.flows.deploy

# Voir les logs d'un flow
prefect flow-run logs <FLOW_RUN_ID>

# Lister les deployments
prefect deployment ls
```

## Retry Policies

| Flow | Retries | Delay | Raison |
|------|---------|-------|--------|
| `ingestion_flow` | 3 | 10s | API externe |
| `sentiment_analysis_flow` | 1 | 30s | GPU/VRAM |
| `strategy_execution_flow` | 0 | — | Déterministe |
| `notifications_flow` | 3 | 5s | Telegram API |

## Patterns

### Ajouter un nouveau flow

1. Créer `src/trading/flows/mon_flow.py` avec `@flow`
2. Créer un deployment : `mon_flow.serve(name="mon-flow")`
3. Créer une Automation via UI ou SDK

### Déclencher un flow depuis l'API FastAPI

```python
from prefect.events import emit_event
emit_event(
    event="mon.event",
    resource={"prefect.resource.id": "mon.resource"},
    payload={"key": "value"}
)
```

### Exécuter en parallèle

```python
from prefect import flow

@flow
def parent_flow():
    strategy_execution_flow.submit("simulation")
    strategy_execution_flow.submit("rotation")
    strategy_execution_flow.submit("ninja")
```

## Anti-patterns à éviter

- **Ne pas** charger les modèles ML dans un flow qui tourne fréquemment → lazy-load avec lock
- **Ne pas** passer d'état en mémoire entre flows → toujours passer par SQLite
- **Ne pas** créer de flows < 100ms de travail → utiliser des tasks dans un flow existant

## Ressources

- Doc Prefect v3 : https://docs.prefect.io/v3/get-started
- Events : https://docs.prefect.io/v3/concepts/events
- Automations : https://docs.prefect.io/v3/concepts/automations
