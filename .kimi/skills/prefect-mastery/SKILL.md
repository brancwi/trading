---
name: prefect-mastery
description: >
  Référence complète pour construire, déployer et débugger des applications
  avec Prefect v3 (orchestrateur de workflows Python).

  ACTIVE CE SKILL immédiatement quand l'utilisateur mentionne ou sous-entend :
  - Prefect, @flow, @task, flow.run(), deployment, serve(), deploy()
  - orchestration de workflows, workflow engine, pipeline automation
  - scheduling, cron, interval, timer, job scheduler
  - retry, cache, concurrence, parallélisme, async tasks
  - event-driven, webhook trigger, automation rule, emit_event
  - ETL, data pipeline, batch processing, workflow DAG
  - remplacer Airflow, Celery, Cron, Rundeck, Luigi
  - work pool, worker, infrastructure dynamique, Docker/K8s deployment
  - human-in-the-loop, pause workflow, interactive approval
  - Prefect UI, Prefect Cloud, Prefect Server, MCP server

  ACTIVE aussi quand l'utilisateur :
  - veut exécuter des tâches en parallèle avec Python
  - cherche comment gérer retry/caching nativement
  - demande un scheduler Python avec UI
  - veut connecter un webhook à un workflow Python
  - a besoin d'observabilité sur des scripts Python (logs, états, UI)
  - mentionne "scheduler pour mon script Python" ou "task runner"

  NE PAS activer si :
  - l'utilisateur parle uniquement de FastAPI/Flask/Django sans orchestration
  - c'est une question purement SQL/DB sans workflow
  - le contexte est uniquement le projet trading (utiliser prefect-trading à la place)
  - c'est une question sur Docker/K8s général sans lien Prefect
---

# Prefect Mastery — Skill Référence v3

## Vue d'ensemble

**Prefect** transforme des fonctions Python en pipelines production-grade avec `@flow` et `@task`. Pas de DSL, pas de YAML obligatoire.

**Version cible** : Prefect 3.x (events/automations ouverts, -90% overhead vs v2)

> **Quand lire les références détaillées** : les fichiers dans `references/` couvrent chaque domaine en profondeur. Lis le skill principal d'abord, puis plonge dans la référence pertinente quand tu en as besoin.

---

## Patterns Essentiels (80% des cas)

### 1. Flow + Tasks basiques

```python
from prefect import flow, task

@task(retries=3, retry_delay_seconds=10)
def fetch(url: str) -> dict:
    import requests
    return requests.get(url, timeout=30).json()

@flow(log_prints=True)
def mon_pipeline():
    data = fetch("https://api.example.com")
    print(data)
```

### 2. Concurrence — `.submit()`

```python
@flow
def parallel_flow():
    f1 = fetch.submit("https://api.example.com/a")
    f2 = fetch.submit("https://api.example.com/b")
    return f1.result(), f2.result()
```

### 3. Event-Driven — émettre + réagir

```python
from prefect.events import emit_event
from prefect import flow

# Émettre un event (n'importe où dans ton code)
emit_event(
    event="fichier.cree",
    resource={"prefect.resource.id": "fichier.123"},
    payload={"path": "/data/file.csv"},
)

# Flow qui réagit à l'event
def mon_flow():
    pass

mon_flow.serve(
    name="mon-deployment",
    triggers=[{
        "type": "event",
        "expect": ["fichier.cree"],
        "parameters": {"payload": "{{ event.payload }}"},
    }],
)
```

### 4. Caching

```python
from prefect import task
from prefect.cache_policies import INPUTS
from datetime import timedelta

@task(cache_policy=INPUTS, cache_expiration=timedelta(minutes=5))
def expensive(x: int) -> int:
    import time; time.sleep(10)
    return x * 2
```

### 5. Déploiement rapide — `serve()`

```python
mon_flow.serve(
    name="mon-deployment",
    interval=300,           # toutes les 5 min
    cron="0 9 * * 1-5",     # ou cron
)
```

---

## Architecture Mentale

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Source    │────▶│   Flow      │────▶│   Result    │
│  (event,    │     │  (orchestre │     │  (persisté, │
│   schedule, │     │   tasks)    │     │   caché)    │
│   manual)   │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
        │                   │
        ▼                   ▼
┌─────────────┐     ┌─────────────┐
│  Events &   │     │   States    │
│ Automations │     │ (Completed, │
│             │     │  Failed...) │
└─────────────┘     └─────────────┘
```

---

## Décisions Rapides

| Question | Réponse |
|----------|---------|
| Un seul appel séquentiel ? | `task()` direct |
| Parallèle dans un flow ? | `task.submit()` |
| Fire-and-forget (web app) ? | `task.delay()` |
| Grouper logique visible dans UI ? | Subflow |
| Trigger externe (webhook, fichier) ? | `emit_event()` + Automation |
| Schedule simple ? | `flow.serve(interval=...)` |
| Infra dynamique (Docker, K8s) ? | `flow.deploy()` + Work Pool |
| Input humain ? | `pause_flow_run()` ou `receive_input()` |
| Éviter les doublons ? | `cache_policy=INPUTS` |

---

## Anti-Patterns Critiques

| ❌ Mauvais | ✅ Bon |
|-----------|--------|
| Flows < 100ms de travail | Task dans un flow existant |
| État en mémoire entre flows | DB / Variable / Block Prefect |
| Un flow par task | Flow avec subflows |
| Hardcoder credentials | Prefect Blocks / Variables |
| Polling actif dans un flow | `.serve()` schedule ou events |
| Ignorer les states | `return_state=True` pour control flow |

---

## Références Détaillées

Lis ces fichiers quand tu as besoin de creuser :

| Fichier | Contenu | Quand le lire |
|---------|---------|---------------|
| `references/fundamentals.md` | Flows, tasks, states, concurrence, caching | Tu débutes ou tu veux comprendre les bases |
| `references/event-driven.md` | Events, automations, webhooks, triggers | Tu veux du reactif / event-driven |
| `references/deployments.md` | `serve()`, `deploy()`, work pools, workers | Tu veux mettre en production |
| `references/advanced.md` | Interactive workflows, async, blocks, transactions | Tu veux des patterns avancés |

---

## CLI Rapide

```bash
prefect server start                    # UI + API locale
prefect flow-run logs <ID>              # Debug un run
prefect deployment ls                   # Voir les deployments
prefect work-pool create --type docker  # Créer un pool
```

---

## Ressources Externes

- Doc : https://docs.prefect.io/v3/get-started
- Events : https://docs.prefect.io/v3/concepts/events
- MCP Server : https://docs.prefect.io/v3/how-to-guides/ai/use-prefect-mcp-server
