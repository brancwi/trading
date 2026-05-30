# Prefect Mastery — Skill Référence Complète v3

## Vue d'ensemble

**Prefect** est un orchestrateur Python natif qui transforme des fonctions Python en pipelines production-grade. Pas de DSL, pas de YAML obligatoire. Tu écris du Python, tu ajoutes `@flow` et `@task`, et Prefect gère : retry, caching, concurrence, monitoring, scheduling, event-driven execution.

**Version cible** : Prefect 3.x (events/automations ouverts, -90% overhead vs v2)

---

## 1. Fondamentaux

### Flows

Un flow est une fonction Python décorée avec `@flow`. C'est l'unité d'orchestration.

```python
from prefect import flow

@flow(log_prints=True)
def mon_flow(name: str):
    return f"Hello {name}"

# Appel normal
mon_flow("world")
```

**Paramètres clés du décorateur `@flow`** :

| Paramètre | Description |
|-----------|-------------|
| `name` | Nom du flow (défaut = nom fonction) |
| `retries` | Nombre de retries si échec |
| `retry_delay_seconds` | Délai entre retries |
| `timeout_seconds` | Timeout total du flow |
| `log_prints=True` | Capture `print()` dans les logs Prefect |
| `persist_result=True` | Persiste le résultat pour inspection |

### Tasks

Une task est une unité de travail atomique, retryable, cacheable, observable.

```python
from prefect import task

@task(retries=3, retry_delay_seconds=10)
def fetch_data(url: str) -> dict:
    import requests
    return requests.get(url, timeout=30).json()
```

**Paramètres clés du décorateur `@task`** :

| Paramètre | Description |
|-----------|-------------|
| `retries` | Nombre de retries |
| `retry_delay_seconds` | Délai entre retries |
| `timeout_seconds` | Timeout par task |
| `persist_result=True` | Active le caching |
| `cache_policy` | Politique de cache (INPUTS, TASK_SOURCE, RUN_ID, DEFAULT) |
| `cache_expiration` | Durée de validité du cache |
| `refresh_cache=True` | Force le recalcul (ignore le cache) |

### 3 façons d'invoquer une task

```python
from prefect import flow, task

@task
def add(a: int, b: int) -> int:
    return a + b

# 1. Appel direct — bloquant, retourne le résultat
@flow
def sync_flow():
    result = add(1, 2)  # blocks, returns 3

# 2. .submit() — non bloquant, retourne PrefectFuture
@flow
def concurrent_flow():
    future = add.submit(1, 2)  # returns immediately
    result = future.result()   # blocks here

# 3. .delay() — fire-and-forget, exécuté par un task worker
@task
def send_email(to: str):
    pass

# Dans un endpoint web :
send_email.delay("user@example.com")  # ne bloque pas la réponse HTTP
```

| Méthode | Bloquant ? | Retourne | Contexte de résolution | Usage |
|---------|-----------|----------|------------------------|-------|
| `task()` | Oui | Résultat | N/A | Séquentiel simple |
| `task.submit()` | Non | `PrefectFuture` | Même contexte | Concurrence dans un flow |
| `task.delay()` | Non | `PrefectFuture` | Any worker | Background / fire-and-forget |

### Subflows

Un subflow est un flow appelé depuis un autre flow. C'est un groupe logique de tasks.

```python
from prefect import flow, task

@task
def extract():
    return {"data": [1, 2, 3]}

@task
def transform(data: list) -> list:
    return [x * 2 for x in data]

@flow
def etl_subflow():
    raw = extract()
    return transform(raw)

@flow
def main_pipeline():
    result = etl_subflow()  # subflow
    print(result)
```

**Subflow vs Task** : Un subflow a son propre DAG dans l'UI Prefect. Utilise un subflow quand tu veux un regroupement logique visible.

---

## 2. States (États)

Tout run (flow ou task) a un **State** riche qui contient : nom, type, données, timestamp.

### Types d'états

| Nom | Type | Terminal ? | Signification |
|-----|------|-----------|---------------|
| `Scheduled` | `SCHEDULED` | Non | Programmé pour plus tard |
| `Late` | `SCHEDULED` | Non | En retard (worker ne répond pas) |
| `Pending` | `PENDING` | Non | En attente de préconditions |
| `Running` | `RUNNING` | Non | En cours |
| `Retrying` | `RUNNING` | Non | Retry en cours |
| `Paused` | `PAUSED` | Non | Attente d'input humain |
| `Completed` | `COMPLETED` | **Oui** | Succès |
| `Cached` | `COMPLETED` | **Oui** | Résultat récupéré du cache |
| `Failed` | `FAILED` | **Oui** | Échec (exception) |
| `TimedOut` | `FAILED` | **Oui** | Timeout |
| `Crashed` | `CRASHED` | **Oui** | Problème infra (SIGTERM, OOM) |
| `Cancelled` | `CANCELLED` | **Oui** | Annulé par l'utilisateur |

### Manipuler les states

```python
from prefect import flow, task
from prefect.states import Completed, Failed

@task
def risky_task(fail: bool):
    if fail:
        return Failed(message="Échec volontaire")
    return Completed(message="Succès")

@flow
def demo_states():
    # Inspecter un state sans lever d'exception
    state = risky_task.submit(fail=True, return_state=True)
    
    if state.is_failed():
        print("La task a échoué, mais le flow continue")
    
    # Récupérer le résultat
    result = state.result()  # peut lever exception si Failed
```

### State Change Hooks

Exécuter du code quand un run change d'état :

```python
from prefect import Task, Flow
from prefect.states import State
from prefect.client.schemas.objects import TaskRun, FlowRun

def on_task_failure(task: Task, run: TaskRun, state: State):
    print(f"Task {task.name} failed: {state.message}")

@task(on_failure=[on_task_failure])
def ma_task():
    raise ValueError("oops")
```

---

## 3. Concurrence & Parallélisme

### ConcurrentTaskRunner (défaut)

```python
from prefect import flow, task

@task
def slow_task(n: int):
    import time
    time.sleep(n)
    return n

@flow
def parallel_flow():
    # Ces 3 tasks tournent en parallèle
    f1 = slow_task.submit(1)
    f2 = slow_task.submit(2)
    f3 = slow_task.submit(3)
    
    # Attendre tous les résultats
    return f1.result(), f2.result(), f3.result()
```

### Dynamic Task Mapping

Créer des tasks dynamiquement à partir de données runtime :

```python
from prefect import flow, task

@task
def process_item(item: str) -> str:
    return item.upper()

@flow
def map_flow():
    items = ["a", "b", "c"]
    
    # V1 : boucle explicite
    futures = [process_item.submit(i) for i in items]
    results = [f.result() for f in futures]
    
    # V2 : task.map() — plus concis
    results = process_item.map(items)  # retourne une liste de résultats
    
    return results
```

### Dépendances explicites avec `wait_for`

```python
from prefect import flow, task

@task
def step_a():
    return "a"

@task
def step_b():
    return "b"

@task
def step_c():
    return "c"

@flow
def dependency_flow():
    a = step_a.submit()
    b = step_b.submit()
    
    # c attend que a ET b soient terminés, même sans utiliser leurs résultats
    c = step_c.submit(wait_for=[a, b])
    return c.result()
```

---

## 4. Caching (Résultats Persistés)

### Cache basique

```python
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE, DEFAULT
from datetime import timedelta

@task(persist_result=True)
def expensive_computation(x: int) -> int:
    import time
    time.sleep(10)
    return x * 2

# DEFAULT = INPUTS + TASK_SOURCE + RUN_ID
# → cache hit si même inputs, même code, même run parent

@task(cache_policy=INPUTS, cache_expiration=timedelta(minutes=5))
def cache_5min(x: int) -> int:
    return x * 2

@task(cache_policy=INPUTS - "debug")  # ignore le paramètre 'debug' pour le cache
def task_avec_debug(x: int, debug: bool = False):
    return x * 2
```

### Cache policies disponibles

| Policy | Signification |
|--------|---------------|
| `DEFAULT` | `INPUTS + TASK_SOURCE + RUN_ID` |
| `INPUTS` | Arguments d'entrée uniquement |
| `TASK_SOURCE` | Code source de la task |
| `RUN_ID` | ID du run parent |
| `NONE` | Pas de cache |

### Cache distribué

Par défaut, le cache est local (`~/.prefect/storage/`). Pour du multi-machine :

```python
from prefect_aws import S3Bucket
from prefect import task

s3 = S3Bucket(bucket_name="mon-bucket")
s3.save("mon-cache")

@task(cache_policy=INPUTS, result_storage=s3)
def distributed_task(x: int):
    return x
```

---

## 5. Event-Driven — Events & Automations

### Émettre un event

```python
from prefect.events import emit_event

emit_event(
    event="mon.event.name",
    resource={
        "prefect.resource.id": "mon.resource.123",
        "prefect.resource.name": "Mon Resource",
    },
    payload={"key": "value"},
)
```

### Automations

Une **Automation** = Trigger + Action. Le trigger écoute un event, l'action exécute un deployment.

```python
from prefect.automations import Automation
from prefect.events.schemas.automations import EventTrigger
from prefect.events.actions import RunDeployment

Automation(
    name="react-to-news",
    trigger=EventTrigger(
        expect=["news.batch.available"],
        posture="Reactive",  # ou "Proactive"
        threshold=1,
        within=0,
    ),
    actions=[RunDeployment(deployment_id="...")],
).create()
```

### Deployment Triggers (plus simple)

```python
from prefect import flow

@flow
def mon_flow(payload: dict):
    print(payload)

mon_flow.serve(
    name="mon-deployment",
    triggers=[
        {
            "type": "event",
            "match": {"prefect.resource.id": "mon.resource.*"},
            "expect": ["mon.event.name"],
            "parameters": {"payload": "{{ event.payload }}"},
        }
    ],
)
```

### Webhooks (Prefect Cloud)

```bash
# Créer un webhook via UI → reçoit un URL unique
# Envoyer des données :
curl -X POST https://api.prefect.cloud/hooks/XYZ \
  -d "model_id=my_model"
```

---

## 6. Déploiement

### `flow.serve()` — Static Infrastructure

Le plus simple. Un processus long qui poll le serveur Prefect.

```python
from prefect import flow

@flow
def mon_flow():
    pass

mon_flow.serve(
    name="mon-deployment",
    interval=300,           # toutes les 5 min
    cron="0 9 * * 1-5",     # ou cron
    tags=["prod", "trading"],
)
```

### `flow.deploy()` — Dynamic Infrastructure (Work Pools)

Pour provisionner dynamiquement de l'infra (Docker, K8s, serverless).

```python
mon_flow.deploy(
    name="mon-deployment",
    work_pool_name="docker-pool",
    image="mon-image:latest",
    cron="0 */6 * * *",
)
```

### Work Pools

| Type | Description | Worker requis ? |
|------|-------------|-----------------|
| **Process** | Subprocess local | Oui |
| **Docker** | Conteneur Docker | Oui |
| **Kubernetes** | Job K8s | Oui |
| **AWS ECS Push** | Fargate/ECS | Non (push) |
| **GCP Cloud Run Push** | Cloud Run | Non (push) |
| **Prefect Managed** | Infra gérée par Prefect | Non |

### Workers

Un worker est un processus qui poll un work pool et lance les runs.

```bash
# Créer un work pool
prefect work-pool create --type docker mon-pool

# Démarrer un worker
prefect worker start --pool mon-pool

# Lister les pools
prefect work-pool ls
```

### Lancer un deployment

```bash
# CLI
prefect deployment run mon-flow/mon-deployment

# Python
from prefect.deployments import run_deployment
run_deployment("mon-flow/mon-deployment", parameters={"x": 1})
```

---

## 7. Workflows Interactives (Human-in-the-Loop)

### Pause pour input

```python
from prefect import flow
from prefect.flow_runs import pause_flow_run
from pydantic import BaseModel

class ApprovalInput(BaseModel):
    approved: bool
    comment: str = ""

@flow
def approval_workflow():
    # ... faire du travail ...
    
    result = pause_flow_run(wait_for_input=ApprovalInput)
    
    if result.approved:
        print(f"Approuvé: {result.comment}")
    else:
        print("Rejeté")
```

Dans l'UI Prefect, un bouton "Resume" apparaît avec un formulaire type-safe.

### Send / Receive (temps réel sans pause)

```python
from prefect import flow
from prefect.input.run_input import receive_input, send_input

@flow
async def chatbot_flow():
    async for message in receive_input(str, timeout=None):
        response = f"Bot: Tu as dit '{message}'"
        await message.respond(response)
```

---

## 8. Async / Await

Prefect supporte nativement `async/await`.

```python
import asyncio
from prefect import flow, task

@task
async def async_fetch(url: str):
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

@flow
async def async_flow():
    results = await asyncio.gather(
        async_fetch("https://api.example.com/a"),
        async_fetch("https://api.example.com/b"),
    )
    return results

# Lancer
asyncio.run(async_flow())
```

---

## 9. Patterns Avancés

### Transactional Semantics

```python
from prefect import flow, task
from prefect.transactions import transaction

@task
def write_to_db(data: dict):
    pass

@task
def rollback_db():
    pass

@flow
def transactional_flow():
    with transaction():
        write_to_db({"key": "value"})
        # Si une exception survient ici, le rollback est appelé
```

### Variables (state global)

```python
from prefect.variables import Variable

await Variable.set("mon-cle", "ma-valeur")
value = await Variable.get("mon-cle")
```

### Secrets

```python
from prefect.blocks.system import Secret

secret = Secret.load("mon-secret")
print(secret.get())
```

### Custom Blocks

```python
from prefect.blocks.core import Block

class TradingConfig(Block):
    _block_type_name = "Trading Config"
    _logo_url = "..."
    
    api_key: str
    max_trade: float = 500.0

# Sauvegarder
config = TradingConfig(api_key="xxx", max_trade=1000)
config.save("ma-config")

# Charger ailleurs
config = TradingConfig.load("ma-config")
```

---

## 10. Anti-Patterns

| ❌ Anti-Pattern | ✅ Solution |
|----------------|-------------|
| Flows < 100ms de travail | Utiliser une task dans un flow existant |
| Passer de l'état en mémoire entre flows | Écrire dans une DB / Variable / Block |
| Créer un flow par task | Grouper les tasks dans un flow avec subflows |
| Ne pas utiliser `log_prints=True` | Toujours activer pour le debug |
| Hardcoder les credentials | Utiliser Prefect Blocks / Variables |
| Ignorer les states | Utiliser `return_state=True` pour le control flow |
| Un seul work pool pour tout | Séparer par environnement (dev/staging/prod) |
| Polling actif dans un flow | Utiliser `.serve()` schedule ou events |

---

## 11. Référence Rapide CLI

```bash
# Server
prefect server start                    # Démarrer le serveur local
prefect server database reset -y        # Reset la DB

# Flows
prefect flow ls                         # Lister les flows
prefect flow-run ls                     # Lister les runs
prefect flow-run logs <ID>              # Voir les logs
prefect flow-run cancel <ID>            # Annuler un run

# Deployments
prefect deployment ls                   # Lister les deployments
prefect deployment run <name>           # Lancer un deployment
prefect deployment pause <name>         # Pauser
prefect deployment resume <name>        # Reprendre

# Work Pools
prefect work-pool create --type docker mon-pool
prefect work-pool ls
prefect worker start --pool mon-pool

# Config
prefect config set PREFECT_API_URL=http://localhost:4200/api
prefect config view                       # Voir la config
```

---

## 12. Ressources

- **Doc officielle** : https://docs.prefect.io/v3/get-started
- **Events** : https://docs.prefect.io/v3/concepts/events
- **Automations** : https://docs.prefect.io/v3/concepts/automations
- **Caching** : https://docs.prefect.io/v3/concepts/caching
- **States** : https://docs.prefect.io/v3/concepts/states
- **Deployments** : https://docs.prefect.io/v3/concepts/deployments
- **Work Pools** : https://docs.prefect.io/v3/concepts/work-pools
- **Interactive** : https://docs.prefect.io/v3/advanced/interactive
- **MCP Server** : https://docs.prefect.io/v3/how-to-guides/ai/use-prefect-mcp-server
