# Prefect — Fondamentaux

## Flows

Un flow est une fonction Python décorée avec `@flow`. C'est l'unité d'orchestration.

```python
from prefect import flow

@flow(log_prints=True)
def mon_flow(name: str):
    return f"Hello {name}"

mon_flow("world")  # Appel normal
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

## Tasks

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

## 3 façons d'invoquer une task

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

## Subflows

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

## States (États)

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
    state = risky_task.submit(fail=True, return_state=True)
    
    if state.is_failed():
        print("La task a échoué, mais le flow continue")
    
    result = state.result()  # peut lever exception si Failed
```

### State Change Hooks

```python
from prefect import Task
from prefect.states import State
from prefect.client.schemas.objects import TaskRun

def on_task_failure(task: Task, run: TaskRun, state: State):
    print(f"Task {task.name} failed: {state.message}")

@task(on_failure=[on_task_failure])
def ma_task():
    raise ValueError("oops")
```

## Concurrence & Parallélisme

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
    f1 = slow_task.submit(1)
    f2 = slow_task.submit(2)
    f3 = slow_task.submit(3)
    return f1.result(), f2.result(), f3.result()
```

### Dynamic Task Mapping

```python
from prefect import flow, task

@task
def process_item(item: str) -> str:
    return item.upper()

@flow
def map_flow():
    items = ["a", "b", "c"]
    results = process_item.map(items)  # retourne une liste
    return results
```

### Dépendances explicites avec `wait_for`

```python
@flow
def dependency_flow():
    a = step_a.submit()
    b = step_b.submit()
    c = step_c.submit(wait_for=[a, b])
    return c.result()
```

## Caching

### Cache basique

```python
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE, DEFAULT
from datetime import timedelta

@task(persist_result=True)
def expensive_computation(x: int) -> int:
    import time; time.sleep(10)
    return x * 2

# DEFAULT = INPUTS + TASK_SOURCE + RUN_ID

@task(cache_policy=INPUTS, cache_expiration=timedelta(minutes=5))
def cache_5min(x: int) -> int:
    return x * 2

@task(cache_policy=INPUTS - "debug")
def task_avec_debug(x: int, debug: bool = False):
    return x * 2
```

### Cache policies

| Policy | Signification |
|--------|---------------|
| `DEFAULT` | `INPUTS + TASK_SOURCE + RUN_ID` |
| `INPUTS` | Arguments d'entrée uniquement |
| `TASK_SOURCE` | Code source de la task |
| `RUN_ID` | ID du run parent |
| `NONE` | Pas de cache |

### Cache distribué

```python
from prefect_aws import S3Bucket
from prefect import task

s3 = S3Bucket(bucket_name="mon-bucket")
s3.save("mon-cache")

@task(cache_policy=INPUTS, result_storage=s3)
def distributed_task(x: int):
    return x
```
