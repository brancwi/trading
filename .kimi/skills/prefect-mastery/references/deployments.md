# Prefect — Déploiement

## `flow.serve()` — Static Infrastructure

Le plus simple. Un processus long qui poll le serveur Prefect.

```python
from prefect import flow

@flow
def mon_flow():
    pass

mon_flow.serve(
    name="mon-deployment",
    interval=300,
    cron="0 9 * * 1-5",
    tags=["prod"],
)
```

## `flow.deploy()` — Dynamic Infrastructure (Work Pools)

Pour provisionner dynamiquement de l'infra (Docker, K8s, serverless).

```python
mon_flow.deploy(
    name="mon-deployment",
    work_pool_name="docker-pool",
    image="mon-image:latest",
    cron="0 */6 * * *",
)
```

## Work Pools

| Type | Description | Worker requis ? |
|------|-------------|-----------------|
| **Process** | Subprocess local | Oui |
| **Docker** | Conteneur Docker | Oui |
| **Kubernetes** | Job K8s | Oui |
| **AWS ECS Push** | Fargate/ECS | Non (push) |
| **GCP Cloud Run Push** | Cloud Run | Non (push) |
| **Prefect Managed** | Infra gérée par Prefect | Non |

## Workers

Un worker est un processus qui poll un work pool et lance les runs.

```bash
prefect work-pool create --type docker mon-pool
prefect worker start --pool mon-pool
prefect work-pool ls
```

## Lancer un deployment

```bash
prefect deployment run mon-flow/mon-deployment
```

```python
from prefect.deployments import run_deployment
run_deployment("mon-flow/mon-deployment", parameters={"x": 1})
```

## Concurrency Limit

```python
from prefect.client.schemas.objects import ConcurrencyLimitConfig

my_flow.serve(
    name="mon-deployment",
    global_limit=ConcurrencyLimitConfig(limit=3),
)
```
