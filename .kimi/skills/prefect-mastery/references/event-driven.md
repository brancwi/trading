# Prefect — Event-Driven

## Émettre un event

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

## Automations

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

## Deployment Triggers (plus simple)

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

## Webhooks (Prefect Cloud)

```bash
# Créer un webhook via UI → reçoit un URL unique
curl -X POST https://api.prefect.cloud/hooks/XYZ \
  -d "model_id=my_model"
```

## Pattern : Event Listener Externe

Prefect n'est pas un serveur websocket natif. Pour écouter des sources temps réel :

```python
# Service asyncio séparé qui écoute et émet des events
import asyncio
from prefect.events import emit_event

async def listener():
    while True:
        data = await websocket.recv()
        emit_event(
            event="data.received",
            resource={"prefect.resource.id": "websocket.1"},
            payload=data,
        )
```
