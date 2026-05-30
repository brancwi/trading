# Prefect — Patterns Avancés

## Interactive Workflows (Human-in-the-Loop)

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
    result = pause_flow_run(wait_for_input=ApprovalInput)
    if result.approved:
        print(f"Approuvé: {result.comment}")
```

Dans l'UI Prefect, un bouton "Resume" apparaît avec un formulaire type-safe.

### Send / Receive (temps réel sans pause)

```python
from prefect import flow
from prefect.input.run_input import receive_input

@flow
async def chatbot_flow():
    async for message in receive_input(str, timeout=None):
        await message.respond(f"Bot: {message}")
```

## Async / Await

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

asyncio.run(async_flow())
```

## Transactions

```python
from prefect import flow, task
from prefect.transactions import transaction

@task
def write_to_db(data: dict):
    pass

@flow
def transactional_flow():
    with transaction():
        write_to_db({"key": "value"})
        # Rollback automatique si exception
```

## Variables (state global)

```python
from prefect.variables import Variable

await Variable.set("mon-cle", "ma-valeur")
value = await Variable.get("mon-cle")
```

## Secrets

```python
from prefect.blocks.system import Secret
secret = Secret.load("mon-secret")
print(secret.get())
```

## Custom Blocks

```python
from prefect.blocks.core import Block

class TradingConfig(Block):
    _block_type_name = "Trading Config"
    api_key: str
    max_trade: float = 500.0

config = TradingConfig(api_key="xxx", max_trade=1000)
config.save("ma-config")
config = TradingConfig.load("ma-config")
```
