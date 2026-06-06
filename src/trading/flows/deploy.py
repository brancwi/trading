"""Script de déploiement Prefect — crée les deployments sans démarrer de serveur."""

import asyncio
import logging

from prefect.client.orchestration import get_client
from prefect.client.schemas.actions import DeploymentScheduleCreate
from prefect.client.schemas.schedules import IntervalSchedule, CronSchedule

from trading.flows.ingestion_flow import ingestion_flow
from trading.flows.sentiment_flow import sentiment_analysis_flow
from trading.flows.strategy_flow import strategy_execution_flow
from trading.flows.command_flow import command_processing_flow
from trading.flows.metrics_flow import metrics_flow
from trading.flows.notifications_flow import notifications_flow
from trading.flows.validation_flow import validation_flow
from trading.flows.fusion_training_flow import fusion_training_flow
from trading.flows.signal_training_flow import signal_training_flow
from trading.flows.ml_signal_flow import ml_signal_generation_flow

logger = logging.getLogger(__name__)

DEPLOYMENTS = [
    ("ingestion-every-2min", ingestion_flow, {"interval": 120}),
    ("metrics-hourly", metrics_flow, {"interval": 3600}),
    ("notifications-daily", notifications_flow, {"cron": "0 20 * * *"}),
    ("validation-every-4h", validation_flow, {"interval": 14400}),
    ("fusion-training-daily", fusion_training_flow, {"cron": "0 2 * * *"}),
    ("signal-training-daily", signal_training_flow, {"cron": "0 3 * * *"}),
    ("sentiment-every-5min", sentiment_analysis_flow, {"interval": 300}),
    ("ml-signals-every-15min", ml_signal_generation_flow, {"interval": 900}),
    ("strategy-every-5min", strategy_execution_flow, {"interval": 300}),
    ("command-on-demand", command_processing_flow, {}),
]


def make_schedule(params):
    if "interval" in params:
        return IntervalSchedule(interval=params["interval"])
    elif "cron" in params:
        return CronSchedule(cron=params["cron"])
    return None


async def create_deployments():
    print("Connexion au serveur Prefect...")
    async with get_client() as client:
        # Supprimer les anciens deployments
        print("Suppression des anciens deployments...")
        deps = await client.read_deployments()
        for d in deps:
            await client.delete_deployment(d.id)
            print(f"  Supprimé: {d.name}")

        # Créer les nouveaux deployments
        print("Création des deployments...")
        for name, flow, params in DEPLOYMENTS:
            # Récupérer le flow_id depuis le serveur
            print(f"  Flow: {flow.name}...")
            flow_obj = await client.read_flow_by_name(flow.name)
            flow_id = flow_obj.id

            schedule = make_schedule(params)
            schedules = []
            if schedule:
                schedules.append(DeploymentScheduleCreate(schedule=schedule, active=True))

            await client.create_deployment(
                flow_id=flow_id,
                name=name,
                work_pool_name="default",
                schedules=schedules,
            )
            print(f"  Créé: {name}")

        print(f"{len(DEPLOYMENTS)} deployments créés avec succès!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(create_deployments())
