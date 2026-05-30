"""Script de déploiement Prefect — crée les deployments et automations."""

import logging

from prefect import flow
from prefect.deployments import run_deployment

from trading.flows.ingestion_flow import ingestion_flow
from trading.flows.sentiment_flow import sentiment_analysis_flow
from trading.flows.strategy_flow import strategy_execution_flow
from trading.flows.command_flow import command_processing_flow
from trading.flows.metrics_flow import metrics_flow
from trading.flows.notifications_flow import notifications_flow

logger = logging.getLogger(__name__)


@flow(name="deploy_all", log_prints=True)
def deploy_all():
    """Crée tous les deployments et automations."""
    # Deployments avec schedules
    ingestion_flow.serve(
        name="ingestion-every-2min",
        interval=120,  # 2 minutes
    )
    metrics_flow.serve(
        name="metrics-hourly",
        interval=3600,  # 1 heure
    )
    notifications_flow.serve(
        name="notifications-daily",
        cron="0 20 * * *",  # 20h00 tous les jours
    )

    # Deployments sans schedule (event-driven)
    sentiment_analysis_flow.serve(name="sentiment-on-demand")
    strategy_execution_flow.serve(name="strategy-on-demand")
    command_processing_flow.serve(name="command-on-demand")

    logger.info("All deployments created. Go to Prefect UI to see them.")


if __name__ == "__main__":
    deploy_all()
