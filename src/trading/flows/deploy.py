"""Script de déploiement Prefect — crée les deployments et automations."""

import logging

from prefect import serve

from trading.flows.ingestion_flow import ingestion_flow
from trading.flows.sentiment_flow import sentiment_analysis_flow
from trading.flows.strategy_flow import strategy_execution_flow
from trading.flows.command_flow import command_processing_flow
from trading.flows.metrics_flow import metrics_flow
from trading.flows.notifications_flow import notifications_flow
from trading.flows.validation_flow import validation_flow
from trading.flows.fusion_training_flow import fusion_training_flow
from trading.flows.signal_training_flow import signal_training_flow

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("Création des deployments Prefect...")
    serve(
        # Deployments avec schedules
        ingestion_flow.to_deployment(name="ingestion-every-2min", interval=120),
        metrics_flow.to_deployment(name="metrics-hourly", interval=3600),
        notifications_flow.to_deployment(name="notifications-daily", cron="0 20 * * *"),
        validation_flow.to_deployment(name="validation-every-4h", interval=14400),
        fusion_training_flow.to_deployment(name="fusion-training-daily", cron="0 2 * * *"),
        signal_training_flow.to_deployment(name="signal-training-daily", cron="0 3 * * *"),
        # Deployments sans schedule (event-driven / on-demand)
        sentiment_analysis_flow.to_deployment(name="sentiment-on-demand"),
        strategy_execution_flow.to_deployment(name="strategy-on-demand"),
        command_processing_flow.to_deployment(name="command-on-demand"),
    )
