"""Flow indépendant — entraînement quotidien du modèle de fusion."""

import logging

from prefect import flow, task

from trading.sentiment.fusion_model import FusionModel
from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)
monitor = MonitorService()


@task(retries=1, retry_delay_seconds=30)
def train_fusion_model() -> dict:
    """Entraîne le FusionModel sur les annotations humaines et validées."""
    model = FusionModel()
    success = model.train()
    info = model.info()

    monitor.record_metric(
        name="fusion_model.trained",
        value=1.0 if success else 0.0,
        unit="boolean",
        source="sentiment",
    )
    monitor.log_event(
        event_type="FUSION_MODEL_TRAINED" if success else "FUSION_MODEL_TRAIN_FAILED",
        actor="fusion_training_flow",
        details_json={"trained": success, "weights": info.get("weights")},
    )

    return {"trained": success, "info": info}


@flow(name="fusion_training_flow", log_prints=True)
def fusion_training_flow() -> dict:
    """Flow d'entraînement fusion — déclenché une fois par jour."""
    result = train_fusion_model()
    logger.info(f"Fusion training flow terminé: {result}")
    return result


if __name__ == "__main__":
    fusion_training_flow()
