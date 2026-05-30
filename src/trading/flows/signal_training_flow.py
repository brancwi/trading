"""Flow indépendant — entraînement du SignalModel v1 (Random Forest)."""

import logging

from prefect import flow, task

from trading.ml.signal_model import SignalModel
from trading.ml.backtest import BacktestEngine
from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)
monitor = MonitorService()


@task(retries=1, retry_delay_seconds=30)
def train_signal_model() -> dict:
    """Entraîne le SignalModel sur l'historique marché + sentiment."""
    model = SignalModel()
    result = model.train()

    if result.get("trained"):
        monitor.record_metric(
            name="signal_model.trained",
            value=1.0,
            unit="boolean",
            source="ml",
        )
        monitor.record_metric(
            name="signal_model.cv_accuracy",
            value=result.get("mean_cv_accuracy", 0.0),
            unit="ratio",
            source="ml",
        )
        monitor.log_event(
            event_type="SIGNAL_MODEL_TRAINED",
            actor="signal_training_flow",
            details_json={
                "samples": result.get("samples"),
                "cv_accuracies": result.get("cv_accuracies"),
                "feature_importances": result.get("feature_importances"),
            },
        )
    else:
        monitor.record_metric(
            name="signal_model.trained",
            value=0.0,
            unit="boolean",
            source="ml",
        )
        monitor.log_event(
            event_type="SIGNAL_MODEL_TRAIN_FAILED",
            actor="signal_training_flow",
            details_json={"reason": result.get("reason")},
        )

    return result


@flow(name="signal_training_flow", log_prints=True)
def signal_training_flow() -> dict:
    """Flow d'entraînement SignalModel — déclenché une fois par jour."""
    result = train_signal_model()
    logger.info(f"Signal training flow terminé: {result}")
    return result


if __name__ == "__main__":
    signal_training_flow()
