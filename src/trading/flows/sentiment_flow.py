"""Flow indépendant — analyse de sentiment et génération de signaux."""

import logging
import time

from prefect import flow, task

from trading.core.database import db_session
from trading.sentiment.analyzer import SentimentAnalyzer
from trading.monitoring.service import MonitorService
from trading.events.emitters import emit_signal_generated

logger = logging.getLogger(__name__)
monitor = MonitorService()


@task(retries=1, retry_delay_seconds=30)
def analyze_unprocessed_news() -> int:
    """Analyse les news non traitées et génère des signaux."""
    start = time.time()
    analyzer = SentimentAnalyzer()
    with db_session() as db:
        count = analyzer.process_unprocessed_news(db)
    latency = time.time() - start

    # Monitoring
    monitor.record_metric("sentiment.inference_latency", latency, unit="seconds", source="sentiment")
    monitor.record_metric("sentiment.signals_generated", float(count), unit="count", source="sentiment")
    monitor.log_event(
        event_type="SENTIMENT_FLOW_COMPLETED",
        actor="sentiment_engine",
        details={"signals_generated": count, "latency_seconds": round(latency, 3)},
    )
    return count


@flow(name="sentiment_analysis_flow", log_prints=True)
def sentiment_analysis_flow() -> dict:
    """Flow d'analyse sentiment — déclenché par event news.batch.available."""
    signals_count = analyze_unprocessed_news()
    logger.info(f"Sentiment: {signals_count} signaux générés")
    return {"signals_generated": signals_count}


if __name__ == "__main__":
    sentiment_analysis_flow()
