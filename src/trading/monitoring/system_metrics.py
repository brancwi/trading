"""Collecteur de métriques système — CPU, mémoire, latence DB, GPU.

Insère automatiquement dans la table `performance_snapshot` de la DB
temporelle (TimescaleDB) pour le monitoring temps réel.
"""

import logging
import time
from datetime import datetime

from trading.monitoring.service import MonitorService

logger = logging.getLogger(__name__)


def collect_system_metrics() -> dict[str, float]:
    """Collecte les métriques système courantes."""
    metrics: dict[str, float] = {}

    # CPU & Mémoire
    try:
        import psutil
        metrics["cpu_percent"] = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        metrics["memory_percent"] = mem.percent
        metrics["memory_used_mb"] = mem.used / (1024 * 1024)
        metrics["memory_available_mb"] = mem.available / (1024 * 1024)
    except ImportError:
        logger.debug("psutil non installé — métriques CPU/mémoire ignorées")
    except Exception as e:
        logger.warning("Erreur métriques système: %s", e)

    # GPU (NVIDIA)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_utilization"] = float(util.gpu)
        metrics["gpu_memory_used_mb"] = mem_info.used / (1024 * 1024)
        metrics["gpu_memory_total_mb"] = mem_info.total / (1024 * 1024)
        pynvml.nvmlShutdown()
    except ImportError:
        logger.debug("pynvml non installé — métriques GPU ignorées")
    except Exception as e:
        logger.warning("Erreur métriques GPU: %s", e)

    # Latence DB principale
    try:
        from sqlalchemy import text
        from trading.core.database import engine
        t0 = time.perf_counter()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - t0) * 1000
        metrics["db_latency_ms"] = round(latency_ms, 2)
    except Exception as e:
        logger.warning("Erreur latence DB: %s", e)

    # Latence DB monitoring
    try:
        from sqlalchemy import text
        from trading.monitoring.database import engine as mon_engine
        t0 = time.perf_counter()
        with mon_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency_ms = (time.perf_counter() - t0) * 1000
        metrics["monitoring_db_latency_ms"] = round(latency_ms, 2)
    except Exception as e:
        logger.warning("Erreur latence monitoring DB: %s", e)

    return metrics


def log_system_metrics() -> None:
    """Collecte et persiste toutes les métriques système dans TimescaleDB."""
    metrics = collect_system_metrics()
    for name, value in metrics.items():
        unit = _guess_unit(name)
        MonitorService.log_performance(
            metric_name=name,
            value=value,
            unit=unit,
            tags={"host": "localhost", "collector": "system_metrics"},
        )
    logger.info("[SystemMetrics] Logged %d metrics", len(metrics))


def _guess_unit(metric_name: str) -> str:
    if "percent" in metric_name or "utilization" in metric_name:
        return "%"
    if "_mb" in metric_name:
        return "MB"
    if "_ms" in metric_name:
        return "ms"
    return "count"


if __name__ == "__main__":
    log_system_metrics()
