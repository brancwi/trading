"""Taux de change EUR/USD dynamique — récupération temps réel + cache.

Le taux est récupéré via Yahoo Finance (yfinance) et mis en cache
pendant 5 minutes pour éviter de surcharger l'API.
Le taux utilisé est logué dans la DB temporelle à chaque trade pour
permettre l'audit et le replay exact des exécutions.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Cache global (_CACHE[rate, expires_at])
_CACHE: dict[str, Any] = {"rate": None, "expires_at": None, "source": None}
_DEFAULT_RATE = 1.08
_CACHE_TTL_SECONDS = 300  # 5 minutes


def get_fx_eur_usd(
    force_refresh: bool = False,
    ttl_seconds: int = _CACHE_TTL_SECONDS,
) -> float:
    """Retourne le taux EUR/USD en temps réel (1 EUR = X USD).

    Le résultat est mis en cache *ttl_seconds* pour limiter les appels
    réseau.  En cas d'erreur on retombe sur le dernier taux connu,
    puis sur le taux par défaut (1.08).
    """
    now = datetime.utcnow()

    # 1) Cache valide ?
    if not force_refresh and _CACHE["expires_at"] and now < _CACHE["expires_at"]:
        rate = _CACHE["rate"]
        logger.debug("[FX] Cache hit: 1 EUR = %.6f USD", rate)
        return float(rate)

    # 2) Récupération temps réel via yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker("EURUSD=X")
        hist = ticker.history(period="1d", interval="1m")
        if hist.empty:
            raise ValueError("yfinance returned empty history for EURUSD=X")
        rate = float(hist["Close"].iloc[-1])
        _CACHE["rate"] = rate
        _CACHE["expires_at"] = now + timedelta(seconds=ttl_seconds)
        _CACHE["source"] = "yfinance"
        logger.info("[FX] Live rate fetched: 1 EUR = %.6f USD (cache %ds)", rate, ttl_seconds)
        return rate
    except Exception as e:
        logger.warning("[FX] Failed to fetch live rate: %s", e)

    # 3) Fallback sur le dernier taux connu
    if _CACHE["rate"] is not None:
        rate = _CACHE["rate"]
        logger.info("[FX] Fallback to cached rate: 1 EUR = %.6f USD", rate)
        return float(rate)

    # 4) Fallback ultime
    logger.warning("[FX] Fallback to default rate: 1 EUR = %.6f USD", _DEFAULT_RATE)
    return _DEFAULT_RATE


def log_fx_rate(rate: float, portfolio_id: str = "") -> None:
    """Logue le taux FX utilisé dans la DB temporelle pour audit."""
    try:
        from trading.monitoring.service import MonitorService
        MonitorService.log_performance(
            metric_name="fx_eur_usd",
            value=rate,
            unit="USD",
            tags={
                "portfolio_id": portfolio_id,
                "source": _CACHE.get("source", "unknown"),
                "timestamp_used": datetime.utcnow().isoformat(),
            },
        )
    except Exception as e:
        logger.debug("[FX] Failed to log fx rate: %s", e)


def invalidate_cache() -> None:
    """Invalide le cache FX (utile pour les tests)."""
    _CACHE["rate"] = None
    _CACHE["expires_at"] = None
    _CACHE["source"] = None
    logger.info("[FX] Cache invalidated")


if __name__ == "__main__":
    rate = get_fx_eur_usd()
    print(f"1 EUR = {rate:.6f} USD")
    log_fx_rate(rate)
