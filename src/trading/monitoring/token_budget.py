"""Token Budget Manager — Circuit-breaker et rate limiting pour les appels LLM.

Usage:
    from trading.monitoring.token_budget import TokenBudgetManager
    
    budget = TokenBudgetManager(daily_budget_usd=5.0)
    
    if budget.can_spend(estimated_cost=0.001):
        result = call_llm(...)
        budget.record_spend(actual_cost=0.0005)
    else:
        logger.warning("Budget épuisé — fallback vers règles simples")
        return default_decision()
"""

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from trading.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Configuration par défaut ───────────────────────────────────────
DEFAULT_DAILY_BUDGET_USD = getattr(settings, "monitoring_alert_cost_daily_usd", 5.0)
DEFAULT_HOURLY_BUDGET_USD = DEFAULT_DAILY_BUDGET_USD / 24
DEFAULT_MINUTE_BUDGET_USD = DEFAULT_HOURLY_BUDGET_USD / 60

# Seuils d'alerte (en % du budget)
WARN_THRESHOLD_PCT = 60
BLOCK_THRESHOLD_PCT = 80
CRITICAL_THRESHOLD_PCT = 95

# Rate limiting
MAX_CALLS_PER_MINUTE = 10
MAX_CALLS_PER_HOUR = 100

# Cooldown après blocage (secondes)
BLOCK_COOLDOWN_SECONDS = 300  # 5 minutes


class TokenBudgetManager:
    """Gère le budget token avec circuit-breaker intégré.
    
    Features:
      - Budget quotidien/heure/minute configurable
      - Rate limiting par minute et heure
      - Circuit-breaker avec cooldown
      - Persistance SQLite (survie aux redémarrages)
      - Alertes à 60%, 80%, 95%
      - Fallback automatique quand budget épuisé
    """

    _instance: Optional["TokenBudgetManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        daily_budget_usd: float = DEFAULT_DAILY_BUDGET_USD,
        db_path: Optional[str] = None,
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.daily_budget = daily_budget_usd
        self.hourly_budget = daily_budget_usd / 24
        self.minute_budget = self.hourly_budget / 60

        # Rate limiting
        self._calls_minute: list[float] = []
        self._calls_hour: list[float] = []
        self._rate_lock = threading.Lock()

        # Circuit breaker
        self._blocked_until: float = 0.0
        self._block_reason: Optional[str] = None
        self._alert_sent: dict[str, bool] = {}  # date -> alert_sent

        # DB persistance
        if db_path is None:
            data_dir = Path("/app/data")
            if not data_dir.exists():
                data_dir = Path.home() / ".trading"
                data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "token_budget.db")
        
        self.db_path = db_path
        self._init_db()
        self._initialized = True

        logger.info(
            "[TokenBudget] Initialisé — daily=$%.2f, hourly=$%.4f, minute=$%.4f, db=%s",
            self.daily_budget, self.hourly_budget, self.minute_budget, self.db_path
        )

    # ── DB ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        """Crée les tables SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_spends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    date TEXT NOT NULL,
                    hour TEXT NOT NULL,
                    minute TEXT NOT NULL,
                    cost_usd REAL NOT NULL,
                    model TEXT,
                    provider TEXT,
                    triggered_by TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_date ON token_spends(date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON token_spends(timestamp)
            """)
            conn.commit()

    def _get_spent(self, date_str: str, hour_str: Optional[str] = None, minute_str: Optional[str] = None) -> float:
        """Récupère le montant dépensé pour une période."""
        with sqlite3.connect(self.db_path) as conn:
            if minute_str:
                cursor = conn.execute(
                    "SELECT SUM(cost_usd) FROM token_spends WHERE date = ? AND hour = ? AND minute = ?",
                    (date_str, hour_str, minute_str)
                )
            elif hour_str:
                cursor = conn.execute(
                    "SELECT SUM(cost_usd) FROM token_spends WHERE date = ? AND hour = ?",
                    (date_str, hour_str)
                )
            else:
                cursor = conn.execute(
                    "SELECT SUM(cost_usd) FROM token_spends WHERE date = ?",
                    (date_str,)
                )
            result = cursor.fetchone()[0]
            return result if result else 0.0

    # ── Rate limiting ───────────────────────────────────────────────

    def _check_rate_limit(self) -> tuple[bool, Optional[str]]:
        """Vérifie les rate limits. Retourne (ok, reason)."""
        now = time.time()
        
        with self._rate_lock:
            # Nettoyer les anciens appels
            self._calls_minute = [t for t in self._calls_minute if now - t < 60]
            self._calls_hour = [t for t in self._calls_hour if now - t < 3600]

            # Vérifier minute
            if len(self._calls_minute) >= MAX_CALLS_PER_MINUTE:
                return False, f"Rate limit: {len(self._calls_minute)} calls/minute (max {MAX_CALLS_PER_MINUTE})"

            # Vérifier heure
            if len(self._calls_hour) >= MAX_CALLS_PER_HOUR:
                return False, f"Rate limit: {len(self._calls_hour)} calls/hour (max {MAX_CALLS_PER_HOUR})"

        return True, None

    def _record_call(self) -> None:
        """Enregistre un appel pour le rate limiting."""
        now = time.time()
        with self._rate_lock:
            self._calls_minute.append(now)
            self._calls_hour.append(now)

    # ── Circuit breaker ─────────────────────────────────────────────

    def _check_circuit_breaker(self) -> tuple[bool, Optional[str]]:
        """Vérifie si le circuit breaker est ouvert."""
        now = time.time()
        
        if now < self._blocked_until:
            remaining = int(self._blocked_until - now)
            return False, f"Circuit breaker: bloqué encore {remaining}s ({self._block_reason})"
        
        return True, None

    def _open_circuit(self, reason: str, duration: int = BLOCK_COOLDOWN_SECONDS) -> None:
        """Ouvre le circuit breaker."""
        self._blocked_until = time.time() + duration
        self._block_reason = reason
        logger.error("[TokenBudget] CIRCUIT BREAKER OUVERT: %s (durée: %ds)", reason, duration)

    # ── Budget checking ─────────────────────────────────────────────

    def can_spend(self, estimated_cost: float = 0.0) -> tuple[bool, Optional[str]]:
        """Vérifie si on peut dépenser. Retourne (ok, reason).
        
        Usage:
            ok, reason = budget.can_spend(estimated_cost=0.001)
            if not ok:
                logger.warning("Cannot spend: %s", reason)
                return fallback()
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%Y-%m-%d-%H")
        minute_str = now.strftime("%Y-%m-%d-%H-%M")

        # 1. Circuit breaker
        ok, reason = self._check_circuit_breaker()
        if not ok:
            return False, reason

        # 2. Rate limiting
        ok, reason = self._check_rate_limit()
        if not ok:
            return False, reason

        # 3. Budget minute
        spent_minute = self._get_spent(date_str, hour_str, minute_str)
        if spent_minute + estimated_cost > self.minute_budget:
            self._open_circuit(f"Budget minute épuisé: ${spent_minute:.6f}/${self.minute_budget:.6f}")
            return False, f"Budget minute épuisé: ${spent_minute:.6f}/${self.minute_budget:.6f}"

        # 4. Budget heure
        spent_hour = self._get_spent(date_str, hour_str)
        if spent_hour + estimated_cost > self.hourly_budget:
            pct = (spent_hour / self.hourly_budget) * 100
            if pct >= CRITICAL_THRESHOLD_PCT:
                self._open_circuit(f"Budget heure critique: {pct:.1f}%")
                return False, f"Budget heure critique: {pct:.1f}%"
            elif pct >= BLOCK_THRESHOLD_PCT:
                return False, f"Budget heure à {pct:.1f}% — blocage"
            elif pct >= WARN_THRESHOLD_PCT:
                logger.warning("[TokenBudget] Budget heure à %.1f%%", pct)

        # 5. Budget jour
        spent_day = self._get_spent(date_str)
        if spent_day + estimated_cost > self.daily_budget:
            pct = (spent_day / self.daily_budget) * 100
            self._open_circuit(f"Budget journalier épuisé: {pct:.1f}%")
            return False, f"Budget journalier épuisé: ${spent_day:.2f}/${self.daily_budget:.2f}"
        
        if spent_day + estimated_cost > self.daily_budget * WARN_THRESHOLD_PCT / 100:
            pct = ((spent_day + estimated_cost) / self.daily_budget) * 100
            if date_str not in self._alert_sent:
                self._alert_sent[date_str] = False
            if not self._alert_sent.get(date_str, False) and pct >= WARN_THRESHOLD_PCT:
                logger.warning("[TokenBudget] ALERTE: Budget journalier à %.1f%% (%.4f$/%.2f$)", 
                              pct, spent_day + estimated_cost, self.daily_budget)
                self._alert_sent[date_str] = True

        return True, None

    def record_spend(
        self,
        cost_usd: float,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        triggered_by: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Enregistre une dépense."""
        now = datetime.now(timezone.utc)
        timestamp = time.time()
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%Y-%m-%d-%H")
        minute_str = now.strftime("%Y-%m-%d-%H-%M")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO token_spends 
                   (timestamp, date, hour, minute, cost_usd, model, provider, triggered_by, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    timestamp,
                    date_str,
                    hour_str,
                    minute_str,
                    cost_usd,
                    model,
                    provider,
                    triggered_by,
                    json.dumps(metadata) if metadata else None,
                )
            )
            conn.commit()

        self._record_call()
        
        # Vérifier si on atteint un seuil critique après cette dépense
        spent_day = self._get_spent(date_str)
        pct = (spent_day / self.daily_budget) * 100
        if pct >= CRITICAL_THRESHOLD_PCT:
            logger.error("[TokenBudget] CRITIQUE: Budget à %.1f%% (%.4f$/%.2f$)", 
                        pct, spent_day, self.daily_budget)
            self._open_circuit(f"Budget journalier critique: {pct:.1f}%", duration=3600)  # 1h cooldown

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Retourne les statistiques de consommation."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        hour_str = now.strftime("%Y-%m-%d-%H")
        minute_str = now.strftime("%Y-%m-%d-%H-%M")

        spent_day = self._get_spent(date_str)
        spent_hour = self._get_spent(date_str, hour_str)
        spent_minute = self._get_spent(date_str, hour_str, minute_str)

        return {
            "daily_budget": self.daily_budget,
            "hourly_budget": self.hourly_budget,
            "minute_budget": self.minute_budget,
            "spent_today": spent_day,
            "spent_this_hour": spent_hour,
            "spent_this_minute": spent_minute,
            "daily_pct": (spent_day / self.daily_budget * 100) if self.daily_budget > 0 else 0,
            "hourly_pct": (spent_hour / self.hourly_budget * 100) if self.hourly_budget > 0 else 0,
            "minute_pct": (spent_minute / self.minute_budget * 100) if self.minute_budget > 0 else 0,
            "circuit_breaker_open": time.time() < self._blocked_until,
            "blocked_until": self._blocked_until if time.time() < self._blocked_until else None,
            "block_reason": self._block_reason if time.time() < self._blocked_until else None,
            "calls_last_minute": len(self._calls_minute),
            "calls_last_hour": len(self._calls_hour),
        }

    def get_report(self) -> str:
        """Génère un rapport texte."""
        stats = self.get_stats()
        status = "🔴 BLOQUÉ" if stats["circuit_breaker_open"] else "🟢 OK"
        
        return f"""[TokenBudget] {status}
  Daily:   ${stats['spent_today']:.4f} / ${stats['daily_budget']:.2f} ({stats['daily_pct']:.1f}%)
  Hourly:  ${stats['spent_this_hour']:.4f} / ${stats['hourly_budget']:.4f} ({stats['hourly_pct']:.1f}%)
  Minute:  ${stats['spent_this_minute']:.4f} / ${stats['minute_budget']:.4f} ({stats['minute_pct']:.1f}%)
  Calls:   {stats['calls_last_minute']}/min, {stats['calls_last_hour']}/hour
  Circuit: {'OPEN — ' + stats['block_reason'] if stats['circuit_breaker_open'] else 'Closed'}
"""

    def reset_circuit(self) -> None:
        """Réinitialise manuellement le circuit breaker."""
        self._blocked_until = 0.0
        self._block_reason = None
        logger.info("[TokenBudget] Circuit breaker réinitialisé manuellement")


# ── Singleton global ──────────────────────────────────────────────

_token_budget: Optional[TokenBudgetManager] = None


def get_token_budget() -> TokenBudgetManager:
    """Retourne l'instance globale du TokenBudgetManager."""
    global _token_budget
    if _token_budget is None:
        _token_budget = TokenBudgetManager()
    return _token_budget
