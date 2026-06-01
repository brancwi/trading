"""OpportunityAgent — découvre des opportunités non détectées par XGBoost.

Méthodes de recherche:
1. Mouvements de prix anormaux (gaps > 3%, volume spikes)
2. Rotations sectorielles (détection via corrélation inter-sectorielle)
3. Opportunités de paires (ex: Boeing↓ → Airbus↑)
4. Surveillance SL/TP des positions existantes
5. Screener technique sur tout l'univers

Usage:
    agent = OpportunityAgent()
    ops = agent.find_opportunities(db, portfolio_id="staging-ninja")
"""

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from trading.core.database import db_session
from trading.core.models import MarketData, Position, Portfolio

logger = logging.getLogger(__name__)

# Mapping opportunités de concurrence / relations
COMPETITOR_PAIRS = {
    # Défense / Aéronautique
    "BA": ["AIR.PA", "LMT"],
    "AIR.PA": ["BA", "LMT"],
    "LMT": ["BA", "AIR.PA", "RTX"],
    "RTX": ["LMT", "BA.L"],
    "BA.L": ["RHM.DE", "LDO.MI"],
    "RHM.DE": ["BA.L", "LDO.MI"],
    "LDO.MI": ["RHM.DE", "BA.L"],
    # Tech / Semiconductors
    "AMD": ["INTC", "NVDA"],
    "NVDA": ["AMD", "INTC"],
    "INTC": ["AMD", "NVDA", "QCOM"],
    "QCOM": ["INTC", "NVDA"],
    # Streaming / Entertainment
    "NFLX": ["DIS", "WBD"],
    "ROKU": ["NFLX", "DIS"],
    # Social / Tech
    "META": ["SNAP", "PINS"],
    # Rideshare
    "UBER": ["LYFT"],
    "LYFT": ["UBER"],
    # E-commerce / Fintech
    "SHOP": ["SQ", "PYPL"],
    # Cloud / Data
    "PLTR": ["SNOW", "DDOG", "NET"],
    "CRWD": ["OKTA", "NET", "S"],
    "DDOG": ["CRWD", "NET", "S"],
    # CDN / Cloud infra
    "FSLY": ["NET", "PLTR"],
    "NET": ["FSLY", "PLTR"],
    # Vidéoconf
    "ZM": ["MSFT", "GOOGL"],
}

# Secteurs défense (réaction aux événements géopolitiques)
DEFENSE_TICKERS = ["LMT", "RTX", "NOC", "GD", "BA.L", "RHM.DE", "LDO.MI", "SAF.PA", "HO.PA", "THALES.PA"]

# Secteurs énergie (réaction aux tensions au Moyen-Orient)
ENERGY_TICKERS = ["XOM", "CVX", "SHEL.L", "TTE.PA", "ENI.MI", "BP.L"]


class OpportunityAgent:
    """Recherche proactive d'opportunités."""

    def __init__(self, lookback_days: int = 5):
        self.lookback_days = lookback_days

    def find_opportunities(
        self,
        db: Session,
        portfolio_id: str,
    ) -> list[dict[str, Any]]:
        """Trouve toutes les opportunités pour un portfolio."""
        opportunities = []

        # 1. Positions existantes — SL/TP triggers
        opportunities.extend(self._check_positions_sl_tp(db, portfolio_id))

        # 2. Mouvements anormaux de prix
        opportunities.extend(self._find_price_anomalies(db))

        # 3. Opportunités de concurrence
        opportunities.extend(self._find_competitor_opportunities(db))

        # 4. Screener technique (RSI extrême, breakout)
        opportunities.extend(self._technical_screener(db))

        logger.info("[OpportunityAgent] %d opportunités trouvées", len(opportunities))
        return opportunities

    def _check_positions_sl_tp(
        self, db: Session, portfolio_id: str
    ) -> list[dict[str, Any]]:
        """Vérifie si des positions ont atteint leur SL ou TP."""
        ops = []
        positions = (
            db.query(Position)
            .filter(Position.portfolio_id == portfolio_id, Position.quantity > 0)
            .all()
        )

        for pos in positions:
            if pos.current_price is None or pos.avg_entry_price is None:
                continue

            pnl_pct = (pos.current_price / pos.avg_entry_price - 1) * 100

            # SL à -3% (config ninja)
            if pnl_pct <= -3.0:
                ops.append({
                    "ticker": pos.ticker,
                    "action": "SELL",
                    "reason": f"STOP-LOSS déclenché: {pnl_pct:.1f}% (seuil -3%)",
                    "confidence": 0.95,
                    "source": "opportunity_sl_tp",
                    "urgency": "high",
                    "pnl_pct": pnl_pct,
                })

            # TP à +6% (config ninja)
            elif pnl_pct >= 6.0:
                ops.append({
                    "ticker": pos.ticker,
                    "action": "SELL",
                    "reason": f"TAKE-PROFIT déclenché: {pnl_pct:.1f}% (seuil +6%)",
                    "confidence": 0.95,
                    "source": "opportunity_sl_tp",
                    "urgency": "high",
                    "pnl_pct": pnl_pct,
                })

        return ops

    def _find_price_anomalies(self, db: Session) -> list[dict[str, Any]]:
        """Détecte les gaps de prix et volume spikes."""
        ops = []
        since = datetime.utcnow() - timedelta(days=self.lookback_days)

        tickers = db.query(MarketData.ticker).distinct().all()
        tickers = [t[0] for t in tickers]

        for ticker in tickers:
            rows = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker, MarketData.timestamp >= since)
                .order_by(MarketData.timestamp.asc())
                .all()
            )
            if len(rows) < 2:
                continue

            prices = [r.price for r in rows]
            volumes = [r.volume for r in rows if r.volume]

            # Gap intraday > 3%
            if len(prices) >= 2:
                latest = prices[-1]
                prev = prices[-2]
                gap = (latest / prev - 1) * 100 if prev else 0

                if abs(gap) >= 3.0:
                    action = "BUY" if gap < 0 else "SELL"
                    ops.append({
                        "ticker": ticker,
                        "action": action,
                        "reason": f"Gap de {gap:+.1f}% en {self.lookback_days}j ({latest:.2f} vs {prev:.2f})",
                        "confidence": min(abs(gap) / 10, 0.8),
                        "source": "opportunity_gap",
                        "urgency": "medium",
                        "gap_pct": gap,
                    })

            # Volume spike > 3x moyenne
            if len(volumes) >= 20:
                avg_vol = np.mean(volumes[-20:-1])
                latest_vol = volumes[-1]
                if avg_vol > 0 and latest_vol / avg_vol >= 3.0:
                    ops.append({
                        "ticker": ticker,
                        "action": "BUY",  # volume spike = intérêt
                        "reason": f"Volume spike: {latest_vol/avg_vol:.1f}x la moyenne",
                        "confidence": 0.6,
                        "source": "opportunity_volume",
                        "urgency": "medium",
                        "volume_ratio": latest_vol / avg_vol,
                    })

        return ops

    def _find_competitor_opportunities(self, db: Session) -> list[dict[str, Any]]:
        """Trouve des opportunités basées sur les relations concurrentielles."""
        ops = []
        since = datetime.utcnow() - timedelta(days=3)

        for ticker, competitors in COMPETITOR_PAIRS.items():
            # Vérifier si le ticker a baissé récemment
            rows = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker, MarketData.timestamp >= since)
                .order_by(MarketData.timestamp.asc())
                .all()
            )
            if len(rows) < 2:
                continue

            old_price = rows[0].price
            new_price = rows[-1].price
            change = (new_price / old_price - 1) * 100 if old_price else 0

            if change <= -3.0:  # Le concurrent a baissé
                for comp in competitors:
                    # Vérifier si le compétiteur existe en DB
                    comp_rows = (
                        db.query(MarketData)
                        .filter(MarketData.ticker == comp, MarketData.timestamp >= since)
                        .order_by(MarketData.timestamp.desc())
                        .limit(1)
                        .all()
                    )
                    if comp_rows:
                        ops.append({
                            "ticker": comp,
                            "action": "BUY",
                            "reason": f"{ticker} a baissé {change:.1f}% → opportunité sur concurrent {comp}",
                            "confidence": 0.7,
                            "source": "opportunity_competitor",
                            "urgency": "medium",
                            "trigger_ticker": ticker,
                            "trigger_change_pct": change,
                        })

        return ops

    def _technical_screener(self, db: Session) -> list[dict[str, Any]]:
        """Screener technique simple (RSI extrême, momentum)."""
        ops = []
        since = datetime.utcnow() - timedelta(days=30)

        tickers = db.query(MarketData.ticker).distinct().all()
        tickers = [t[0] for t in tickers]

        for ticker in tickers:
            rows = (
                db.query(MarketData)
                .filter(MarketData.ticker == ticker, MarketData.timestamp >= since)
                .order_by(MarketData.timestamp.asc())
                .all()
            )
            if len(rows) < 14:
                continue

            closes = pd.Series([r.price for r in rows])
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            latest_rsi = rsi.iloc[-1]

            if pd.isna(latest_rsi):
                continue

            if latest_rsi <= 20:
                ops.append({
                    "ticker": ticker,
                    "action": "BUY",
                    "reason": f"RSI survente: {latest_rsi:.1f} (seuil 20)",
                    "confidence": 0.75,
                    "source": "opportunity_rsi",
                    "urgency": "medium",
                    "rsi": latest_rsi,
                })
            elif latest_rsi >= 80:
                ops.append({
                    "ticker": ticker,
                    "action": "SELL",
                    "reason": f"RSI surachat: {latest_rsi:.1f} (seuil 80)",
                    "confidence": 0.75,
                    "source": "opportunity_rsi",
                    "urgency": "medium",
                    "rsi": latest_rsi,
                })

        return ops

    def get_defense_opportunities(self, db: Session) -> list[dict[str, Any]]:
        """Retourne les tickers défense en cas d'événement géopolitique."""
        available = []
        for t in DEFENSE_TICKERS:
            exists = db.query(MarketData).filter(MarketData.ticker == t).first()
            if exists:
                available.append(t)

        return [
            {
                "ticker": t,
                "action": "BUY",
                "reason": "Événement géopolitique — secteur défense",
                "confidence": 0.6,
                "source": "opportunity_macro",
                "urgency": "high",
            }
            for t in available
        ]

    def get_energy_opportunities(self, db: Session) -> list[dict[str, Any]]:
        """Retourne les tickers énergie en cas de tension au Moyen-Orient."""
        available = []
        for t in ENERGY_TICKERS:
            exists = db.query(MarketData).filter(MarketData.ticker == t).first()
            if exists:
                available.append(t)

        return [
            {
                "ticker": t,
                "action": "BUY",
                "reason": "Tension énergétique — secteur pétrole/gaz",
                "confidence": 0.6,
                "source": "opportunity_macro",
                "urgency": "high",
            }
            for t in available
        ]
