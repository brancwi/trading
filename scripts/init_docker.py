#!/usr/bin/env python3
"""Initialise la DB et les données pour Docker Compose."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trading.core.database import init_db, db_session
from trading.core.models import Portfolio


def main():
    print("[Docker Init] Création des tables...")
    init_db()

    with db_session() as db:
        defaults = [
            Portfolio(
                id="simulation",
                name="Simulation Day Trading",
                strategy_type="simulation",
                base_currency="USD",
                cash_initial=3000,
                cash_current=3000,
                max_trade_amount=500,
                fee_per_order=1.0,
                status="active",
                config_json='{"sentiment_threshold": 0.5, "cash_min": 100}',
            ),
            Portfolio(
                id="rotation",
                name="Rotation Sectorielle",
                strategy_type="rotation",
                base_currency="USD",
                cash_initial=3000,
                cash_current=3000,
                max_trade_amount=600,
                fee_per_order=1.0,
                status="active",
                config_json='{"stop_loss_pct": -12, "take_profit_pct": 20, "take_profit_sell_pct": 50}',
            ),
            Portfolio(
                id="ninja",
                name="Ninja Opportuniste",
                strategy_type="ninja",
                base_currency="EUR",
                cash_initial=500,
                cash_current=500,
                max_trade_amount=150,
                fee_per_order=1.0,
                status="active",
                config_json='{"cash_min": 50, "min_sectors": 3}',
            ),
        ]
        for port in defaults:
            exists = db.query(Portfolio).filter(Portfolio.id == port.id).first()
            if not exists:
                db.add(port)
        db.commit()

    print("[Docker Init] Base initialisée avec succès.")


if __name__ == "__main__":
    main()
