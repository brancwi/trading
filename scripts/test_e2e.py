#!/usr/bin/env python3
"""Test end-to-end — valide MCP, capital movements, stratégies cash_available.

Usage:
    uv run python scripts/test_e2e.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text
from trading.core.database import db_session, engine
from trading.core.models import Portfolio, Signal, CapitalMovement, Position, Trade
from trading.strategies.simulation import SimulationStrategy
from trading.mcp.server import (
    list_portfolios,
    get_capital_movements,
    reserve_capital,
    release_capital,
)


def test_db_connection() -> None:
    """Vérifie la connexion PostgreSQL."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();")).scalar()
        print(f"✅ PostgreSQL connecté: {result.split()[0]} {result.split()[1]}")


def test_portfolios() -> None:
    """Vérifie que les 3 portefeuilles existent."""
    with db_session() as db:
        ports = db.query(Portfolio).all()
        ids = {p.id for p in ports}
        expected = {"simulation", "rotation", "ninja"}
        assert expected.issubset(ids), f"Portefeuilles manquants: {expected - ids}"
        print(f"✅ Portefeuilles OK: {ids}")


def test_capital_movements() -> None:
    """Test reserve / release via MCP tools."""
    # Reset
    with db_session() as db:
        port = db.query(Portfolio).filter_by(id="simulation").first()
        port.reserved_cash = 0.0
        port.cash_current = port.cash_initial
        db.query(CapitalMovement).filter_by(portfolio_id="simulation").delete()
        db.commit()

    # Reserve
    result = reserve_capital("simulation", 500.0, "Test E2E reserve")
    assert result["status"] == "reserved"
    assert result["reserved_cash"] == 500.0
    assert result["cash_available"] == 2500.0
    print(f"✅ Reserve OK: reserved=500, available=2500")

    # Verify DB
    with db_session() as db:
        port = db.query(Portfolio).filter_by(id="simulation").first()
        assert port.reserved_cash == 500.0
        assert port.cash_available == 2500.0

    # Release partial
    result = release_capital("simulation", 200.0)
    assert result["status"] == "released"
    assert result["reserved_cash"] == 300.0
    assert result["cash_available"] == 2700.0
    print(f"✅ Release OK: reserved=300, available=2700")

    # History
    movements = get_capital_movements("simulation", limit=10)
    assert len(movements) == 2
    types = {m["movement_type"] for m in movements}
    assert types == {"reserve", "release"}
    print(f"✅ Capital movements history OK: {len(movements)} entries")


def test_strategy_respects_reserved_cash() -> None:
    """Vérifie que SimulationStrategy respecte cash_available."""
    with db_session() as db:
        # Setup: simulation has 3000 cash, 300 reserved → 2700 available
        port = db.query(Portfolio).filter_by(id="simulation").first()
        port.cash_current = 3000.0
        port.reserved_cash = 300.0
        db.commit()

        # Clean slate
        db.query(Trade).filter_by(portfolio_id="simulation").delete()
        db.query(Position).filter_by(portfolio_id="simulation").delete()
        db.query(Signal).filter(Signal.consumed == 1).delete()
        db.commit()

        # Create a BUY signal for AAPL at $100
        signal = Signal(
            ticker="AAPL",
            action="BUY",
            sentiment=0.8,
            strength=0.9,
            confidence=0.85,
            price_at_signal=100.0,
        )
        db.add(signal)
        db.commit()

        # Run strategy with price $100
        strat = SimulationStrategy("simulation")
        trades = strat.run(db, {"AAPL": 100.0})

        # Should have bought: max_trade=500, cash_available=2700, cash_min=100
        # trade_amount = min(500, 2700 - 100) = 500
        # qty = 500 / 100 = 5
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
        trade = trades[0]
        assert trade.ticker == "AAPL"
        assert trade.action == "BUY"
        assert trade.quantity == 5.0  # 500 / 100
        print(f"✅ Strategy trade OK: {trade.quantity} AAPL @ {trade.price}")

        # Verify cash was deducted from cash_current (not cash_available)
        port = db.query(Portfolio).filter_by(id="simulation").first()
        expected_cash = 3000.0 - 500.0 - 1.0  # trade amount + fee
        assert abs(port.cash_current - expected_cash) < 0.01, (
            f"Cash mismatch: {port.cash_current} != {expected_cash}"
        )
        # reserved_cash should be unchanged
        assert port.reserved_cash == 300.0
        print(f"✅ Cash deducted correctly: cash_current={port.cash_current}, reserved={port.reserved_cash}")

        # Now reserve almost all remaining cash and try again
        remaining = port.cash_current - port.reserved_cash
        port.reserved_cash = port.cash_current - 50  # available = 50
        db.commit()

        # Create another signal
        signal2 = Signal(
            ticker="TSLA",
            action="BUY",
            sentiment=0.8,
            strength=0.9,
            confidence=0.85,
            price_at_signal=200.0,
        )
        db.add(signal2)
        db.commit()

        trades2 = strat.run(db, {"TSLA": 200.0})
        # cash_available = 50, cash_min = 100 → no trade
        assert len(trades2) == 0, "Should not trade when cash_available < cash_min"
        print(f"✅ Strategy correctly skips trade when cash_available insufficient")


def cleanup() -> None:
    """Remet la DB dans un état propre."""
    with db_session() as db:
        port = db.query(Portfolio).filter_by(id="simulation").first()
        port.reserved_cash = 0.0
        port.cash_current = port.cash_initial
        db.query(CapitalMovement).filter_by(portfolio_id="simulation").delete()
        db.query(Trade).filter_by(portfolio_id="simulation").delete()
        db.query(Position).filter_by(portfolio_id="simulation").delete()
        db.query(Signal).filter(Signal.consumed == 1).delete()
        db.commit()
        print("✅ Cleanup OK")


def main() -> int:
    print("=" * 60)
    print("Trading Engine V4.2 — Test End-to-End")
    print("=" * 60)

    try:
        test_db_connection()
        test_portfolios()
        test_capital_movements()
        test_strategy_respects_reserved_cash()
        cleanup()
        print("=" * 60)
        print("🎉 Tous les tests E2E ont réussi !")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"❌ ÉCHEC: {e}")
        return 1
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
