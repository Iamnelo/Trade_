"""Tests for the live position reconciler."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from trade.execution.types import VenuePosition
from trade.runtime.reconciler import LivePositionReconciler, divergent


class _FakeVenue:
    def __init__(self, positions: Sequence[VenuePosition]) -> None:
        self._positions = list(positions)

    async def submit_order(self, order):
        raise NotImplementedError

    async def cancel_order(self, *, symbol, client_order_id):
        raise NotImplementedError

    async def cancel_all(self, *, symbol=None):
        raise NotImplementedError

    async def get_positions(self, *, symbol=None):
        if symbol is not None:
            return [p for p in self._positions if p.symbol == symbol]
        return list(self._positions)


def _venue_pos(symbol: str, qty: float) -> VenuePosition:
    return VenuePosition(symbol=symbol, quantity=qty, entry_price=100.0, unrealized_pnl=0.0)


async def test_reports_no_divergence_when_positions_match() -> None:
    venue = _FakeVenue([_venue_pos("BTCUSDT", 0.5)])
    reconciler = LivePositionReconciler(venue=venue, local_positions_fn=lambda: {"BTCUSDT": 0.5})
    results = await reconciler.reconcile_once()
    assert len(results) == 1
    assert not results[0].diverges
    assert divergent(results) == []


async def test_reports_divergence_when_venue_and_local_disagree() -> None:
    venue = _FakeVenue([_venue_pos("BTCUSDT", 0.5)])
    reconciler = LivePositionReconciler(venue=venue, local_positions_fn=lambda: {"BTCUSDT": 0.3})
    results = await reconciler.reconcile_once()
    assert len(results) == 1
    r = results[0]
    assert r.diverges
    assert r.delta == pytest.approx(0.2)


async def test_symbol_present_only_on_venue_shows_up_as_divergence() -> None:
    venue = _FakeVenue([_venue_pos("BTCUSDT", 0.5)])
    reconciler = LivePositionReconciler(venue=venue, local_positions_fn=lambda: {})
    results = await reconciler.reconcile_once()
    assert results[0].local_qty == 0.0
    assert results[0].venue_qty == 0.5
    assert results[0].diverges


async def test_symbol_present_only_locally_shows_up_as_divergence() -> None:
    venue = _FakeVenue([])
    reconciler = LivePositionReconciler(venue=venue, local_positions_fn=lambda: {"ETHUSDT": 1.0})
    results = await reconciler.reconcile_once()
    assert results[0].symbol == "ETHUSDT"
    assert results[0].venue_qty == 0.0
    assert results[0].diverges


async def test_multi_symbol_output_sorted() -> None:
    venue = _FakeVenue([_venue_pos("BTCUSDT", 0.5), _venue_pos("ETHUSDT", 1.0)])
    reconciler = LivePositionReconciler(
        venue=venue,
        local_positions_fn=lambda: {"BTCUSDT": 0.5, "ETHUSDT": 1.0, "SOLUSDT": 2.0},
    )
    results = await reconciler.reconcile_once()
    assert [r.symbol for r in results] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_tolerance_absorbs_tiny_deltas() -> None:
    venue = _FakeVenue([_venue_pos("BTCUSDT", 0.5000001)])
    reconciler = LivePositionReconciler(
        venue=venue,
        local_positions_fn=lambda: {"BTCUSDT": 0.5},
        tolerance_qty=1e-4,
    )
    results = await reconciler.reconcile_once()
    assert not results[0].diverges


def test_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError):
        LivePositionReconciler(
            venue=_FakeVenue([]),
            local_positions_fn=lambda: {},
            tolerance_qty=-1.0,
        )
