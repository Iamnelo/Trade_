"""Bybit v5 signed ExecutionVenue implementation.

Wraps the private trading endpoints (`/v5/order/create`,
`/v5/order/cancel`, `/v5/order/cancel-all`, `/v5/position/list`) behind
the `ExecutionVenue` Protocol. Network + 5xx transients are retried with
exponential backoff via tenacity; business errors (non-zero retCode) are
raised as `ExchangeError` immediately and NOT retried.

Idempotency: every submitted order carries `orderLinkId = order.client_order_id`.
Bybit rejects duplicate `orderLinkId` within a 24h window, so a retry of a
timed-out submit is safe — either the original succeeded (Bybit rejects the
retry, and the OMS then queries), or it truly failed (Bybit accepts the
retry). The OMS side of that recovery lives in Phase 4b's reconciler.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlencode

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trade.exchanges.bybit_signed import BybitSigner
from trade.execution.types import (
    ExchangeError,
    TransientExchangeError,
    VenueOrderAck,
    VenuePosition,
)
from trade.mre.types import Order, Side
from trade.utils.clock import utcnow

_ORDER_CREATE = "/v5/order/create"
_ORDER_CANCEL = "/v5/order/cancel"
_ORDER_CANCEL_ALL = "/v5/order/cancel-all"
_POSITION_LIST = "/v5/position/list"

_RETRYABLE_HTTP = (httpx.NetworkError, httpx.TimeoutException, TransientExchangeError)


class BybitExecutionVenue:
    def __init__(
        self,
        *,
        base_url: str,
        signer: BybitSigner,
        category: str = "linear",
        client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._signer = signer
        self._category = category
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # ExecutionVenue methods
    # ------------------------------------------------------------------

    async def submit_order(self, order: Order) -> VenueOrderAck:
        side = "Buy" if order.side is Side.BUY else "Sell"
        body = {
            "category": self._category,
            "symbol": order.symbol,
            "side": side,
            "orderType": "Market",
            "qty": _fmt_qty(order.quantity),
            "orderLinkId": order.client_order_id,
        }
        result = await self._signed_post(_ORDER_CREATE, body)
        return VenueOrderAck(
            client_order_id=str(result.get("orderLinkId", order.client_order_id)),
            venue_order_id=str(result["orderId"]),
            accepted_at=utcnow(),
        )

    async def cancel_order(self, *, symbol: str, client_order_id: str) -> None:
        body = {
            "category": self._category,
            "symbol": symbol,
            "orderLinkId": client_order_id,
        }
        await self._signed_post(_ORDER_CANCEL, body)

    async def cancel_all(self, *, symbol: str | None = None) -> int:
        body: dict[str, Any] = {"category": self._category}
        if symbol is not None:
            body["symbol"] = symbol
        result = await self._signed_post(_ORDER_CANCEL_ALL, body)
        entries = result.get("list") or []
        return len(entries) if isinstance(entries, list) else 0

    async def get_positions(self, *, symbol: str | None = None) -> Sequence[VenuePosition]:
        query: dict[str, str] = {"category": self._category}
        if symbol is not None:
            query["symbol"] = symbol
        result = await self._signed_get(_POSITION_LIST, query)
        rows: list[dict[str, str]] = result.get("list") or []
        out: list[VenuePosition] = []
        for row in rows:
            size = float(row.get("size", "0"))
            if size == 0.0:
                continue
            side = row.get("side", "None")
            signed_qty = size if side == "Buy" else -size
            out.append(
                VenuePosition(
                    symbol=str(row["symbol"]),
                    quantity=signed_qty,
                    entry_price=float(row.get("avgPrice", "0") or 0),
                    unrealized_pnl=float(row.get("unrealisedPnl", "0") or 0),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Signed HTTP transport
    # ------------------------------------------------------------------

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE_HTTP),
    )
    async def _signed_post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        # Serialise once, sign the exact bytes we send.
        body_json = json.dumps(body, separators=(",", ":"), sort_keys=True)
        headers = self._signer.headers_for(body_json)
        response = await self._client.post(
            f"{self._base_url}{path}", content=body_json, headers=headers
        )
        return _unwrap(response)

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        retry=retry_if_exception_type(_RETRYABLE_HTTP),
    )
    async def _signed_get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        # Bybit's GET signature is over the URL-encoded query string.
        query = urlencode(sorted(params.items()))
        headers = self._signer.headers_for(query)
        response = await self._client.get(f"{self._base_url}{path}?{query}", headers=headers)
        return _unwrap(response)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_qty(qty: float) -> str:
    """Bybit expects qty as a decimal string. Avoid scientific notation."""
    return f"{qty:.10f}".rstrip("0").rstrip(".") or "0"


def _unwrap(response: httpx.Response) -> dict[str, Any]:
    # 5xx errors are transient; retry them. 4xx and other statuses are hard.
    if 500 <= response.status_code < 600:
        raise TransientExchangeError(
            code=response.status_code, message=f"http {response.status_code}"
        )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ExchangeError(code="?", message=f"unexpected payload: {payload!r}")

    ret_code = payload.get("retCode")
    if ret_code == 0:
        result = payload.get("result", {})
        return result if isinstance(result, dict) else {}
    ret_msg = str(payload.get("retMsg", ""))
    raise ExchangeError(code=ret_code if ret_code is not None else "?", message=ret_msg)
