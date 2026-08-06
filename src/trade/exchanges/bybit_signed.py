"""Bybit v5 request signer.

Signature formula (from Bybit v5 REST docs):

    signature = HMAC_SHA256(secret, timestamp + api_key + recv_window + payload)

where `payload` is:
- for POST: the exact JSON body bytes sent in the request
- for GET:  the URL-encoded query string with keys in insertion order

The signer takes `payload` as a raw string so the caller controls
encoding — this eliminates the classic bug where the signed JSON does not
match the JSON actually put on the wire.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from collections.abc import Callable

_DEFAULT_RECV_WINDOW_MS = 5000


class BybitSigner:
    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        recv_window_ms: int = _DEFAULT_RECV_WINDOW_MS,
        now_ms: Callable[[], int] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be non-empty")
        if not api_secret:
            raise ValueError("api_secret must be non-empty")
        if recv_window_ms < 1:
            raise ValueError("recv_window_ms must be positive")
        self._api_key = api_key
        self._secret = api_secret.encode()
        self._recv_window_ms = recv_window_ms
        self._now_ms = now_ms or (lambda: int(time.time() * 1000))

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def recv_window_ms(self) -> int:
        return self._recv_window_ms

    def sign(self, *, timestamp_ms: str, payload: str) -> str:
        message = f"{timestamp_ms}{self._api_key}{self._recv_window_ms}{payload}"
        return hmac.new(self._secret, message.encode(), hashlib.sha256).hexdigest()

    def headers_for(self, payload: str) -> dict[str, str]:
        """Build the full Bybit-required header set for a signed request."""
        timestamp_ms = str(self._now_ms())
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp_ms,
            "X-BAPI-RECV-WINDOW": str(self._recv_window_ms),
            "X-BAPI-SIGN": self.sign(timestamp_ms=timestamp_ms, payload=payload),
            "Content-Type": "application/json",
        }
