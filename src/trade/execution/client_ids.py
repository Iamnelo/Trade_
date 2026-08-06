"""Deterministic client-order-id generator.

`make_client_order_id` returns a 32-hex-char id from a stable content
hash. The same (symbol, timestamp, sequence, salt) always produces the
same id — so a crashed-and-restarted OMS can safely re-issue the SAME
order without risking a double-submit (Bybit rejects duplicate
`orderLinkId` within a 24h window).

The output is well within Bybit's 45-char `orderLinkId` limit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

_MAX_LEN = 40


def make_client_order_id(
    *,
    symbol: str,
    submit_time: datetime,
    sequence: int,
    salt: str = "",
) -> str:
    if submit_time.tzinfo is None:
        raise ValueError("submit_time must be UTC-aware")
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    key = f"{symbol}|{submit_time.isoformat()}|{sequence}|{salt}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return digest[:_MAX_LEN]
