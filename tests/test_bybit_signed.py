"""Tests for the Bybit v5 signer."""

from __future__ import annotations

import hashlib
import hmac

import pytest

from trade.exchanges.bybit_signed import BybitSigner


def test_signature_matches_manual_hmac_computation() -> None:
    signer = BybitSigner(
        api_key="my-api-key",
        api_secret="my-secret-key",
        recv_window_ms=5000,
        now_ms=lambda: 1_700_000_000_000,
    )
    payload = '{"category":"linear","symbol":"BTCUSDT"}'
    got = signer.sign(timestamp_ms="1700000000000", payload=payload)
    expected = hmac.new(
        b"my-secret-key",
        b"1700000000000my-api-key5000" + payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    assert got == expected


def test_headers_for_bundles_all_required_fields() -> None:
    signer = BybitSigner(api_key="ak", api_secret="sk", recv_window_ms=5000, now_ms=lambda: 12345)
    headers = signer.headers_for("payload-x")
    assert headers["X-BAPI-API-KEY"] == "ak"
    assert headers["X-BAPI-TIMESTAMP"] == "12345"
    assert headers["X-BAPI-RECV-WINDOW"] == "5000"
    assert headers["Content-Type"] == "application/json"
    assert len(headers["X-BAPI-SIGN"]) == 64  # sha256 hex


def test_headers_pull_fresh_timestamp_each_call() -> None:
    counter = {"v": 100}

    def now() -> int:
        counter["v"] += 1
        return counter["v"]

    signer = BybitSigner(api_key="ak", api_secret="sk", now_ms=now)
    h1 = signer.headers_for("p")
    h2 = signer.headers_for("p")
    assert h1["X-BAPI-TIMESTAMP"] != h2["X-BAPI-TIMESTAMP"]
    # Different timestamps => different signatures on the same payload.
    assert h1["X-BAPI-SIGN"] != h2["X-BAPI-SIGN"]


def test_rejects_empty_credentials() -> None:
    with pytest.raises(ValueError, match="api_key"):
        BybitSigner(api_key="", api_secret="s")
    with pytest.raises(ValueError, match="api_secret"):
        BybitSigner(api_key="k", api_secret="")


def test_rejects_bad_recv_window() -> None:
    with pytest.raises(ValueError, match="recv_window"):
        BybitSigner(api_key="k", api_secret="s", recv_window_ms=0)
