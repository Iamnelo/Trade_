"""Tests for the Store implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from trade.data.storage import InMemoryStore, LocalStore, S3Store


def test_in_memory_roundtrip() -> None:
    s = InMemoryStore()
    s.put("a/b.parquet", b"hello")
    assert s.exists("a/b.parquet")
    assert s.get("a/b.parquet") == b"hello"
    assert s.list_keys("a/") == ["a/b.parquet"]
    assert not s.exists("missing")


def test_local_store_roundtrip(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    s.put("raw/kline/bybit/linear/BTCUSDT/60/2024/01.parquet", b"payload")
    assert s.exists("raw/kline/bybit/linear/BTCUSDT/60/2024/01.parquet")
    assert s.get("raw/kline/bybit/linear/BTCUSDT/60/2024/01.parquet") == b"payload"
    keys = s.list_keys("raw/")
    assert keys == ["raw/kline/bybit/linear/BTCUSDT/60/2024/01.parquet"]


def test_local_store_atomic_replace(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    key = "manifests/x.json"
    s.put(key, b"v1")
    s.put(key, b"v2")
    assert s.get(key) == b"v2"
    # No .tmp files remain visible in listings.
    all_keys = s.list_keys("")
    assert not any(k.endswith(".tmp") for k in all_keys)


def test_local_store_lists_by_partial_prefix(tmp_path: Path) -> None:
    s = LocalStore(tmp_path)
    s.put("raw/bybit_kline/foo.parquet", b"x")
    s.put("raw/binance_kline/bar.parquet", b"y")
    s.put("manifests/thing.json", b"z")
    got = s.list_keys("raw/")
    assert sorted(got) == [
        "raw/binance_kline/bar.parquet",
        "raw/bybit_kline/foo.parquet",
    ]


def _make_client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "x"}}, operation_name="HeadObject"
    )


def test_s3_store_delegates_to_client() -> None:
    fake_client = MagicMock()
    fake_client.get_object.return_value = {"Body": MagicMock(read=lambda: b"payload")}
    s = S3Store(bucket="trade", client=fake_client)

    s.put("k", b"payload")
    fake_client.put_object.assert_called_once_with(Bucket="trade", Key="k", Body=b"payload")
    assert s.get("k") == b"payload"


def test_s3_store_exists_true_and_false() -> None:
    fake_client = MagicMock()
    s = S3Store(bucket="trade", client=fake_client)

    fake_client.head_object.return_value = {}
    assert s.exists("present") is True

    fake_client.head_object.side_effect = _make_client_error("404")
    assert s.exists("absent") is False


def test_s3_store_exists_propagates_unexpected_errors() -> None:
    fake_client = MagicMock()
    fake_client.head_object.side_effect = _make_client_error("500")
    s = S3Store(bucket="trade", client=fake_client)
    try:
        s.exists("x")
    except ClientError:
        pass
    else:
        raise AssertionError("Unexpected S3 error must propagate")


def test_s3_store_list_paginates() -> None:
    fake_paginator: Any = MagicMock()
    fake_paginator.paginate.return_value = iter(
        [
            {"Contents": [{"Key": "a"}, {"Key": "b"}]},
            {"Contents": [{"Key": "c"}]},
        ]
    )
    fake_client = MagicMock()
    fake_client.get_paginator.return_value = fake_paginator
    s = S3Store(bucket="trade", client=fake_client)
    assert s.list_keys("prefix/") == ["a", "b", "c"]
