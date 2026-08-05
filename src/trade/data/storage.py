"""Bytes-in, bytes-out object storage.

`Store` is a minimal protocol implemented by both `LocalStore` (writes into a
directory) and `S3Store` (writes into any S3-compatible bucket, MinIO in local
dev). Higher-level modules — parquet writer, manifest — depend only on the
protocol so tests can inject an in-memory fake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import boto3
from botocore.exceptions import ClientError


class Store(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...
    def list_keys(self, prefix: str = "") -> list[str]: ...


class InMemoryStore:
    """Testing-only store backed by a dict."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._data[key] = data

    def get(self, key: str) -> bytes:
        return self._data[key]

    def exists(self, key: str) -> bool:
        return key in self._data

    def list_keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self._data if k.startswith(prefix))


class LocalStore:
    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._root / key

    def put(self, key: str, data: bytes) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace via a tmp file so a half-written parquet is never observed.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list_keys(self, prefix: str = "") -> list[str]:
        if prefix:
            base = self._path(prefix)
            if base.is_file():
                return [prefix]
            if not base.exists():
                # The prefix might be a partial directory name; walk from root and filter.
                return sorted(
                    self._rel(p)
                    for p in self._root.rglob("*")
                    if p.is_file()
                    and not p.name.endswith(".tmp")
                    and self._rel(p).startswith(prefix)
                )
        else:
            base = self._root
        return sorted(
            self._rel(p) for p in base.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        )

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self._root)).replace("\\", "/")


class S3Store:
    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
        client: Any = None,
    ) -> None:
        self._bucket = bucket
        self._client = client or boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body: bytes = response["Body"].read()
        return body

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def list_keys(self, prefix: str = "") -> list[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(str(obj["Key"]))
        return sorted(keys)
