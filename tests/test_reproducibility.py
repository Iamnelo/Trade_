"""Tests for the reproducibility-hash helpers and git/lockfile utilities."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from trade.reproducibility.git import current_git_sha, lockfile_sha
from trade.reproducibility.hash import compute_reproducibility_hash


def test_reproducibility_hash_is_deterministic() -> None:
    kwargs: dict[str, object] = {
        "dataset_manifest_ids": ["ds_a", "ds_b"],
        "feature_manifest_ids": ["fs_a"],
        "model_config": {"lr": 0.01, "n_estimators": 100},
        "code_git_sha": "deadbeef",
        "python_lockfile_sha": "cafef00d",
    }
    h1 = compute_reproducibility_hash(**kwargs)  # type: ignore[arg-type]
    h2 = compute_reproducibility_hash(**kwargs)  # type: ignore[arg-type]
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_reproducibility_hash_input_order_independent() -> None:
    h_ab = compute_reproducibility_hash(
        dataset_manifest_ids=["a", "b"],
        feature_manifest_ids=["x", "y"],
        model_config={"k1": 1, "k2": 2},
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    h_ba = compute_reproducibility_hash(
        dataset_manifest_ids=["b", "a"],
        feature_manifest_ids=["y", "x"],
        model_config={"k2": 2, "k1": 1},
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
    )
    assert h_ab == h_ba


def test_reproducibility_hash_changes_when_any_input_changes() -> None:
    base: dict[str, object] = {
        "dataset_manifest_ids": ["a"],
        "feature_manifest_ids": ["x"],
        "model_config": {"lr": 0.01},
        "code_git_sha": "deadbeef",
        "python_lockfile_sha": "cafef00d",
    }
    h_base = compute_reproducibility_hash(**base)  # type: ignore[arg-type]

    variants: list[dict[str, object]] = [
        {**base, "dataset_manifest_ids": ["a", "b"]},
        {**base, "feature_manifest_ids": ["z"]},
        {**base, "model_config": {"lr": 0.02}},
        {**base, "code_git_sha": "c0ffee"},
        {**base, "python_lockfile_sha": "beefbeef"},
    ]
    for v in variants:
        assert compute_reproducibility_hash(**v) != h_base  # type: ignore[arg-type]


def test_reproducibility_hash_rejects_empty_code_sha() -> None:
    with pytest.raises(ValueError, match="code_git_sha"):
        compute_reproducibility_hash(
            dataset_manifest_ids=[],
            feature_manifest_ids=[],
            model_config={},
            code_git_sha="",
            python_lockfile_sha="cafef00d",
        )


def test_lockfile_sha_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "lock"
    p.write_bytes(b"lockfile-bytes")
    assert lockfile_sha(p) == hashlib.sha256(b"lockfile-bytes").hexdigest()


def test_lockfile_sha_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        lockfile_sha(tmp_path / "nope")


def test_current_git_sha_in_this_repo() -> None:
    # This project is a git repo; expect a real 40-char sha.
    sha = current_git_sha()
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_current_git_sha_outside_repo(tmp_path: Path) -> None:
    # A fresh directory is not a repo — expect a RuntimeError.
    with pytest.raises(RuntimeError):
        current_git_sha(cwd=tmp_path)
    _ = subprocess  # keep import used
