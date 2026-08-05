"""Git commit + lockfile digest helpers used by the reproducibility hash."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def current_git_sha(*, cwd: Path | None = None) -> str:
    """Return HEAD as a full sha. Raises if not inside a git repo."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git rev-parse HEAD failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    sha = result.stdout.strip()
    if not sha:
        raise RuntimeError("git rev-parse HEAD returned empty output")
    return sha


def lockfile_sha(lockfile_path: Path) -> str:
    """SHA-256 of the exact bytes of a python lockfile (uv.lock, requirements.txt, ...)."""
    if not lockfile_path.exists():
        raise FileNotFoundError(f"lockfile not found: {lockfile_path}")
    return hashlib.sha256(lockfile_path.read_bytes()).hexdigest()
