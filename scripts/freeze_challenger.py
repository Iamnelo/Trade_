"""Freeze one preselected challenger without touching the Daily V1 manifest."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from freeze_winners import freeze_one

from trade.reproducibility.git import current_git_sha, lockfile_sha

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lockfile", type=Path, default=REPO_ROOT / "uv.lock")
    parser.add_argument(
        "--code-git-sha", help="Published source commit SHA; defaults to local HEAD"
    )
    args = parser.parse_args()

    git_sha = args.code_git_sha or current_git_sha(cwd=REPO_ROOT)
    lock_sha = lockfile_sha(args.lockfile) if args.lockfile.exists() else "no-lockfile"
    spec_path = args.spec.resolve()
    out_root = args.out_root.resolve()
    manifest_path = args.manifest.resolve()

    out_root.mkdir(parents=True, exist_ok=True)
    entry = freeze_one(
        spec_path=spec_path,
        code_git_sha=git_sha,
        lockfile_sha_str=lock_sha,
        data_root=REPO_ROOT,
        out_root=out_root,
    )
    manifest = {
        "frozen_at": datetime.now(tz=UTC).isoformat(),
        "code_git_sha": git_sha,
        "lockfile_sha": lock_sha,
        "note": "Provisional shadow challenger; simulated paper execution only.",
        "winners": [entry],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
