"""Reproducibility hash — HARD REQUIREMENT for every model release.

`compute_reproducibility_hash` returns a stable SHA-256 over the exact
inputs that determine a model artifact:

- dataset manifest IDs (sorted)
- feature manifest IDs (sorted)
- model config (canonicalised)
- code git sha
- python lockfile sha

Two runs with the same reproducibility hash MUST produce byte-identical
model artifacts. A model may not be released if the hash cannot be
reproduced from committed artifacts. This turns "reproducible" from an
aspiration into a compile-time proof.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


def _canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _canonical_dict(m: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sort dict keys so equivalent configs hash to the same value."""
    out: dict[str, Any] = {}
    for k in sorted(m.keys()):
        v = m[k]
        if isinstance(v, Mapping):
            out[k] = _canonical_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [_canonical_dict(x) if isinstance(x, Mapping) else x for x in v]
        else:
            out[k] = v
    return out


def compute_reproducibility_hash(
    *,
    dataset_manifest_ids: Sequence[str],
    feature_manifest_ids: Sequence[str],
    model_config: Mapping[str, Any],
    code_git_sha: str,
    python_lockfile_sha: str,
) -> str:
    if not code_git_sha:
        raise ValueError("code_git_sha must be a non-empty commit hash")
    if not python_lockfile_sha:
        raise ValueError("python_lockfile_sha must be a non-empty digest")
    payload = {
        "dataset_manifest_ids": sorted(dataset_manifest_ids),
        "feature_manifest_ids": sorted(feature_manifest_ids),
        "model_config": _canonical_dict(model_config),
        "code_git_sha": code_git_sha,
        "python_lockfile_sha": python_lockfile_sha,
    }
    canonical = _canonical_json_dumps(payload)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
