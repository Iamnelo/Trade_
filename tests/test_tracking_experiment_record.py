"""Test for the ExperimentRecord logging shape."""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import pytest

from trade.tracking.mlflow import log_experiment_record


@pytest.fixture(autouse=True)
def _allow_file_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    if "MLFLOW_TRACKING_URI" in os.environ:
        monkeypatch.delenv("MLFLOW_TRACKING_URI")


def test_log_experiment_record_writes_all_fields(tmp_path: Path) -> None:
    uri = f"file://{tmp_path.as_posix()}/mlruns"
    run_id = log_experiment_record(
        dataset_manifest_ids=["ds_a", "ds_b"],
        feature_manifest_ids=["fs_x"],
        model_config={"lr": 0.01, "depth": 6},
        code_git_sha="deadbeef",
        python_lockfile_sha="cafef00d",
        reproducibility_hash="a" * 64,
        tracking_uri=uri,
        experiment_name="phase25_test",
    )
    assert run_id

    mlflow.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(run_id)

    assert run.data.params["reproducibility_hash"] == "a" * 64
    assert run.data.params["dataset_manifest_ids"] == "ds_a,ds_b"
    assert run.data.params["feature_manifest_ids"] == "fs_x"
    assert run.data.params["code_git_sha"] == "deadbeef"
    assert run.data.params["cfg.lr"] == "0.01"
    assert run.data.params["cfg.depth"] == "6"
    assert run.data.tags["reproducibility_hash"] == "a" * 64
