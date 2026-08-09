"""Paper CLI: arming guard + journal status/verify commands."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from trade.cli.app import app
from trade.cli.paper import _resolve_execution
from trade.paper.journal import PaperJournal


def test_resolve_execution_defaults_disabled() -> None:
    assert _resolve_execution(False, None) is False
    assert _resolve_execution(False, "ARM PAPER EXECUTION") is False


def test_resolve_execution_requires_exact_confirm() -> None:
    with pytest.raises(typer.BadParameter):
        _resolve_execution(True, None)
    with pytest.raises(typer.BadParameter):
        _resolve_execution(True, "arm paper execution")
    assert _resolve_execution(True, "ARM PAPER EXECUTION") is True


def test_cli_verify_ok(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "j")
    journal.record("engine_start", {"mode": "test"})
    journal.record("decision", {"symbol": "BTCUSDT", "action": "HOLD"})

    result = CliRunner().invoke(app, ["paper", "verify", "--journal-dir", str(tmp_path / "j")])
    assert result.exit_code == 0, result.output
    assert "audit chain: OK" in result.output


def test_cli_status_reports_events(tmp_path: Path) -> None:
    journal = PaperJournal(tmp_path / "j")
    journal.record("fill", {"symbol": "BTCUSDT", "equity": 10123.45})

    result = CliRunner().invoke(app, ["paper", "status", "--journal-dir", str(tmp_path / "j")])
    assert result.exit_code == 0, result.output
    assert "10123.45" in result.output
    assert "audit chain: OK" in result.output


def test_cli_run_refuses_to_arm_without_confirm(tmp_path: Path) -> None:
    # No network / no manifest needed: the arming guard fires before anything.
    result = CliRunner().invoke(
        app,
        ["paper", "run", "--arm-execution", "--journal-dir", str(tmp_path / "j")],
    )
    assert result.exit_code != 0
    assert "Refusing to arm" in result.output
