"""Tests for ensemble CLI subcommand."""

from rotary_target.cli import main


def test_ensemble_dry_run(capsys):
    exit_code = main(["ensemble", "--predictions", "a.json", "b.json", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "a.json" in out
    assert "b.json" in out
    assert "dry-run" in out


def test_ensemble_missing_file():
    exit_code = main(["ensemble", "--predictions", "nonexistent_file.json"])
    assert exit_code == 2
