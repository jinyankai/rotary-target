from rotary_target.cli import main


def test_doctor_strict_allows_missing_optional_training_assets(capsys):
    exit_code = main(["doctor", "--strict"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "dota" in captured.out


def test_train_dry_run_prints_mmrotate_command(capsys):
    exit_code = main(
        [
            "train",
            "--experiment",
            "configs/experiments/roi_trans_r50_dota1.yaml",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "roi_trans_r50_fpn_1x_dota_le90.py" in captured.out
    assert "dry-run" in captured.out


def test_report_dry_run_handles_empty_outputs(capsys):
    exit_code = main(["report", "--dry-run"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No completed experiment metrics" in captured.out
