from pathlib import Path

from rotary_target.data.dota import parse_label_line, validate_dota_root
from rotary_target.data.split_config import build_mmrotate_split_configs


def test_parse_dota_label_line_accepts_raw_hyphenated_class():
    obj = parse_label_line(
        "105 120 170 121 168 180 103 179 small-vehicle 0",
        line_number=1,
    )
    assert obj is not None
    assert obj.category == "small-vehicle"
    assert obj.difficult == 0


def test_validate_dota_root_checks_expected_layout(tmp_path: Path):
    root = tmp_path / "DOTA-v1.0"
    for relative in [
        "train/images",
        "train/labelTxt",
        "val/images",
        "val/labelTxt",
        "test/images",
    ]:
        (root / relative).mkdir(parents=True)
    (root / "train" / "labelTxt" / "P0001.txt").write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.5\n"
        "105 120 170 121 168 180 103 179 small-vehicle 0\n",
        encoding="utf-8",
    )
    (root / "val" / "labelTxt" / "P0002.txt").write_text(
        "10 20 30 20 30 40 10 40 plane 1\n",
        encoding="utf-8",
    )

    report = validate_dota_root(root)

    assert report.ok
    assert report.checked_label_files == 2
    assert report.objects == 2


def test_build_mmrotate_split_configs_uses_base_json_shape(tmp_path: Path):
    configs = build_mmrotate_split_configs(tmp_path / "raw", tmp_path / "processed", tile_size=512, gap=128)
    assert sorted(configs) == ["test", "trainval"]
    assert configs["trainval"]["sizes"] == [512]
    assert configs["trainval"]["gaps"] == [128]
    assert configs["trainval"]["ann_dirs"] is not None
    assert configs["test"]["ann_dirs"] is None
