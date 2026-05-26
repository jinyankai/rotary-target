from pathlib import Path

from rotary_target.config import ExperimentConfig, _parse_yaml_subset


ROOT = Path(__file__).resolve().parents[1]


def test_all_experiment_configs_parse():
    configs = sorted((ROOT / "configs" / "experiments").glob("*.yaml"))
    assert configs
    for path in configs:
        experiment = ExperimentConfig.from_file(path)
        assert experiment.adapter == "mmrotate"
        assert experiment.dataset.name == "DOTA-v1.0"
        assert experiment.work_dir.as_posix().startswith("outputs/mmrotate/")


def test_yaml_subset_parser_supports_nested_mappings():
    data = _parse_yaml_subset(
        """
experiment_name: smoke
adapter: mmrotate
algorithm: roi_trans
dataset:
  name: DOTA-v1.0
  root: data/raw/DOTA-v1.0
training:
  epochs: 12
  enabled: true
"""
    )
    assert data["dataset"]["root"] == "data/raw/DOTA-v1.0"
    assert data["training"]["epochs"] == 12
    assert data["training"]["enabled"] is True
