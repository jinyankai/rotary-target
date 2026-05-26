from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment config cannot be loaded or validated."""


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load a small YAML experiment file.

    PyYAML is used when available. The fallback parser intentionally supports
    only the subset used by this repo so CI can run without optional packages.
    """

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        data = _parse_yaml_subset(text)
    else:
        loaded = yaml.safe_load(text)
        data = loaded if loaded is not None else {}

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level config must be a mapping: {config_path}")
    return data


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: Path
    split_root: Path | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "DatasetConfig":
        name = _require_str(data, "name")
        root = Path(_require_str(data, "root"))
        split_root = data.get("split_root")
        return cls(name=name, root=root, split_root=Path(split_root) if split_root else None)


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    experiment_name: str
    adapter: str
    algorithm: str
    dataset: DatasetConfig
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)
    backend: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        data = load_mapping(config_path)
        dataset = data.get("dataset")
        if not isinstance(dataset, dict):
            raise ConfigError("'dataset' must be a mapping")
        return cls(
            path=config_path,
            experiment_name=_require_str(data, "experiment_name"),
            adapter=_require_str(data, "adapter"),
            algorithm=_require_str(data, "algorithm"),
            dataset=DatasetConfig.from_mapping(dataset),
            training=_optional_mapping(data, "training"),
            evaluation=_optional_mapping(data, "evaluation"),
            report=_optional_mapping(data, "report"),
            backend=_optional_mapping(data, "backend"),
        )

    @property
    def work_dir(self) -> Path:
        raw = self.training.get("work_dir") or f"outputs/{self.adapter}/{self.experiment_name}"
        return Path(str(raw))


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing required string field: {key}")
    return value


def _optional_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{key}' must be a mapping when provided")
    return dict(value)


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line[: len(raw_line) - len(raw_line.lstrip(" \t"))]:
            raise ConfigError("Tabs are not supported in YAML indentation")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ConfigError(f"Unexpected trailing YAML content at line {index + 1}")
    if not isinstance(value, dict):
        raise ConfigError("Top-level YAML subset must be a mapping")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_dict(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    data: dict[str, Any] = {}
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {text}")
        if text.startswith("- "):
            break
        key, separator, raw_value = text.partition(":")
        if not separator or not key.strip():
            raise ConfigError(f"Expected 'key: value' mapping entry near: {text}")
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            data[key] = _parse_scalar(raw_value)
        elif index < len(lines) and lines[index][0] > current_indent:
            data[key], index = _parse_block(lines, index, lines[index][0])
        else:
            data[key] = {}
    return data, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    values: list[Any] = []
    while index < len(lines):
        current_indent, text = lines[index]
        if current_indent < indent:
            break
        if current_indent > indent:
            raise ConfigError(f"Unexpected indentation near: {text}")
        if not text.startswith("- "):
            break
        item = text[2:].strip()
        index += 1
        if item:
            values.append(_parse_scalar(item))
        elif index < len(lines) and lines[index][0] > current_indent:
            child, index = _parse_block(lines, index, lines[index][0])
            values.append(child)
        else:
            values.append(None)
    return values, index


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and value[0] == value[-1]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
