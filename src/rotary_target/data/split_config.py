from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_mmrotate_split_configs(
    raw_root: str | Path,
    out_root: str | Path,
    *,
    tile_size: int = 1024,
    gap: int = 200,
    nproc: int = 10,
) -> dict[str, dict[str, Any]]:
    raw = Path(raw_root).expanduser().resolve()
    out = Path(out_root).expanduser().resolve()
    common = {
        "nproc": nproc,
        "sizes": [tile_size],
        "gaps": [gap],
        "rates": [1.0],
        "img_rate_thr": 0.6,
        "iof_thr": 0.7,
        "no_padding": False,
        "padding_value": [104, 116, 124],
        "save_ext": ".png",
    }
    train_label = _find_label_dir(raw / "train")
    val_label = _find_label_dir(raw / "val")
    return {
        "trainval": {
            **common,
            "img_dirs": [
                _slash(raw / "train" / "images"),
                _slash(raw / "val" / "images"),
            ],
            "ann_dirs": [
                _slash(train_label),
                _slash(val_label),
            ],
            "save_dir": _slash(out / "trainval"),
        },
        "test": {
            **common,
            "img_dirs": [_slash(raw / "test" / "images")],
            "ann_dirs": None,
            "save_dir": _slash(out / "test"),
        },
    }


def write_mmrotate_split_configs(configs: dict[str, dict[str, Any]], config_dir: str | Path) -> dict[str, Path]:
    directory = Path(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, payload in configs.items():
        path = directory / f"{name}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written[name] = path
    return written


def _find_label_dir(split_dir: Path) -> Path:
    """Find the label directory under a split (train/val), trying common names."""
    for name in ["labelTxt-v1.0", "labelTxt"]:
        candidate = split_dir / name
        if candidate.exists():
            return candidate
    return split_dir / "labelTxt-v1.0"


def _slash(path: Path) -> str:
    return path.as_posix() + ("" if path.as_posix().endswith("/") else "/")
