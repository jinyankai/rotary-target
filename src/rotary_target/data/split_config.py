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
    raw = Path(raw_root).resolve()
    out = Path(out_root).resolve()
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
    return {
        "trainval": {
            **common,
            "img_dirs": [
                _slash(raw / "train" / "images"),
                _slash(raw / "val" / "images"),
            ],
            "ann_dirs": [
                _slash(raw / "train" / "labelTxt"),
                _slash(raw / "val" / "labelTxt"),
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


def _slash(path: Path) -> str:
    return path.as_posix() + ("" if path.as_posix().endswith("/") else "/")
