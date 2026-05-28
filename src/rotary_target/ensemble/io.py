"""Prediction I/O for ensemble fusion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Prediction:
    cx: float
    cy: float
    w: float
    h: float
    theta: float
    score: float
    class_id: int

    def as_tuple(self) -> tuple:
        return (self.cx, self.cy, self.w, self.h, self.theta)


def load_predictions(path: str | Path) -> list[Prediction]:
    """Load predictions from a JSON file.

    Expected format: list of dicts with keys cx, cy, w, h, theta, score, class_id.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Prediction(**item) for item in data]


def save_predictions(predictions: list[Prediction], path: str | Path) -> Path:
    """Save predictions to a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in predictions]
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out
