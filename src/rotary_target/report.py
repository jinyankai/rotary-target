from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSummary:
    name: str
    path: Path
    metrics: dict[str, Any]


def collect_runs(outputs_root: str | Path = "outputs") -> list[RunSummary]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    summaries: list[RunSummary] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {"error": "metrics.json is not valid JSON"}
        name = metrics_path.parent.name
        summaries.append(RunSummary(name=name, path=metrics_path.parent, metrics=metrics))
    return summaries


def render_markdown(summaries: list[RunSummary]) -> str:
    lines = [
        "# Rotated Object Detection Experiment Summary",
        "",
        "This report is generated from `metrics.json` files under the outputs directory.",
        "",
    ]
    if not summaries:
        lines.extend(
            [
                "No completed experiment metrics were found yet.",
                "",
                "Expected each run to write a file such as `outputs/mmrotate/<experiment>/metrics.json`.",
            ]
        )
        return "\n".join(lines) + "\n"

    lines.append("| Experiment | mAP | Notes |")
    lines.append("| --- | ---: | --- |")
    for summary in summaries:
        metric = summary.metrics.get("mAP", summary.metrics.get("map", ""))
        notes = summary.metrics.get("notes", "")
        lines.append(f"| {summary.name} | {metric} | {notes} |")
    lines.append("")
    return "\n".join(lines)
