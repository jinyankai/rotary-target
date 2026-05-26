from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from rotary_target.config import ExperimentConfig


@dataclass(frozen=True)
class CommandSpec:
    argv: list[str]
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""

    def shell_line(self) -> str:
        return subprocess.list2cmdline(self.argv)


class AlgorithmAdapter(Protocol):
    name: str

    def train_command(
        self,
        experiment: ExperimentConfig,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        ...

    def eval_command(
        self,
        experiment: ExperimentConfig,
        checkpoint: str | Path | None = None,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        ...

    def visualize_command(
        self,
        experiment: ExperimentConfig,
        image: str | Path,
        checkpoint: str | Path,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        ...
