from __future__ import annotations

import os
import sys
from pathlib import Path

from rotary_target.adapters.base import CommandSpec
from rotary_target.algorithms import get_algorithm
from rotary_target.config import ExperimentConfig


class MMRotateAdapter:
    name = "mmrotate"

    def train_command(
        self,
        experiment: ExperimentConfig,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        root = self._resolve_backend_root(experiment, backend_root)
        config_path = self._resolve_config(experiment, root)
        tool = self._tool_path(root, "train.py")
        argv = [sys.executable, str(tool), str(config_path), "--work-dir", str(experiment.work_dir)]
        launcher = str(experiment.training.get("launcher", "none"))
        if launcher and launcher != "none":
            argv.extend(["--launcher", launcher])
        resume_from = experiment.training.get("resume_from")
        if resume_from:
            argv.extend(["--resume-from", str(resume_from)])
        return CommandSpec(
            argv=argv,
            cwd=root,
            description=f"Train {experiment.algorithm} with MMRotate",
        )

    def eval_command(
        self,
        experiment: ExperimentConfig,
        checkpoint: str | Path | None = None,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        root = self._resolve_backend_root(experiment, backend_root)
        config_path = self._resolve_config(experiment, root)
        tool = self._tool_path(root, "test.py")
        checkpoint_path = checkpoint or experiment.evaluation.get("checkpoint")
        if not checkpoint_path:
            checkpoint_path = experiment.work_dir / "latest.pth"
        metric = str(experiment.evaluation.get("metric", "mAP"))
        return CommandSpec(
            argv=[
                sys.executable,
                str(tool),
                str(config_path),
                str(checkpoint_path),
                "--eval",
                metric,
            ],
            cwd=root,
            description=f"Evaluate {experiment.algorithm} with MMRotate",
        )

    def visualize_command(
        self,
        experiment: ExperimentConfig,
        image: str | Path,
        checkpoint: str | Path,
        backend_root: str | Path | None = None,
    ) -> CommandSpec:
        root = self._resolve_backend_root(experiment, backend_root)
        config_path = self._resolve_config(experiment, root)
        tool = root / "demo" / "image_demo.py"
        return CommandSpec(
            argv=[
                sys.executable,
                str(tool),
                str(image),
                str(config_path),
                str(checkpoint),
                "--out-file",
                str(experiment.work_dir / "visualizations" / "demo.jpg"),
            ],
            cwd=root,
            description=f"Visualize {experiment.algorithm} detection output",
        )

    def split_dota_commands(
        self,
        split_config_paths: list[str | Path],
        backend_root: str | Path | None = None,
    ) -> list[CommandSpec]:
        root = Path(backend_root or os.environ.get("MMROTATE_ROOT", "<MMROTATE_ROOT>"))
        tool = self._tool_path(root, "data/dota/split/img_split.py")
        return [
            CommandSpec(
                argv=[sys.executable, str(tool), "--base-json", str(config_path)],
                cwd=root,
                description=f"Split DOTA images with {config_path}",
            )
            for config_path in split_config_paths
        ]

    def _resolve_backend_root(
        self,
        experiment: ExperimentConfig,
        backend_root: str | Path | None = None,
    ) -> Path:
        raw = backend_root or experiment.backend.get("root") or os.environ.get("MMROTATE_ROOT")
        return Path(raw) if raw else Path("<MMROTATE_ROOT>")

    def _resolve_config(self, experiment: ExperimentConfig, backend_root: Path) -> Path:
        explicit = experiment.training.get("config")
        if explicit:
            config = Path(str(explicit))
            return config if config.is_absolute() else Path.cwd() / config
        spec = get_algorithm(experiment.algorithm)
        return backend_root / "configs" / spec.upstream_config

    @staticmethod
    def _tool_path(root: Path, relative_tool: str) -> Path:
        return root / "tools" / relative_tool
