from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


COMMANDS = [
    [sys.executable, "-m", "pytest"],
    [sys.executable, "-m", "rotary_target", "doctor", "--strict"],
    [
        sys.executable,
        "-m",
        "rotary_target",
        "train",
        "--experiment",
        "configs/experiments/roi_trans_r50_dota1.yaml",
        "--dry-run",
    ],
]


def main() -> int:
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1] / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing
    for command in COMMANDS:
        print("+ " + subprocess.list2cmdline(command), flush=True)
        completed = subprocess.run(command, check=False, env=env)
        if completed.returncode != 0:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
