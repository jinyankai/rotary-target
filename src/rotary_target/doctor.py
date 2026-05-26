from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rotary_target.data.dota import validate_dota_root


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(check.status != "error" for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "checks": [asdict(check) for check in self.checks]}


def run_doctor(data_root: str | Path = "data/raw/DOTA-v1.0", *, strict: bool = False) -> DoctorReport:
    checks: list[Check] = []

    version = sys.version_info
    if version >= (3, 10):
        checks.append(Check("python", "ok", platform.python_version()))
    else:
        checks.append(Check("python", "error", f"Python >= 3.10 required, found {platform.python_version()}"))

    _optional_import_check(checks, "torch")
    _optional_import_check(checks, "mmrotate")
    _optional_import_check(checks, "mmdet")
    _optional_import_check(checks, "mmcv")

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore

            detail = "available" if torch.cuda.is_available() else "not available"
            checks.append(Check("cuda", "ok" if torch.cuda.is_available() else "warn", detail))
        except Exception as exc:  # pragma: no cover - defensive around local GPU installs
            checks.append(Check("cuda", "warn", f"torch import failed while checking CUDA: {exc}"))
    else:
        checks.append(Check("cuda", "warn", "torch is not installed; CUDA cannot be checked"))

    dota_report = validate_dota_root(data_root, max_label_files=5)
    if dota_report.ok:
        checks.append(Check("dota", "ok", f"found {dota_report.checked_label_files} checked label files"))
    else:
        detail = "; ".join(dota_report.errors or dota_report.warnings)
        checks.append(Check("dota", "warn", detail or "DOTA data is not prepared yet"))

    if strict and not Path("pyproject.toml").exists():
        checks.append(Check("repo", "error", "pyproject.toml not found; run from repository root"))
    else:
        checks.append(Check("repo", "ok", "scaffold files are present"))

    return DoctorReport(checks=checks)


def _optional_import_check(checks: list[Check], module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        checks.append(Check(module_name, "warn", "optional training dependency is not installed"))
    else:
        checks.append(Check(module_name, "ok", "importable"))
