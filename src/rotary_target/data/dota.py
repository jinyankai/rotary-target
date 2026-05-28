from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


DOTA_CLASSES = (
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
)


@dataclass(frozen=True)
class DotaObject:
    points: tuple[float, float, float, float, float, float, float, float]
    category: str
    difficult: int


@dataclass
class LabelFileReport:
    path: Path
    objects: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class DotaValidationReport:
    root: Path
    exists: bool
    checked_label_files: int = 0
    objects: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exists and not self.errors


def parse_label_line(line: str, *, line_number: int = 0, path: Path | None = None) -> DotaObject | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(("imagesource:", "gsd:", "acquisition dates:")):
        return None

    parts = stripped.split()
    location = f"{path}:{line_number}" if path else f"line {line_number}"
    if len(parts) != 10:
        raise ValueError(f"{location}: expected 10 fields, got {len(parts)}")

    try:
        coords = tuple(float(value) for value in parts[:8])
    except ValueError as exc:
        raise ValueError(f"{location}: OBB coordinates must be numeric") from exc

    category = parts[8]
    if category not in DOTA_CLASSES:
        raise ValueError(f"{location}: unknown DOTA v1.0 class '{category}'")

    try:
        difficult = int(parts[9])
    except ValueError as exc:
        raise ValueError(f"{location}: difficult flag must be 0 or 1") from exc
    if difficult not in {0, 1}:
        raise ValueError(f"{location}: difficult flag must be 0 or 1")

    return DotaObject(points=coords, category=category, difficult=difficult)


def validate_label_file(path: str | Path) -> LabelFileReport:
    label_path = Path(path)
    report = LabelFileReport(path=label_path)
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        lines = label_path.read_text(encoding="gbk").splitlines()

    for index, line in enumerate(lines, start=1):
        try:
            obj = parse_label_line(line, line_number=index, path=label_path)
        except ValueError as exc:
            report.errors.append(str(exc))
            continue
        if obj is not None:
            report.objects += 1
    return report


def validate_dota_root(root: str | Path, *, max_label_files: int = 50) -> DotaValidationReport:
    root_path = Path(root).expanduser().resolve()
    report = DotaValidationReport(root=root_path, exists=root_path.exists())
    if not report.exists:
        report.warnings.append("DOTA root does not exist yet. Download is intentionally out of scope.")
        return report

    expected_dirs = [
        root_path / "train" / "images",
        root_path / "val" / "images",
    ]
    test_images = root_path / "test" / "images"
    if test_images.exists():
        expected_dirs.append(test_images)

    for directory in expected_dirs:
        if not directory.exists():
            report.errors.append(f"Missing expected DOTA directory: {directory}")

    label_dir_names = ["labelTxt-v1.0", "labelTxt"]
    label_dirs: list[Path] = []
    for split in ["train", "val"]:
        for name in label_dir_names:
            candidate = root_path / split / name
            if candidate.exists():
                label_dirs.append(candidate)
                break
        else:
            report.errors.append(
                f"Missing label directory under {root_path / split} "
                f"(looked for: {', '.join(label_dir_names)})"
            )

    label_files: list[Path] = []
    for label_dir in label_dirs:
        label_files.extend(sorted(label_dir.glob("*.txt")))

    if not label_files:
        report.errors.append("No DOTA labelTxt files found under train/val.")
        return report

    for label_path in label_files[:max_label_files]:
        file_report = validate_label_file(label_path)
        report.checked_label_files += 1
        report.objects += file_report.objects
        report.errors.extend(file_report.errors)

    if len(label_files) > max_label_files:
        report.warnings.append(
            f"Checked {max_label_files} of {len(label_files)} label files. Increase --max-files for a full scan."
        )
    return report
