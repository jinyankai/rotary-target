---
name: rotary-target
description: Project-specific workflow for the rotated object detection coursework scaffold.
---

# Rotary Target Skill

Use this skill when modifying this repository.

## Workflow

1. Run `python -m rotary_target doctor --strict` to understand local readiness.
2. Add algorithms through `src/rotary_target/algorithms.py` and experiment YAML files.
3. Keep dry-run behavior working without DOTA, MMRotate, CUDA, or checkpoints.
4. Add or update tests under `tests/` for config parsing, CLI dry-runs, or DOTA label handling.
5. Do not download or commit datasets/checkpoints unless explicitly requested.

## Canonical Checks

```powershell
python -m pytest
python -m rotary_target doctor --strict
python -m rotary_target train --experiment configs/experiments/roi_trans_r50_dota1.yaml --dry-run
```
