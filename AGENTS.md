# AGENTS.md

This repository is a harness-first scaffold for a rotated object detection course project.

## Operating Contract

- Do not download datasets, pretrained weights, or checkpoints unless the user explicitly asks.
- Do not commit `data/`, `outputs/`, `checkpoints/`, `work_dirs/`, `wandb/`, or model files such as `*.pth`.
- Keep public commands runnable without DOTA, GPU, or MMRotate unless the command is explicitly a real training command.
- Prefer adding new algorithms through the adapter registry and experiment YAML files instead of one-off scripts.

## Canonical Checks

```powershell
python -m pytest
python -m rotary_target doctor --strict
python -m rotary_target train --experiment configs/experiments/roi_trans_r50_dota1.yaml --dry-run
```

## Deeper Docs

- `docs/agent-harness/operating-contract.md`
- `docs/agent-harness/checks.md`
- `docs/agent-harness/tool-constraints.md`
- `docs/agent-harness/review-flow.md`
- `docs/storage.md`
- `docs/data-preparation.md`
- `docs/experiments.md`
