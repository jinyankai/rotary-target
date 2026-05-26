# Algorithm Adapter Design

The adapter layer keeps coursework experiments comparable across algorithms and backends.

## Current Backend

`MMRotateAdapter` maps a repo-level experiment YAML to MMRotate commands:

- `train_command`
- `eval_command`
- `visualize_command`
- `split_dota_command`

Default upstream config paths are declared in `src/rotary_target/algorithms.py`.

## Adding an Algorithm

1. Add an `AlgorithmSpec` entry in `src/rotary_target/algorithms.py`.
2. Add `configs/experiments/<algorithm>_dota1.yaml`.
3. Add a dry-run test if the command shape differs.
4. Document expected metric output in `docs/experiments.md`.

## Adding a Backend

1. Implement the adapter protocol in `src/rotary_target/adapters/`.
2. Register it in `src/rotary_target/registry.py`.
3. Keep dry-run behavior dependency-free.
4. Do not make CI install heavyweight GPU dependencies.
