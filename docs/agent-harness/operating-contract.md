# Repository Operating Contract

This project optimizes for reproducible coursework experiments with minimal accidental state.

- Source files, configs, tests, and docs are tracked.
- Data, checkpoints, logs, and generated visualizations are local artifacts.
- All major actions should have a CLI path under `rt`.
- Commands that require DOTA, MMRotate, or GPU must offer a `--dry-run` mode where practical.
- Claims about model quality must cite metrics files, command logs, or report artifacts.

The initial goal is scaffold reliability, not state-of-the-art accuracy.
