# Configs

`configs/experiments/` contains repo-level experiment manifests. They are intentionally small and backend-agnostic enough for the `rt` CLI to validate, dry-run, and dispatch work.

The MMRotate adapter maps `algorithm` keys to upstream MMRotate config paths. If your local MMRotate checkout uses a different branch or config naming convention, set `training.config` in an experiment YAML to point to the exact config file.
