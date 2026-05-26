# Review Flow

Before opening a PR or handing off changes:

1. Run the canonical checks in `docs/agent-harness/checks.md`.
2. Confirm no ignored heavy artifact was force-added.
3. Inspect `git status --short`.
4. Summarize changed behavior, not just changed files.
5. List missing heavy checks separately, especially any training/evaluation not run.

For experiment changes, include:

- Experiment YAML path.
- Algorithm key and backend.
- Dataset root convention.
- Dry-run command evidence.
- Any metric file path if a real run was completed.
