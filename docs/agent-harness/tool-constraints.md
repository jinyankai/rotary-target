# Tool And Artifact Constraints

- Network access may be restricted; do not assume GitHub, Google Drive, or model-host downloads work.
- Ask before downloading large datasets or checkpoints.
- Never commit secrets, credentials, dataset archives, extracted images, or model weights.
- Keep generated reports small and textual unless the user asks for visual deliverables.
- Use exact command output or file paths as evidence when reporting results.

Allowed local artifact roots:

- `data/raw/`
- `data/processed/`
- `outputs/`
- `checkpoints/`
- `work_dirs/`
- `reports/`
