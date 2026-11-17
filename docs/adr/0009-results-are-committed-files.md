# 0009 — Results are committed files

Run records (JSON), results/leaderboard.csv, results/best.json and results/budget-ledger.csv are
committed and reviewed in PRs, so every number has a diff, a reviewer and a git SHA.

## Considered Options
- An MLflow server: another service to run and secure, and results outside review.
- Committed files (chosen).

## Consequences
- No MLflow server. Langfuse, if ever used, is for agent traces, not training runs.
- Checkpoints never go into git (the hook refuses files over 1 MB); they go to Releases.
