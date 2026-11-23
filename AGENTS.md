# r0 — agents start here

## Verification checks
Environment setup and deliberate reuse: [README](README.md#verification-environments).
Before you report a change to `src/` as complete, run in order:
1. `make check`   2. `make smoke` (once T7 lands)   3. `make eval-smoke` (once T12 lands)
Put the `make smoke` loss summary and the `make eval-smoke` table in the PR body as evidence.

## Rules that are not negotiable
- Never call Modal or any cloud API, never read secrets or `.env`, never download data.
  Modal is the training platform, but you do not invoke it yourself. To start a training or
  TabPFN-comparison run, the ticket gets the `agent:train` label; the guarded workflow holds
  the token and enforces the caps (docs/adr/0004).
- The training path never sees real data (docs/adr/0002). Only modules/relbench_adapter imports relbench.
- Nothing from the future: features and context labels use rows dated strictly before the
  seed time (docs/adr/0005). Every featurizer change keeps the future-rows property test green.
- Never edit eval/suite_v0.toml, eval/suite_v0.lock.json, src/r0/modules/evaluation/,
  scripts/check_ratchet.py, results/best.json or docs/adr/0003-*. If a ticket seems to need it,
  stop and say so in the PR body; the reviewer will label needs-a-human.
- Don't add dependencies unless the ticket says so. Don't copy code from other repos (docs/adr/0006).
- Configs under configs/ci/ stay tiny; tests enforce the caps.
- TabPFN is a baseline only: never use its predictions, probabilities or embeddings as
  training targets, labels, filters or distillation signal (docs/adr/0011).
- This repo is public: never commit secrets, tokens or personal notes.

## Navigation
- Vocabulary: CONTEXT.md (also GLOSSARY.md). Decisions: docs/adr/. Requirements: docs/requirements.md.
- Modules are deep modules: src/r0/modules/README.md.
- Shared skills and the factory's workflows live in the home repo, ~/Projects/ml-factory
  (github.com/Avistian/ml-factory). Files in .github/workflows/ that say "Caller" only call
  the real workflow there: edit it there, not here.

## When something breaks
| Symptom | Cause |
| ------- | ----- |
| `lint-imports` fails on relbench | relbench imported outside modules/relbench_adapter |
| tests fail with SocketBlockedError | a test touched the network; use fixtures |

## Agent skills
### Issue tracker
GitHub issues on `Avistian/r0`, via `gh`. See docs/agents/issue-tracker.md.
