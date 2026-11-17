# 0004 — Agents may start compute only through the guarded lane

Modal is the training platform. Codex (or I) may start pretraining and TabPFN-comparison evals
only through agent-train.yml (agent:train label or workflow_dispatch, concurrency 1). A dedicated
Modal workspace token lives only in Actions secrets I add myself, never on pull_request or fork
workflows. The course keeps paid side effects behind a human; here the human sets the money caps
and still merges every results PR. This is a compute-only deviation from the course.

## Considered Options
- No agent-started compute at all: safest, but every run waits for me.
- Agents call Modal directly: fastest, but the token and the spend sit inside the agent's reach.
- A guarded lane (chosen): rules in code, not in prompts.

## Consequences
- Hard monthly spend limit in the Modal dashboard; per-run caps in the workflow and in code
  (scripts/check_compute_caps.py); COMPUTE_ENABLED kill switch; CPU smoke before any paid run;
  a budget ledger (results/budget-ledger.csv).
- Implement and review workflows may run only make check, smoke and eval-smoke;
  tests/workflows/test_workflows.py enforces it. Merges stay human.
