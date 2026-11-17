# 0010 — The factory lives in the home repo, and r0 calls it

Shared skills and the factory's workflows live in the public home repo ml-factory
(github.com/Avistian/ml-factory), and every agent session starts there, following Matt's home
repo pattern. The workflows are GitHub reusable workflows (on: workflow_call) in
ml-factory/.github/workflows/. r0 keeps its own AGENTS.md, build commands, CONTEXT.md, ADRs, code,
scripts and prompts, plus short caller workflows whose job is
`uses: Avistian/ml-factory/.github/workflows/<name>.yml@<tag or SHA>`. A public repo may call
reusable workflows stored in a public repo, so this works while both are public.

## Considered Options
- The whole factory inside r0: every new project would copy it again.
- Synced copies (a script copies the files into r0 and a PR merges them): needed only if
  ml-factory ever goes private while r0 stays public.
- A monorepo: would mix the factory and the project.
- Reusable workflows called from r0 (chosen).

## Consequences
- Workflow logic is edited only in ml-factory; r0's callers pin a tag or commit SHA, and moving
  to a new version is a reviewed PR in r0 that changes the ref.
- Secrets stay in r0 and are passed by name from each caller. ml-factory holds no secrets.
- ml-factory is public: no secrets, tokens or personal notes are ever committed there.
- Decisions that affect the whole system keep a copy in ml-factory/docs/adr/.
