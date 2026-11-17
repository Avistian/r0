# Issue tracker: GitHub

Issues and specs for this repo live in GitHub issues on `Avistian/r0`.
Use the `gh` CLI with an explicit `--repo Avistian/r0` for issue operations.
Local Markdown files are drafts, not the issue tracker.

## Conventions

- **Create**: `gh issue create --repo Avistian/r0 --title "..." --body-file <draft-path>`.
- **Read**: `gh issue view <number> --repo Avistian/r0 --json number,title,body,labels,comments`.
- **List**: `gh issue list --repo Avistian/r0 --state open --json number,title,body,labels,assignees`.
  Add `--label` or change `--state` as needed; paginate when completeness matters.
- **Update body**: `gh issue edit <number> --repo Avistian/r0 --body-file <draft-path>`.
- **Comment**: `gh issue comment <number> --repo Avistian/r0 --body-file <draft-path>`.
- **Labels**: `gh issue edit <number> --repo Avistian/r0 --add-label "..."`
  or `--remove-label "..."`.
- **Claim**: `gh issue edit <number> --repo Avistian/r0 --add-assignee @me`.
- **Close**: post the explanation as a comment first, then
  `gh issue close <number> --repo Avistian/r0`. Keep commenting separate from closing,
  including when the issue is already closed.

For multiline bodies, write the exact Markdown to a draft file inside this repo and use
`--body-file`. Keep personal notes and sensitive material out of both issues and commits.
Never file issues into the source skill repository or another upstream project.
This setup records conventions; it does not itself authorize publishing issues or comments.

## Parent tickets and blockers

Prefer GitHub's native sub-issues and dependencies when supported. The installed `gh`
2.62.0 lacks the newer issue relationship flags, so use the GitHub API through `gh`:

- **Database ID**: `gh api repos/Avistian/r0/issues/<number> --jq .id`.
  Relationship endpoints take this numeric database ID, not the issue number or node ID.
- **Attach child**: `gh api --method POST repos/Avistian/r0/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`.
- **Add blocker**: `gh api --method POST repos/Avistian/r0/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`.
- **Read blockers**: `gh api --paginate repos/Avistian/r0/issues/<child>/dependencies/blocked_by`.
  Inspect each blocker's state; a ticket is unblocked only when all blockers are closed.

If those features are unavailable, put `Part of #<parent>` and
`Blocked by: #<number>, #<number>` at the top of the child body, and maintain a child
task list in the parent. Check each referenced blocker's state before starting.
Report failures rather than treating unavailable relationship data as zero blockers.

## Pull requests as a triage surface

**PRs as a request surface: no.**

GitHub issues and PRs share a number space. For an ambiguous reference, resolve its type
before editing it. PRs remain the implementation review surface.

## Engineering skill operations

When `to-tickets`, `triage`, `to-spec`, or `qa` says to publish to the issue tracker,
create or update the appropriate GitHub issue. When a skill says to fetch the relevant
ticket, read its body, labels, and comments using the commands above.

For `wayfinder`, use one issue labelled `wayfinder:map` containing Notes,
Decisions-so-far, and Fog. Link child tickets as sub-issues, or use the fallback task
list. Use `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or
`wayfinder:task` as appropriate. Choose the first open, unassigned child in map order
with no open blockers. Claim it before implementation. On resolution, comment with
the result, close the ticket, and add an evidence link to the map's Decisions-so-far.

## Python implementation and verification

Use `CODING_STANDARDS.md` for implementation conventions. Where an engineering skill
assumes TypeScript tooling, use Python, pytest for tests, and Hypothesis for property
tests. Use local fixtures; tests must not access the network or download data.
Every featurizer change must keep the future-rows property test green.

Use `make check` for the repository verification gate. Before reporting a `src/`
change complete, run in order: `make check`, `make smoke` once T7 lands, and
`make eval-smoke` once T12 lands. Include the smoke loss summary and eval-smoke table
in the PR body, and explicitly report checks that failed or were not run.

Follow all restrictions in `AGENTS.md`, including protected paths and dependency
rules. Never invoke Modal or training cloud APIs, read secrets or `.env`, or download
data. Training and TabPFN-comparison requests use the `agent:train` ticket label and
the guarded workflow described in ADR 0004; do not run them directly.

Edit this guide directly when conventions change. Re-run setup only to switch
trackers or restart the configuration.
