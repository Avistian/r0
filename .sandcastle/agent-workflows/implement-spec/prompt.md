You work in this repository's software factory. Read AGENTS.md,
CODING_STANDARDS.md and GLOSSARY.md. Read SPEC_MODE from the environment.

If SPEC_MODE is `implement`, read /tmp/ticket.md (the ticket and trusted
comments) and /tmp/spec.md (context). Follow the implementation instructions
in .sandcastle/agent-workflows/implement/prompt.md. Build only this ticket,
leave changes in the working tree without staging or committing, and write
/tmp/pr-body.md. The workflow commits through the normal hook, pushes and
opens the draft PR.

If SPEC_MODE is `decompose`, read /tmp/spec.md and turn it into an ordered
queue of tickets. Do not modify or commit repository files. Return only JSON
matching the supplied output schema; the workflow creates the issues.

- Use repository vocabulary and deep modules under src/r0/modules/.
- Each ticket delivers one testable behaviour, with `## What` and
  `## Done when` sections in its body. Include the applicable make checks.
- Order tickets by dependency, using positions starting at 1 and depends_on
  references to earlier positions. Keep the queue between 1 and 12 tickets.
- Choose the schema's shape that fits each ticket; prefer service-module
  for module behaviour and cross-cutting only when splitting is impractical.
- Preserve the repository's training, data, dependency and protected-file rules.

Any other SPEC_MODE is an error: stop and explain it.
