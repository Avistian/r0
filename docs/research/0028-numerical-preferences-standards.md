# Standards review — numerical preferences

Reviewed `git diff 12ad1907d772fe23fae2ad907ac5ffd9177e755f...593c924f5e2ea8da31d4c34d9da430aef4b38489` across all five changed files. Read-only preflight confirmed candidate HEAD and only untracked `.venv`; no checkout, repository edits, environment access or tests performed.

## Documented-standard violations

None identified in this delta. `CODING_STANDARDS.md:1–3` requires injected randomness: sampler draws use its injected Generator. `src/r0/modules/README.md:2–5` requires public entry-point access, including tests: exports and tests comply. No protected evaluation paths, dependencies, downloaded data, or external implementation copies are introduced by the diff. Clean-room provenance is stated, but independent authorship cannot be established from a diff alone.

Standards examined: r0 `AGENTS.md`, `CODING_STANDARDS.md`, module README, `CONTEXT.md`, `docs/requirements.md`, repository README, ADR 0006 and relevant ADR/agent guidance. The home repository's TypeScript course reference was excluded as instructed by its `AGENTS.md`.

## Heuristic findings

**Possible Duplicated Code / Shotgun Surgery — low priority.** `src/r0/modules/priors/lib/parameter_preferences.py:110–112,147–149,257–260,339–347` independently encode retained-record accounting. The record uses `self.request.parameter_count + 1 + len(self.units) + len(self.values)`; admission uses `request.parameter_count + 1 + 2 * width`; raw decoding uses `3 + len(d["request"].get("shape", ())) + len(d["units"]) + len(d["values"])`. Preference accounting is likewise repeated. Adding one retained numeric field requires synchronized changes to all three representations or live sampling and decoding can apply different budgets. Consider one dimension-based accounting helper usable before allocation, with separate callers supplying validated/raw dimensions. This is a maintenance judgment, not a documented-standard breach; pre-allocation admission itself should remain intact.

No additional actionable baseline smells identified in exports, limit configuration, tests, or documentation. Tooling-enforced matters were excluded. This is a scoped standards review, not issue #28 acceptance or evidence that pending full checks passed.
