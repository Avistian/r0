## Standards

Reviewed `9871fdaf7037c7ed9b09926f5073f63f5cf6ff31...7eea0bd5ab6f553aa07f014ed5a353b4d0db69e9` in `/tmp/r0-28` against `AGENTS.md`, `CODING_STANDARDS.md`, `src/r0/modules/README.md`, and `docs/agents/protocol-review.md`. Scope is the partial increment documented in `docs/issue-28-matrix-families.md`.

**Hard violations: none found.** New draws receive an explicit generator; external consumers/tests use public module entry points; no dependency or protected-file changes appear. Completion language keeps broader issue requirements and pending evidence visible. Tool-enforced concerns were excluded.

**Judgment calls: one nonblocking possible Duplicated Code smell.** `src/r0/modules/priors/lib/causal_matrix_sampling.py:240–245` and `src/r0/modules/priors/lib/kernel_points.py:106–114` separately implement positive-vector probability normalization, including the identical zero-mass rejection. Both contain `largest = max(weights)` and `scaled = tuple(v / largest for v in weights)`; sampling then uses `tuple(v / math.fsum(scaled) for v in scaled)`, while the record computes the same total once. A shared helper would keep sampling and retained-record validation consistent if the normalization policy changes. This is a maintainability suggestion, not a documented-standard breach or correctness blocker.

Preflight and final inspection both found HEAD at the requested commit with only the existing untracked `.venv`. No checkout, repository edit, environment mutation, or test execution was performed; the parent owns full verification. Existing profile-registry repetition is not treated as a new blocker.

Standards: 0 hard violations; 1 nonblocking judgment call (possible Duplicated Code).
