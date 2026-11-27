Standards review of `0341b4c7320d042783b6da1a73e9ad2c28f307b8...47630e8a44268622a4fa9ab352ef4167a3458b4d` only; not overall issue #28 acceptance. Read-only review; tests NOT_RUN.

Documented-standard finding:

- **P2 — Reproduction evidence remains unpublished at this candidate.** `docs/issue-28-shared-matrix-preferences.md:141–142` says “Targeted RED/GREEN logs cover” the categorical, matrix and profile checks, but supplies no links, exact commands/environment, or source/test states for those historical runs. `CODING_STANDARDS.md:26–32` requires identified inputs and the executed procedure for verification claims. External logs do exist: samples inspected at `/tmp/r0-28-categorical-global-{red,green}.log` and `/tmp/r0-28-preferred-profiles-{red,green}.log` show failures and later passes. The owner supplied the command wrapper and intends to publish evidence after review. This finding concerns the pinned document's missing reproduction path, not absent execution. Resolve by linking a retained execution manifest with commands, environment and recoverable RED/GREEN source/test identities, or marking unreconstructable portions unverified.

Judgment-only smell:

- **Possible Duplicated Code, nonblocking.** `src/r0/modules/priors/lib/categorical_preferences.py:283–296` and `matrix_preference_prior.py:63–74` repeat the same named decay/noise requests and bounds, including `"simple_random_weights_decay_rate", 0.1 / math.log1p(width), 6.0`. Their intended distinction is request shape `(1,)` versus scalar. A shared request factory with explicit shape could prevent future law changes drifting between these two copies while preserving their distinct sampling scopes and retained costs.

No other non-tooling documented violation found. Public module boundaries and explicit RNG injection conform. No source copying determination is made from this diff alone. Initial checkout status was only `?? .venv`; final status also showed a concurrent modification to `scripts/causal_topology_census.py`, outside this pinned review. The reviewer made no candidate edits and preserved the symlink.

Total: one P2 evidence-publication finding; one nonblocking smell.
