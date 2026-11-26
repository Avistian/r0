## Standards

Reviewed `git diff 977461c276c722592cac7bdd271d9250b93db972...e6d8faa4b2b8c2de54a7ab62c6235357730293c0` against `AGENTS.md`, `CODING_STANDARDS.md`, `src/r0/modules/README.md`, and `docs/agents/protocol-review.md`. Preflight and final checks identified HEAD `e6d8faa4b2b8c2de54a7ab62c6235357730293c0`; only the pre-existing untracked `.venv` was present. Checkout and environment were untouched. Full checks are owned by the parent reviewer and were not duplicated.

**Hard violations: none found.** The new sampling helpers receive an explicit `np.random.Generator`. The scope document distinguishes partial implementation, pending verification, and unmeasured scientific/browser/CUDA claims. Tooling-enforced items were excluded from this axis.

**Heuristic concern (nonblocking): possible Repeated Switches.** Adding `"causal-importance-v1"` requires updating the family map and separate causal-profile membership tuple in `src/r0/modules/priors/lib/recipe_sampling.py:321,331`, then the supported-profile tuple, width-selection membership, activation-selection membership and dispatch in `src/r0/modules/priors/lib/causal_sampling.py:60,89,97,102`. The repeated hunks include `if profile in (...)` and `if profile == "causal-importance-v1"`. The existing pattern has grown another independently maintained profile definition; a future profile can be accepted by one path but omitted by another. Consider consolidating profile metadata/dispatch into one private mapping, preserving existing random-draw ordering. This is a judgement call under the skill's smell baseline, not a documented-standard breach.

Standards result: 0 hard violations; 1 nonblocking heuristic concern. This review does not establish scientific efficacy or completion of all issue #28 scope.
