# Retained independent standards reports

# Independent Standards review: factorized checkpoint

Head: `ec522eddc704a37679906a5e1c1274fea3ae65ad`.
Base: `5562ca0f2f5f9628c5386c02755dae9229b778e4`.
Reviewed `git diff 5562ca0f2f5f9628c5386c02755dae9229b778e4...ec522eddc704a37679906a5e1c1274fea3ae65ad` and seven commits: `04f8e1c`, `559d2af`, `a14f620`, `fa26bcb`, `dc84af8`, `40e002c`, `ec522ed`.

Preflight and final inspection: `/tmp/r0-28`, expected HEAD, only existing untracked `.venv`. No checkout edits, tests, cloud/network calls, downloads, dependency operations or secrets access. Only this external review artifact was written.

Read standards: `AGENTS.md`, `CODING_STANDARDS.md`, `src/r0/modules/README.md`, `docs/agents/protocol-review.md`, `README.md`; context: requirements, accepted specification and checkpoint document. Applied the delegated Standards axis of the code-review skill; tooling-enforced checks were excluded.

## Documented violations

None identified in this bounded static review. New random draws use the injected Generator (`CODING_STANDARDS.md:1-3`); normalization and query metrics are unchanged. No protected paths or dependencies changed. Public test imports respect the module-entry-point rule.

`FactorizedWeights` validates nested tuples and finite numeric leaves (`weight_matrices.py:13-42`); frozen dataclasses therefore do not conceal mutable factor lists. `MatrixCausalNode` validates shape, and `CausalRecipe` admits resource dimensions before calling `materialize()` and checking dense agreement (`causal.py:116-128`). These are static observations, not executed adversarial checks.

## Nonblocking heuristic

Possible **Duplicated Code / Feature Envy**, `src/r0/modules/priors/lib/causal_weight_sampling.py:21-36`: `+ (7 + rank * (len(node.weights) + len(node.bias) + 1) if rank else 0)` and `(8 * width ... if rank)` reproduce stochastic-node accounting. The 7 equals current default exogenous count 4 plus response count 3; it is added once per nonroot, so no current node-count error was found. `StochasticCausalNode` owns the corresponding formulas (`causal_nodes.py:90-96`). A shared dimension-only accounting helper would prevent future defaults from drifting between preallocation admission and final node admission. This is a maintenance judgement, not a documented-rule violation.

The existing duplicated profile registry acquires another entry; no new mismatch was identified, and this is not counted as a new finding.

Checkpoint/source language remains explicitly partial; required source sampling laws and actual browser/Colab execution remain pending. Authored test-pass claims were not independently replayed; full gates remain pending. This review does not establish external source fidelity or overall issue #28 acceptance.


---

# Scoped Standards recheck

Exact head: `ec39b46ef7972a5729735850cb126e6d6f53fad9`.
Scope: `git diff ec522eddc704a37679906a5e1c1274fea3ae65ad..ec39b46ef7972a5729735850cb126e6d6f53fad9`, comprising `209b4e6`, `ef1bacc`, `ec39b46` only. Earlier review retained at `/tmp/r0-28-factorized-standards-ec522ed.md`.

Preflight and final status: `/tmp/r0-28`, exact expected HEAD, only existing untracked `.venv`. No tests, network/cloud calls, downloads, secrets access, dependency operations, checkout edits or commits. This external report is the only write.

## Resolution of previous heuristic

RESOLVED for this checkpoint: the duplicated stochastic admission constants are removed from `causal_weight_sampling.py:26-50`. `default_noise_parameters = StochasticCausalNode().structural_parameter_count` derives current default metadata overhead from its owner. Both node admission and sampling now call `StochasticCausalNode.working_channels_for_width` (`causal_nodes.py:94-100`). Defaults currently carry zero activation parameters, so existing node activation parameters are counted once. This resolves the reported maintenance concern; it is not a claim that arbitrary future changes to defaults require no review.

## New documented violations and heuristics

None identified in this bounded static delta review against the previously read repository standards. No new actionable smell was identified. The historical duplicated profile registry remains nonblocking and is outside this delta.

The new `exact_terms_limit` is a frozen integer record field with explicit positive, bounded validation (`weight_matrices.py:33,40-41`), preserved by the existing dictionary decoder. Exact evaluation checks `rows * columns * rank` before constructing its Decimal product (`91-118`); dimensions follow already validated rectangular factors. Terms are streamed and results remain tuple matrices; nonfinite outputs fail explicitly. The Decimal context is local, and the change introduces no randomness, external I/O, dependencies, normalization or metric changes. Existing recipe resource admission still precedes `materialize()`.

The new tests retain overflow, underflow and cancellation examples through public recipe decoding/generation. They were read, not executed. Exponent-bound completeness, precision proof, numerical correctness, runtime cost and supported-size evidence remain for the separate mathematical/specification reviewer and execution evidence. This review does not convert tests into a scientific or resource acceptance claim.

Result: prior resource-accounting heuristic resolved; zero new hard Standards findings. No overall issue #28 acceptance or full-check claim.
