# Shared sampling context for issue 28

This checkpoint implements the outer decision scope proposed in
[the integration design](research/0028-pipeline-sampling-integration.md).
It does not yet connect that scope to activation/matrix plans, a graph sampler,
ancestral fitting, saved tasks or the explorer. Those remain required by the
[accepted specification](specs/0028-causal-network-generators.md).

## Public contract

`priors.JointSamplingContext(rng, mode="meta", categorical_policy="shared",
global_policy="scalar-quantile-v1", limits=PreferenceLimits())` owns one numerical
and one categorical sampler. Operations use named, nested transactions:

```python
with context.transaction("graph") as graph:
    with graph.transaction("node-0") as node:
        sigma = node.numerical("sigma", NumericalParameter("gaussian_noise_sigma", 0.0, 1.0))
        node.forced_identity("target-converter", reason="target-warp-disabled")
record = context.freeze()
```

Each numerical/categorical method returns its frozen draw. The forced method
returns its event. Each event has an operation path, unique role and exactly one
typed draw reference, or an explicit identity reason without a random draw.
`JointSamplingRecord` and `JointDecision` are public frozen records;
`joint_sampling_from_dict` reconstructs their JSON representation.

Sharing means the same outer preference index for a name and effective mode,
including across distinct operations. Categorical polynomial weights retain
fresh nested numerical scopes. `shared` and `refresh` categorical policies retain
their existing meanings. Catalog changes under a cached name are rejected.
Request mode `meta` inherits the context mode, as in the existing samplers.

The default global policy caches one scalar unit per name and remaps that unit
into each request's bounds. Global array requests are rejected. This explicitly
declares a scalar/remapping adaptation, not source-global array parity.
`strict-contract-v1` permits arrays but requires identical cached global contracts.
Within a named effective mode, scale, integer conversion and boundary-mass policy
cannot change. Meta/local bounds and shapes can vary; their stochastic preferences
are shared rather than forcing identical realizations.

## Transaction and ownership rules

Both samplers, the ledger and injected RNG are staged together. A failed operation
or nested operation does not commit any of its choices. A rejected primitive draw
is also atomic if its exception is caught within a transaction. No automatic retry
occurs. Parent handles cannot be used while a child is active, and exited handles
are closed. A caller RNG changed externally during a transaction is preserved and
the transaction is rejected. RNG state comparison serializes only trusted memory;
it never reads a pickle or deserializes caller data.

Nonempty completed operation paths cannot be reopened. Empty transactions leave
no event. Decoding requires canonical depth-first paths, unique operation roles
and exactly ordered coverage of every outer draw. Parent events may resume after
child events. Paths have at most 16 components; operation names, roles and reasons
are nonempty, trimmed, NUL-free and at most 128 UTF-8 bytes each.

Generic ownership does not prove semantic destination bindings. For example, the
context cannot prove that a role called `sigma` was assigned to a node's noise.
Future typed plans must check source request names, destination fields, branch
consumption and the exact target group receiving a forced identity.

## Aggregate admission

`PreferenceLimits` applies to the combined scope before native draws allocate
their arrays. Numerical and categorical retained parameters count once, including
categorical nested weights/hyperparameters. Each sampled event adds one numeric
reference. A forced identity counts as one decision against `max_draws` and has
zero numeric values/parameters. Raw decode counts are admitted before child
records allocate tuples or recompute numerical realizations. Both child records
must retain the same overall limits as their aggregate.

This budget covers choice records. It does not cover future materialized matrices,
fit constants, root innovations or live graph buffers; combined causal resource
admission is still required when graph planning/fitting is integrated.

Existing sampler APIs gain read-only admission counts and an optional budget
override on `fork`. Existing default forks, sampling arithmetic and serialized
records are preserved. Existing raw decoder accounting is shared with the joint
preflight without changing its formulas.

## Evidence status at source freeze

The initial 14 public behavior checks covered sharing/JSON, nested rollback, stale
handles, duplicate roles, external RNG mutation with five NumPy bit generators,
both global policies, three exact/below-bound aggregate caps, forced identities
and operation/metadata validation. Nine existing sampler checks also pass.
RED/GREEN logs are retained locally under the Git-ignored
`docs/evidence/issue-28/joint-sampling/`; the retained test file provides the
executed behavior checks. An initial regression command named a nonexistent
categorical test file and ran no tests; its corrected command passed.

The 23-check regression command was executed from `/tmp/r0-28` with the existing
Python 3.12.3 virtualenv and NumPy 2.5.5, without installation or synchronization:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_joint_sampling.py \
  tests/priors/test_parameter_preferences.py tests/priors/test_categorical_preferences.py
```

The wrapper verifies that imports resolve to this checkout. The corrected raw log
is `docs/evidence/issue-28/joint-sampling/r0-28-joint-regression-corrected.log`.
The nine inherited checks are the latter two files; the first contains 14 new
cases. These initial results concern source `2812d23b390a44f5b817c4a2a112de369861ee56`.

Independent review REJECTED that source: a JSON string supplied as an operation
path was coerced into individual characters, changing the ownership hierarchy.
The corrected decoder requires a list/tuple before conversion. Two new public
regressions reject both string and dictionary paths; the same command now passes
25 cases (16 joint, nine inherited). Raw RED and GREEN logs are
`r0-28-joint-path-type-red.log` and `r0-28-joint-path-type-green.log` in the same
local evidence directory. The original malformed input and accepted wrong result
are retained by the independent verifier. This correction does not erase REJECT.

The first full check stopped at Markdown-example formatting. Its corrected run
passed static gates, then was deliberately terminated during pytest after the
independent product defect was found; that run is INCOMPLETE, not a suite pass.

Full repository checks, fresh independent specification and standards reviews,
and exact-head legacy-output comparison: PENDING. No predictive, GPU, hosted,
browser or full issue-completion claim is made by this checkpoint.
