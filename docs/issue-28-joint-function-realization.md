# Issue 28: retained source function construction

All eight source function plans now have numerical construction and bounded JSON
reconstruction through `priors.materialize_joint_function` and
`priors.joint_function_realization_from_dict`. This joins the previously reviewed
[plans](issue-28-joint-function-plans.md), [matrix constructions](issue-28-joint-matrices.md)
and [frozen evaluators](issue-28-frozen-functions.md). It is one component of the
[accepted scope](specs/0028-causal-network-generators.md), not full #28 acceptance.
The [source-law design](research/0028-joint-root-function-plans.md) records cached
primary-source attribution and the intentional freezing/calibration adaptations.

## Construction behavior

| Family | Retained construction and binding |
| --- | --- |
| Linear | Output-by-input normalized matrix construction; no bias. |
| Quadratic | Complete input permutation when width exceeds 20, followed by the matrix batch; otherwise deterministic feature order. Constant coordinate is last; matrices may be nonsymmetric. |
| GP | Product-spectrum uniforms or radial uniforms, Gaussian directions, raw factor and vector weights; phases and Gaussian coefficients. Radial factor weights apply to rows. Zero direction rejects without retry. |
| Tree | Calibration sample, population-deviation feature masses, selected features and threshold rows, Gaussian leaves; strict split bits and averaged tree outputs. |
| Discretization | Calibration sample, full row permutation except for a singleton, owned matrix and center targets. Retains the source singleton, explicit-count and sampled-count cases; first center wins equal-distance ties. |
| EM | Selected rows with replacement, Gaussian anchor jitter and spread draws, calibration assignments, and the output matrix constructed after those assignments. The normalization constant is not multiplied by input dimension. |
| MLP | Ordered matrix constructions, each activation's actual preceding-stage input, fitted constants and shift-row choices, and final calibration outputs. All one-to-three-layer/start/end configurations are supported. |
| Product | Two complete cheap children constructed left then right on the same retained input sample. The right child never fits on the left child's output. |

The five matrix families remain available wherever plans own a normalized matrix.
Source-specific numerical choices remain bound to the named decision ledger.
Raw construction fields are checked against their frozen equations during record
construction and JSON reconstruction. Metadata, sample digests, geometry,
permutations, fitted moments, and ordered child ownership are checked together.

MLP reconstruction recomputes activation constants from the retained stage input
and shift-row choices without sampling an RNG. This shares fitting arithmetic with
`calibrate_source_activation`; query evaluation still applies frozen constants.
Discretization retains center targets, while its frozen evaluator can reproduce
them deterministically from the same centers and linear map. EM fitting and frozen
evaluation share the assignment equation, with explicit failure for nonfinite
probabilities. These operations do not refit from emitted/query rows.

## Calibration, transactions and resource boundary

Fitted functions receive a caller-owned floating calibration array and a separate
fit RNG. Records freeze the supplied sample and its canonical shape/little-endian
float64 SHA-256. MLP records additionally bind stage identities and inputs to their
preceding outputs. Caller mutation after fitting cannot change retained samples.

Construction and fit streams are staged; failures leave caller RNGs and the
frozen named ledger unchanged. Product staging spans both children. This is helper
atomicity, not a complete named/raw/fit/graph transaction or concurrency guarantee.
Raw draws and equations are inspectable; records do not cryptographically certify
an RNG history. The caller remains responsible for establishing sample independence
and graph ancestry. An identity or matching digest alone is not that evidence.

Admission includes planned arrays, retained samples/outputs, raw and derived
copies, nested metadata and the shared ledger. Product admission aggregates both
children and duplicated frozen equations before either child is constructed.
Decoders bound the entire raw payload before conversion. These conservative
estimates are not measured RSS or OS-enforced process limits. Graph-wide live
ancestor/buffer accounting, operation registration and stream manifests remain open.

NumPy float64 random draws and robust arithmetic adaptations do not promise Torch
bit identity. Source-prescribed GP uniform clipping and tree feature-mass floors
remain explicit source laws; numerical failures are not silently retried.

## Verification

Source: `c1ee89e766194993c1a79fcc7400f7b2b48fefbe`.
Comparison base: `4c7c937cc67659d6bbb3365f12da749fb60705d4`.

Full `make check`: **989 passed, one skipped in 483.42 seconds**, including all
static gates. The priors-only run passed 615 tests with one skipped. Focused MLP
checks cover 60 layer/activation-flag/matrix-family combinations plus numerical,
JSON, rollback and admission cases. These are finite implementation checks.
Training/evaluation smoke commands remain unavailable: the CLI accepts replay,
oracle and diagnose-priors, not train/eval. No model loss or evaluation is claimed.

Independent specification review: **PASS_COMPONENT**, no product defect found.
Its source-derived numerical probes cover all eight families, all 25 ordered cheap
products, all 12 MLP layer/flag configurations, 602 numeric-Boolean mutations,
valid-digest stale-stage rejection, late failures/recovery, aggregate/preallocation
admission, and exact baseline comparisons for 29 activation cases and frozen EM.
The independent matrix arithmetic oracle covers Gaussian wrappers and uniform
radial weights; other previously accepted matrix families are reused, not newly
recertified in every composition. Author checks exercise all five matrix families.

Independent standards review: **zero documented violations, two nonblocking
heuristics**. Decoder shape checks and RNG transaction code are duplicated across
families. The retained report identifies exact hunks and recommends shared private
helpers. No functional rejection is hidden by those maintenance observations.
Initial verifier lint failures and the author's failed overflow fixture remain
retained and classified separately from product failures.

## Retained evidence and reproduction

[Spec report](research/0028-joint-function-realization-spec-report.md),
[standards report](research/0028-joint-function-realization-standards.txt),
[full check](research/0028-joint-function-realization-check-c1ee89e.txt),
[development history](research/0028-joint-function-realization-development.json),
[archive metadata](research/0028-joint-function-realization-spec-artifacts.json)
and [reproduction notes](research/0028-joint-function-realization-reproduce.md)
retain inputs, exact executed programs, raw outcomes and identities. Fresh replays
repeat those cases and do not add coverage counts. Published final program copies
live in `scripts/verify_joint_function_realization`.

Restore into a new directory, create a checkout of the pinned source, and run the
portable bundle with an existing compatible environment. The programs verify
HEAD, tracked Python source hashes, actual import location and cached input hashes.

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-joint-function-realization-spec-artifacts.json \
  /tmp/function-construction-evidence
git worktree add --detach /tmp/function-construction-source \
  c1ee89e766194993c1a79fcc7400f7b2b48fefbe
R0_REVIEW_ROOT=/tmp/function-construction-source \
R0_REVIEW_INPUTS=/tmp/function-construction-evidence/portable/inputs \
R0_REVIEW_OUTPUT=/tmp/function-construction-replay \
UV_PROJECT_ENVIRONMENT=/path/to/existing/environment \
  bash /tmp/function-construction-evidence/portable/RUN.sh
```

Original evidence remains unchanged inside `original/`; the portable appendix
retains its own execution history. Cached source attribution is retained, not
refreshed upstream verification. Root realization, graph fitting/ownership and
transactions, remaining paper options, typed variables, interventions, diagnostics,
workbench delivery and browser/Colab/GPU/scale evidence remain required by #28.
