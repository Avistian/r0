# Issue 28: joint source root plans

Root-base selection now composes with the shared named decision context and the
required source function subtree. This is a planning component of the
[accepted scope](specs/0028-causal-network-generators.md), not generated root values
or complete #28 delivery. The [root/function research](research/0028-joint-root-function-plans.md)
retains source laws and cached-source attribution.

## Behavior

`JointRootPrior`, `JointRootPlan`, `sample_joint_root` and
`joint_root_plan_from_dict` are available through `priors`.
The default `random_base_points` catalog preserves source order:
uniform, unit ball, Gaussian, covariance. The source calls its ball choice
`RandomCirclePoints`; the plan names its actual distribution. Explicit nonempty,
distinct catalog subsets are an inspectable r0 exploration option.

Every root owns one full function plan at its `function` child operation. Input
and output widths both equal the root width, and the function retains the root's
independent calibration row count. A root cannot substitute a cheap selector or
omit the function. All eight function families and all five matrix families remain
available through the shared function prior.

Covariance roots first own a raw-factor weight plan of batch one and root width,
then select uniform or Gaussian through `random_covariance_base_points`, then
sample the complete function subtree. Their weights inherit the configured
vector family and scalarization policy from the function's matrix prior. These
weights describe a future raw covariance factor; they do not introduce a
normalized-matrix selector for that factor. Event-free uniform weights still have
a structural child in the root plan. Other roots reject covariance-only fields.

Plans bind their exact selector requests, values, geometry, prior and ordered
child events to a frozen `JointSamplingRecord`. The enclosing context transaction
restores both named decisions and the caller RNG if any nested branch fails.
Existing local/meta/global contracts remain in force, including documented
restrictions of strict versus scalar-quantile and source-batch policies.

## Reconstruction and resource boundary

JSON reconstruction bounds raw metadata before conversion, rejects unknown nested
records and fields, and delegates function/weight validation to their owners.
Both stored function-prior copies contribute to raw numeric admission. Caller
limits override payloads. A simple two-channel linear root has 14 raw numeric
leaves; the author check accepts 14 and rejects 13.

The function-prior reconstruction helper is shared with the function decoder; it
checks nested prior structure itself. This preserves the previous function-plan
contract while preventing permissive coercion in root-prior reconstruction.

These plans do not allocate represented matrices or root samples. Raw array and
calibration admission, graph-wide registration of event-free operations, full
stream manifests, independent ancestral fitting and whole-graph transactions
remain required. Metadata gates do not establish generated-root or graph-memory
acceptance.

## Validation

Source: `bc96787f5a423179b0b5b413f1d096ce81ef1a0f`.
Comparison base: `8f3366a80bf7abb80f0d76e45e92031ea774234e`.

Full `make check`: **887 passed, one skipped in 457.92 seconds**, all static gates.
Focused root/function checks: 17 passed. They exercise all four base choices,
both covariance bases and weight families, mandatory function ownership,
roundtrip/unknown-field rejection, aggregate raw counts and nested failure
rollback. Training/evaluation smoke commands remain unavailable in the CLI;
no model loss or evaluation result is claimed.

Independent standards review: zero documented violations and zero actionable
smell findings. Independent specification review: **PASS_COMPONENT**, with no scoped findings.
The retained public probes cover 320 base/function/matrix/weight combinations,
15 AST-derived source request laws, all 12 MLP layer/flag combinations, both
covariance bases and GP spectra, complete catalogs, ordering and JSON corruption.
Exact metadata boundaries include a 16-number/3187-estimated-byte fixture;
12 sequential named-event cutoffs verify rollback. Independent and author
fixtures differ; these are separate finite checks, not measured RSS or scientific
experiments. Two initial runtime harness errors remain retained as such; no
product rejection was found in this component review.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

## Remaining work

The [function-plan checkpoint](issue-28-joint-function-plans.md) and
[frozen numerical evaluators](issue-28-frozen-functions.md) remain included.
Raw function and root construction, fitting/provenance, graph integration,
remaining accepted paper options, typed states, interventions, diagnostics and
local/Colab workbench delivery remain required. No browser, Colab, GPU, scale or
predictive acceptance follows from this component's checks.

## Retained evidence and replay

[Independent report](evidence/issue-28/legacy-research/0028-joint-root-plans-spec-report.md),
[reproduction instructions](research/0028-joint-root-plans-reproduce.md),
[standards review](evidence/issue-28/legacy-research/0028-joint-root-plans-standards.txt),
[full check](evidence/issue-28/legacy-research/0028-joint-root-plans-check-bc96787.txt),
[author history](evidence/issue-28/legacy-research/0028-joint-root-plans-development.json) and
[independent archive metadata](evidence/issue-28/legacy-research/0028-joint-root-plans-spec-artifacts.json)
retain the exact inputs, programs, fixtures and raw outcomes. Fresh portable
replays repeat existing cases and do not increase coverage counts.

Restore the archive into a new directory, then follow its `REPRODUCE.md` from a
checkout of the pinned source. Programs require new output directories, verify
source/import/input hashes and accept `R0_REVIEW_INPUTS` for restored inputs.
The supplemental program reads the fresh main probe's fixtures. Published program
copies are in `scripts/verify_joint_root_plans`.

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-joint-root-plans-spec-artifacts.json \
  /tmp/root-plan-evidence
```

Cached input attribution is retained; upstream was not refreshed. The recovery
receipt verifies every archived identity. Re-running the probes requires the
same established compatible environment used for repository checks.
