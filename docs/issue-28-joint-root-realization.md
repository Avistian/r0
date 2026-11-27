# Issue 28: retained root construction and frozen-function emission

Source checkpoint: `78632956b1caf236ba521397d564311e2b006465`.
Base: `2488161135eca07374a8e433c9b76de81f573e5d`.
This is a component checkpoint. Whole issue #28 remains incomplete.

## Implemented behavior

`materialize_joint_root` realizes each source root base followed by its mandatory
frozen d-to-d function. All four base families compose with all eight function
families. The record retains raw base construction, raw calibration draws,
transformed calibration inputs, function realization, calibration identity and
outputs. Three explicitly injected generators separate construction, calibration
inputs and fitting choices; aliasing their bit generators is rejected. All are
staged and committed only after the complete root succeeds.

Uniform inputs use [-1,1); Gaussian inputs are standard normal; ball inputs use
row-normalized Gaussian directions and independent U^(1/d) radii. Covariance
roots use raw A diag(w), weighting columns, and a uniform or Gaussian base. They
do not use a normalized matrix wrapper. Both uniform and polynomial weight laws
are supported. Zero/singular covariance factors and zero ball radii are valid;
a zero ball direction is undefined and rejected without a retry.

`sample_joint_root_values` emits fresh rows from the frozen base and function,
retaining the raw sample and final outputs alongside the root. It never refits
the function. Its sample identity must differ from the calibration identity.
Construction and emission decoders validate raw payload bounds, planned geometry,
versions, fields, ownership, and the raw/derived equations. Root and emission
records include the complete calibration provenance; they are diagnostic records,
not a predictor-facing data format.

The base sampler draws all Gaussian directions before all ball radii. Consequently
partitioning one sequential RNG stream into different calls is not a promise of
identical raw rows. Fixed-function evaluation remains independent of which query
batch it sees. Named stream manifests, original-row replay and graph-wide stream
ownership remain subsequent integration work. Distinct RNG objects and caller
identities do not alone certify independent seeds or calibration provenance.

Resource gates cover retained records and conservative working-array estimates,
including function evaluation at calibration and emission sizes. They are not
measured process RSS or OS-enforced limits. Helper staging is not graph-wide
transactionality. Finite calibration samples do not certify population moments.

## Verification status

- Author focused checks: 109 passed in 3.04s; raw log
  [focused log](evidence/issue-28/joint-root-realization/r0-28-root-realization-final-focused.log).
- Retained genuine REDs: missing realization decoder, missing emission API,
  missing emission decoder. GREENs and intermediate lint/type/fixture-pattern
  corrections remain in separate development logs.
- Checks include all32 base/function combinations for construction/emission and
  JSON reconstruction; both covariance base/weight laws; nested MLP/product with
  all5 matrix families; raw draw orientation/order; frozen-root preservation;
  tampered fields, Boolean numeric leaves, fit identities and outputs; caller
  resource caps; valid zero factors/radii; and actual overflow rollback/recovery.
- Full `make check` at exact source HEAD: **1098 passed, 1 skipped**, 480.12s.
  Formatting, lint, typing, import boundaries, module tests and file-size gates passed.
  [Retained full check](evidence/issue-28/legacy-research/0028-joint-root-realization-check-7863295.txt).
- Independent Spec review: **ACCEPT component**, no blocking findings. Independent
  equations for seven function families; MLP standalone-fit equivalence and fixed-row
  chunk checks reuse the previously accepted MLP algebra. Literal base oracles, all32
  combinations, four covariance cases, corruptions, aliases, fault-injected post-fit
  rollback and genuine overflow recovery are retained.
  [Spec report](evidence/issue-28/legacy-research/0028-joint-root-realization-spec-report.md).
- Independent Standards review: no hard violations; two nonblocking possible
  duplication findings, covering RNG transactions and base-sample decoding.
  [Standards report](evidence/issue-28/legacy-research/0028-joint-root-realization-standards.txt).
- Independent observed minimum admission thresholds for a bounded linear fixture:
  child/root/emission parameters 53/138/158 and working bytes 11748/32208/67264.
  Rejection below each aggregate threshold establishes gate behavior, not RSS.
- Retained inputs, executed programs, attempts and public-program replay are in the
  [spec archive metadata](evidence/issue-28/legacy-research/0028-joint-root-realization-spec-artifacts.json);
  [recovery receipt](evidence/issue-28/legacy-research/0028-joint-root-realization-spec-recovery.json).
  [Development log metadata](evidence/issue-28/legacy-research/0028-joint-root-realization-development.json)
  preserves author RED/GREEN, formatting/type repairs and full-check evidence.

## Remaining issue scope

Graph registry/stream ownership, graph-wide transactions, topological ancestral
calibration, root/function node attachment, manifests and saved task integration
remain required. Root function and pipeline/noise/importance must precede fitting
descendants. The existing source fitter cannot simply receive new roots after
its current affine-Gaussian fit. Other paper-specific graph/tree families, typed
categorical/binary variables, interventions, the local/Colab workbench, parallel
storage/restart, diversity and supported-scale evidence also remain required.
No browser, Colab, GPU, cloud, training or downstream predictive-quality evidence
is claimed by this root checkpoint.

## Reproduction

Recover the archive into a new directory with:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-joint-root-realization-spec-artifacts.json /tmp/root-evidence
```

At the exact source checkpoint, use its existing environment with the recovered
`original/review.py`, setting `R0_REVIEW_ROOT` to that checkout and `PYTHONPATH` to
its `src`. Each run creates a fresh output directory alongside the verifier. The
[retained instructions](research/0028-joint-root-realization-reproduce.md) describe
the source/input pins and oracle limitations. Public copies of the exact executed
programs live in `scripts/verify_joint_root_realization/`; pair those copies with
the recovered `pins.json`, `source_points.txt`, and `source_function.txt` in a fresh
working directory. The publication replay did this from outside the checkout.

No source/tests changed between this source checkpoint and its documentation
publication. `make smoke` and `make eval-smoke` remain unavailable until their
training/evaluation entry points exist; no loss or evaluation table is claimed.
