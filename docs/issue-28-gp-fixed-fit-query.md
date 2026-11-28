# Fixed-fit GP-root query continuation

`GPRootCalibrationQuery` retains its complete calibrated source (or prior query),
a newly keyed GP-root DAG, and the original limits. Its `original` property is
the current query; `fit` and `calibration` resolve the unchanged independent fit.
The authored graph, GP bindings, target adapter, kernel laws and row geometry
remain fixed across continuation. Fixed-location requests may explicitly supply
new locations; uniform and grid requests retain their named location laws.
This is a new finite joint GP draw, not conditioning on prior rows.

Public construction is `realize_gp_root_calibration_query(source, seed=...,
task_id=..., locations=None, limits=None)`. The raw-or-typed
`preflight_gp_root_calibration_query` validates the entire retained source and
prospective wrapper before source replay or new draws. Every original, fit and
ancestral identity remains reserved. The shared metadata depth and aggregate
parameter/working-memory laws apply to the complete future lineage; there is no
separate generation-count cutoff and no history truncation. Numerical failures
remain fail-closed without retry or implicit jitter.

The strict codec, public generation and run-request dispatch, canonical task
validation, saved-task ZIP path, repeated-row query trace and independent fit
trace accept the new version. Execution metadata and transport guards run before
replay. Fixed/grid retained coordinates are bound to their declared request
before covariance replay. Shared matrices retain their complete population axes;
fit statistics retain explicit center/scale axes.

## Bounded author verification

Public RED/GREEN cases cover all four graph families, fixed-fit arithmetic,
large identities, continuation, JSON/ZIP replay, repeated original rows, fit
inspection, singular fail-closed draws, malformed raw receipts, ancestor reuse,
execution metadata, run requests and producer custody. The inherited malformed
transport tests are included in the focused run. No full suite was run.

Outside-test evidence in the ignored
`docs/evidence/issue-28-gp-fixed-fit-query/` directory contains:

- 24 scalar covariance/Cholesky, graph and target calculations across four
  regression graph families and functional binary/multiclass adapters, uniform
  and fixed locations, and shared/per-column kernels. Maximum error was
  6.67e-16; instrumentation observed no calibration-constructor calls.
- Sequential actual continuation: 24 admitted draws followed by a depth-bound
  rejection with zero RNG/Cholesky calls; separate 256 MiB and 50,000-parameter
  fixtures admit two draws then reject the next with zero work. These counts
  describe fixtures, not hard-coded lifecycle limits.
- Complete original/calibrated task artifact comparisons against the inherited
  fit-inspection source, including records, mechanism, tensors and split indices.

This is an author-tested core increment awaiting independent review. Query Next
controls and broader GP intervention work remain required follow-ons. It does
not establish whole issue #28 acceptance or independently accept inherited fit
calibration/inspection changes.
