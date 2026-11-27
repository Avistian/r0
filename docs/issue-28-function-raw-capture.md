# Issue 28: function and root raw capture

This extends the matrix raw ledger through all eight source function families and
all four root base families. It is part of the complete sampled graph work in
[the accepted specification](specs/0028-causal-network-generators.md), not completion
of #28. The earlier matrix uniform-bound REJECT remains retained separately.

`materialize_joint_function` and `materialize_joint_root` accept optional
`capture=RawDrawLedger()`. Returned realization records and their numerical
recipes are unchanged. Capture records original values at their owning draw
sites, before transformations. Record order is global across the named streams;
`raw`, `fit` and `calibration` identify their respective actual generators.

Inventory:

- Linear/quadratic: nested matrices and the complete quadratic permutation above
  input width 20.
- Fourier: frequency uniforms, radial directions/factor/weights when selected,
  phase uniforms and Gaussian output coefficients.
- Tree: raw feature choices with probabilities, fit threshold-row indices and
  Gaussian leaf values.
- Discretization: full fit-row permutation and nested output matrix.
- EM: fit anchor rows, Gaussian jitter/spreads and nested output matrix.
- Product: both cheap children in order, including their separate fit choices.
- MLP: every ordered matrix and each activation's fit shift-row choices. Captured
  reconstruction uses the same numerical fitting helper after whole-MLP admission.
- Roots: covariance factor/weights, base values and ball radii, then the selected
  function's complete construction and fitting draws.

`RawDrawLedger.transactions` stages 1–64 distinct named NumPy generators and the
ledger together. Nested single-stream transactions inherit their existing name.
Nested multi-stream transactions must retain the same name-to-generator bindings.
The caller and yielded stream mappings cannot add, remove or replace bindings.
The five supported NumPy engines remain PCG64, PCG64DXSM, Philox, SFC64 and MT19937.

On a late construction/validation/record-budget failure, all construction, fit and
calibration draws and new records are discarded; earlier committed records remain.
Unrecorded RNG advances and false stream labels are rejected. External caller RNG
mutation is preserved and causes rejection; transactions require exclusive caller
ownership, not concurrent mutation support. State-byte admission is aggregate over
participating generators. Captured records remain replayable from independently
supplied trusted generators; no unpickling or public raw JSON decoder is added.

Author evidence includes genuine failing/passing checks for missing multi-stream
support and mapping mutation, plus all 32 root/function combinations across five
engines. Each combination checks captured/uncaptured realization equality,
record-by-record replay and next RNG values, then deliberately fails the last draw
budget and checks all streams and an earlier ledger record remain unchanged.
Existing literal-equation and realization tests also run. Full `make check` and
independent review are pending at source commit time; later evidence must state
its exact source identity. Smoke/eval-smoke remain unavailable without train/eval.

Still unfinished: prefix draws, complete sampled graph builder and typed
plan→realization→runtime ownership, independent ancestor fit populations, aggregate
whole-graph admission, raw decoding, broader paper options and end-to-end
local/Colab/browser/GPU/resource/scientific verification. This component does not
establish those requirements.
