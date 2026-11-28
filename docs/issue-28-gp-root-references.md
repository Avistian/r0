# Retained GP-root causal reference curves

`evaluate_gp_root_reference` evaluates explicit simultaneous observed-value cuts
at one original population row. It accepts plain `GPRootDAGRecipe`, independent
`GPRootCalibration`, and fixed-fit `GPRootCalibrationQuery` archives. This is an
r0 diagnostic extension, not a conditional expectation inferred from a GP.

The input grid contains every observed feature in declared order. Unobserved
GP-root coordinates retain the selected original row's values. Each grid point
executes the complete original row geometry, preserving any row-dependent
mechanism behavior. `structural_innovations="original-row"` retains the original
keyed structural innovations; `"zero"` explicitly zeros structural equation
noise. Domain emission uniforms, task observation noise, and task label uniforms
retain their original row draws under both policies. No GP population or target
fit is constructed for grid points. Existing construction replay still verifies
retained source custody.

The immutable `gp-root-reference-v1` receipt includes complete canonical source
JSON (including query ancestry and independent fit), source and population
identities, observed schema, raw structural scores, fixed-calibrated scores,
probabilities where applicable, and actual fixed-draw targets. `target_mean` is
only the continuous adapter mean at the specified structural state. It does not
marginalize hidden causes. Categorical targets retain integer labels and their
vocabulary; they have no numeric target mean. Categorical observed features
require exact integer codes, never interpolation.

Functional, native scalar, heterogeneous native vector, and converted graph
runtimes are supported. Converted category cuts distinguish `capture` (change
only the readout) and `representation_and_capture` (also replace the encoded
state consumed downstream), using existing causal intervention semantics.
Unsupported cuts fail closed through the shared intervention validator.

`gp_root_reference_from_source` admits raw source metadata and prospective
receipt/full-row workspace before source replay. Explicit caller limits combine
with retained limits. The grid is bounded to 129 points and 20,000 total row-point
executions. `gp_root_reference_from_dict` checks complete receipt schema,
identity, primitive types, shape and vocabulary before replay, then requires
exact reconstruction. Ordinary generation and existing archive schemas remain
unchanged.

The library core and guarded workbench are separate implementation slices.
Composed intervention-source curves remain a required integration follow-on.
The source supports accepted GP root laws and coordinates through its retained
archive; this diagnostic adds no new source sampling law or paper attribution.

## Guarded workbench integration

Plain, independently calibrated and fixed-fit query GP-root datasets expose an
original-row reference panel. The ordinary feature selector chooses the plotted
coordinate; blank grid controls sweep continuous observed ranges or enumerate
declared category codes. Optional full-grid JSON supports simultaneous cuts and
repeated points. Separate controls choose structural innovation policy and the
converted capture/representation policy. Display options distinguish raw scores,
fixed-calibrated scores, class probabilities, continuous adapter means, and actual
fixed-draw targets. Category points are plotted without interpolation.

Every evaluation runs through the timeout-protected generation worker. Grid
metadata is bounded before expansion, then the core admits complete source,
history and prospective execution before replay. Failures preserve the active
saved dataset. The JSON download is the exact core receipt, preserving integer
identities even above JavaScript's safe-integer range. Replay/open and fixed-fit
query controls retain their existing archive behavior.

Reference work can exceed the original construction's retained budget, especially
when query ancestry and complete fit provenance are present. Such requests reject;
raising only exploration limits cannot override a smaller archived limit. Choose
explicit adequate limits when constructing the original source. No cap is silently
raised and no provenance is discarded. Hosted Colab and GPU verification remain
separate; the local callback uses the same guarded implementation.

Composed GP intervention-source references remain a required follow-on, alongside
independent review of this UI and its separately frozen reference core.
