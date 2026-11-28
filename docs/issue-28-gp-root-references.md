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

This slice is the library core. Guarded explorer controls, reference downloads
and composed intervention-source curves remain required integration follow-ons.
The source supports accepted GP root laws and coordinates through its retained
archive; this diagnostic adds no new source sampling law or paper attribution.
