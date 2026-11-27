# Source-specific root distributions

This work covers the remaining P1.root.01–03 and P1.code.02–06 inventory options.
Lossless integer Zipf records, a strict coordinate-law emitter and frozen
source-qualified profile selection are implemented. Frozen normalization,
graph/network integration and explorer delivery remain separate work; these
prerequisites do not close those IDs.

## Primary-source distinctions

[P1 C.2.3](https://arxiv.org/html/2207.01848v6#A3.SS2.SSS3) names Gaussian, Zipfian
and “Multivariate Distribution” inputs without specifying the latter as a
categorical law. That paper option must not be renamed to match a code branch.

The [pinned implementation](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py#L18-L21)
draws Gaussian means and scales of the form abs(Z*mean).
Its [root branches](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py#L88-L115)
include standard/presampled Gaussians, U[0,1], categorical sampling from random
positive weights, and Zipf exponent 2+2U clipped at ten. Categorical draws are
batch-standardized; Zipf draws are batch-centered. Mixed selection uses two
thresholds sampled below0.66; a third sampled Zipf threshold is unused. Preserve
this distinction in source-qualified records rather than inventing a new
three-weight selector and calling it identical.

## Implementation contract

The source profiles must retain per-coordinate law choices, parameter draws,
calibration inputs and transformations. Use explicit lifecycle names for frozen
independent calibration versus a batch-reference realization; neither may be
silently substituted for the other. Independent calibration must not consume
query rows. Degenerate calibration rejects atomically without retries.

Expose forced families alongside sampled per-coordinate mixtures. Preserve
source clipping as a declared law, with its unclipped integer samples retained.
A paper-level unbounded Zipf option is distinct from the clipped source branch.
Existing heavy-tailed, bounded and Gaussian-mixture extensions remain separately
named. Covariance/prototype reconstructions need their own disclosed contracts;
neither a correlated Gaussian nor a categorical branch proves the paper's
underspecified multivariate distribution.

Every new family needs public generation, replay, raw custody, original-row
inspection, intervention coupling where applicable, guarded local construction
and save/reopen coverage. Existing v1 root laws and their bytes stay unchanged.
Current network and functional-root type/admission seams require explicit
versioned integration, not a private sampling bypass.

## Raw Zipf prerequisite

RawDrawLedger accepts a finite exponent greater than one and retains positive
int64 values. No clipping, centering or floating conversion occurs in the raw
record. Root profiles must declare those later transformations independently.
Replay checks the supplied original generator state and exact integer draws.
Invalid parameters or malformed retained values do not publish RNG changes.


## Coordinate-law emission prerequisite

SourceRootCoordinate and SourceRootLaw retain an authored vector of Gaussian,
unit-uniform, symmetric-uniform, categorical or Zipf coordinates. Zipf clipping
is explicit; unclipped integer samples remain in the raw ledger. Numeric output
uses binary64, so integer-to-float conversion may round values above the exact
integer range. Gaussian zero scale is allowed as a deliberate constant draw;
a later standardization lifecycle must handle degeneracy explicitly.

draw_source_root accepts a caller-owned generator, optional raw ledger, ownership
path and limits. It admits arrays and retained raw scalars before allocation,
stages all draws, and publishes RNG/ledger changes only after finite output.
Existing records survive late failure. Strict coordinate decoding rejects unknown
fields and unused controls. Unbounded Zipf moment order is exponent minus one;
bounded and continuous laws here have all finite positive absolute moments.
This is law emission, not a claim of source profile selection or joint-root
graph integration.

Aggregate raw admission uses public frozen RawDrawSpec declarations and
RawDrawLedger.admit_draws before allocating the root output. This includes
already retained records, the actual bound stream name, ownership metadata,
all raw numbers, and the ledger's snapshot allowance. It shares the exact
per-record cost with capture. Admission is read-only, not a reservation: later
capture still checks its live remaining capacity. Invalid ownership rejects
before output allocation; finite-output failures still roll back atomically.
## Independent calibration lifecycle

SourceRootCalibration freezes per-coordinate identity, centering or sample-standard
scaling (ddof=1) from separately drawn calibration rows. Query application uses
only those stored locations/scales: appending query rows cannot change an existing
row's result. Constants, underflowed/nonfinite scales and nonfinite transformed
values reject without epsilon or retry, with caller RNG/raw ledger rollback.
The caller can retain calibration draws through RawDrawLedger, including original
integer Zipf draws before clipping. The strict serialized transform is an authored
record; it does not itself authenticate calibration provenance. A later sampling
envelope must bind the raw records and stream snapshots to this transform.

This explicit r0 independent-calibration lifecycle differs from pinned P1 code's
batch-reference normalization. Batch-reference generation, profile parameter
selection, a replay-validating calibration envelope, and joint-root/network/UI
integration remain required. Neither lifecycle may silently replace the other.

## Batch-reference normalization lifecycle

`draw_source_root_batch_reference` draws and normalizes the same batch. Identity,
centering and sample-standard scaling remain coordinate-specific. The distinct
`source-root-batch-reference-v1` record stores this batch's statistics and has no
apply-to-future-rows method. Adding rows can change every normalized row; this is
not interchangeable with independent frozen calibration. The two strict decoders
reject each other's version. Raw ledger records retain the underlying emission,
including unclipped integer Zipf draws, and degeneracy rejects atomically.

This implements normalization arithmetic and lifecycle, not full pinned-source
parameter-selection timing or an authenticated replay envelope. Those and the
versioned graph/network/explorer integration remain pending. Shared arithmetic
with independent calibration does not imply the same sampling lifecycle.

## Functional graph and standalone network integration

`SourceRootBase` combines a coordinate law with an optional authored independent
frozen calibration. Functional nodes explicitly use `functional-causal-node-v2`
for this descriptor; v1 nodes reject it. Existing v1 root records and draw order
remain unchanged. Standalone networks construct the corresponding versioned root
node and retain the descriptor in their own archive. Batch-reference statistics
cannot be passed as a frozen graph root calibration.

The shared evaluator, original-row inspection and support/query root streams use
the same root adapter. Source-coordinate certificates preserve boundedness and
Zipf moment order; finite calibration samples never upgrade a population moment
certificate. Independent variance matching for network input views uses the same
root law and frozen transform. Limits accompany graph and network evaluation.

This descriptor is authored root integration. The separate sampled wrapper below
carries source profile construction custody and replay. Guarded explorer controls
remain a separate integration slice.


## Frozen source-profile selection

`SourceRootProfile` offers the pinned P1 `normal`, `uniform` and conditional
`mixed` selector laws. `sample_source_root_profile` takes distinct `model_rng`
and `selection_rng` generators and returns `SampledSourceRootProfile`. The
model stream owns the full Gaussian parameter vectors; the selection stream
owns one selection call. The resulting `.law` can be emitted without any
normalization using `draw_source_root`. Both initial/final generator snapshots,
raw draws, Gaussian parameters (including unused ones), and three mixed
thresholds remain in the immutable record. Strict public decoders reconstruct
and replay the whole envelope. NumPy probability laws and retained sequence
are reproducible; Torch/Python/NumPy cross-library bitwise parity is not claimed.

For P1 mixed selection, stored thresholds are `(unused Zipf, categorical,
normal)` in source order. Conditional family probabilities are
`1-normal`, `normal*(1-categorical)`, and `normal*categorical`; the Zipf threshold
is deliberately not a third weight. The full presampled Gaussian vectors are
retained even for a unit-uniform selection, matching the source constructor's
otherwise unused draws. Category weights and Zipf exponents are frozen with
one selected call. This is explicitly `frozen-selection-r0-v1`, not an assertion
that the source keeps per-forward categorical/Zipf parameters fixed per model.
Repeated emission does not resample these parameters. The
`.source_normalization` metadata names source mixed-category sample-standardization
and Zipf centering; those transforms are **not applied** here. Independent frozen
calibration and generated-batch reference normalization remain distinct layers.

`forced-r0` accepts one family per coordinate: standard Gaussian, presampled
Gaussian, unit uniform, symmetric uniform, categorical, clipped sampled Zipf,
and fixed-exponent unbounded Zipf. This explicit heterogeneous composition is
an r0 extension, not the paper's underspecified multivariate distribution.
There are no retries or silent repairs for nonfinite or zero-total weights.

The [P2 initialization description](https://www.nature.com/articles/s41586-024-08328-6)
selects normal versus symmetric uniform per root node. `p2-node-mixture-r0`
selects once for the entire root vector, with an explicitly configurable
`p2_normal_probability`. It currently uses unit normal scale and uniform
half-width one. These fixed settings and the selection probability are a named
reconstruction; varying source scale hyperparameters and shared-prototype
coupling remain required follow-ons, not implicit features of this selector.

Whole-profile admission bounds all conditional branches and maximum category
count before any stochastic parameter draw. It includes both streams, raw
records and replay snapshots; conditional branches can use less than that
conservative budget. Public records preserve exact law and parameter custody
without modifying existing source-root or legacy root record versions. The sampled
wrapper below supplies graph/network integration; explorer workflows and explicit
source-root replacement interventions remain separate required work.


## Sampled graph-root provenance

`sample_source_root_base(profile, modes=..., calibration_rows=..., calibration_rng=...)`
wraps an accepted `SampledSourceRootProfile` in `SampledSourceRootBase`, version
`sampled-source-root-base-v1`. The source selection record is retained verbatim,
including unused model parameters, source labels, raw draws and both RNG streams.
The inherited `.law` must equal the retained profile's selected law. This is a
separate version from authored `SourceRootBase`; its bytes and version stay unchanged.

Modes are explicit per coordinate: `identity`, `center`, `standardize`. All-identity
uses zero calibration rows and no calibration RNG. Any nonidentity mode requires
an independently supplied calibration RNG and positive calibration rows; sample
standardization requires at least two. The complete initial/final calibration
snapshots, raw values and frozen statistics are retained. Center uses the finite
sample mean; standardize uses sample standard deviation with `ddof=1`. Exact
constants, underflowed scales and nonfinite outcomes reject atomically without
an epsilon, retry or caller RNG advance. Identity coordinates in a mixed fit
still have their independent sample draws retained, with location zero/scale one.

This lifecycle is an explicit **r0 independent frozen normalization adaptation**.
It is not the pinned source's normalization of the current generated batch.
`profile.source_normalization` continues to describe source behavior, separately
from the chosen `.modes`. Fitting never takes query rows, generated task values or
query labels. It does not certify unit population variance or upgrade an
unbounded Zipf moment certificate. `draw_source_root_batch_reference` remains
the separate source-batch reference API.

Construction and strict public `sampled_source_root_base_from_dict` reopening
admit profile metadata, both replay lifecycles, calibration geometry, raw records
and simultaneous snapshot/workspace reserves before drawing or restoring state.
The decoder replays the complete construction and compares every serialized field;
changing a law, fit statistic, raw value, snapshot or source label rejects.
Functional graph and network generation validate the retained profile and
calibration again before root emission, then apply the frozen transform to fresh
root draws. Graph admission also includes this replay work, including native and
converted graphs that use the common functional-node adapter. Functional root
nodes require v2, as for authored source roots. Original-row inspection and network
variance matching use the same adapter; persisted task recipes retain the full
sampled wrapper and can be reopened without discarding it to an authored law.

The wrapper does not add a source-profile explorer, prototype-mixture law,
source-scale hyperprior or source-root replacement intervention. Those remain
separate required slices; accepting this record alone is not whole-issue acceptance.

## Source-root explorer delivery

The shared explorer and local notebook callback expose 107 source-root construction
presets: all seven standalone architecture families crossed with the three pinned
P1 selectors (standard or presampled Gaussian parameters), the explicitly labeled
P2 node-mixture reconstruction, seven forced coordinate choices, and a heterogeneous
seven-coordinate root. Additional heterogeneous presets demonstrate independent
centering and standardization. Full JSON controls can compose the root wrapper with
an authored network or a source-profiled network hyperprior, set coordinate-wise
normalization modes, choose profile parameters and raw limits, and retain an exact
root construction seed separately from architecture and dataset identities.

The `source-root-network` wrapper retains the `SampledSourceRootBase` envelope.
The root seed owns separate model-parameter, coordinate-selection and independent
calibration streams. Identity uses no calibration rows or stream; other modes use
explicit independent rows. Frozen Next changes the dataset identity while retaining
all root selection, calibration and raw records. Construction details expose these
records and distinguish pinned code, paper omissions and r0 adaptations.

Network editor preparation and original-row trace replay decode inside the bounded
worker, with the saved archive and exact task identity. Architecture edits retain
the sampled root by default. The explicit authored-root policy copies its law and
optional independent calibration into `SourceRootBase`; it does not claim the edited
law was sampled. The original archive stays unchanged. Invalid text remains editable,
validated diagrams remain intact, and timeout/failure cannot publish a replacement.
Save, reopen, local export and response curves use the shared workflows. Curves
consume observed inputs (after any root normalization), so they do not apply the
root transform a second time.

This delivery does not implement source batch refitting as a frozen independent
root, resolve the P1 paper's underspecified multivariate law, or assert P2 prototype
coupling. Root plus internal neural-noise composition remains a required separate
integration. Hosted Colab execution is NOT_RUN; the local shared callback is covered.
