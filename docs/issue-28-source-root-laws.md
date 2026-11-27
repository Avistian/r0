# Source-specific root distributions

This work covers the remaining P1.root.01–03 and P1.code.02–06 inventory options.
Lossless integer Zipf records and a strict coordinate-law emitter are implemented.
Root profile selection, frozen normalization, graph/network integration and
explorer delivery are still pending; these prerequisites do not close those IDs.

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
