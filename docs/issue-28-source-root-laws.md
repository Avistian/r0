# Source-specific root distributions

This work covers the remaining P1.root.01–03 and P1.code.02–06 inventory options.
The first implementation step adds lossless integer Zipf raw records. Root profile
construction, frozen normalization, graph/network integration and explorer
delivery are still pending; this prerequisite does not close those inventory IDs.

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
