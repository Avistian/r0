# Issue 28: independent target calibration (candidate)

This extends [authored functional graphs](issue-28-functional-graph-integration.md)
with `CalibratedFunctionalCausalRecipe` (`functional-causal-v2`). It implements the
independent target-normalization part of accepted specification decisions 12–13;
full sampled-graph construction and ancestral function/pipeline fitting remain open.
Version one retains its original unnormalized schema and stream namespace.

## Contract

The version-two recipe selects 32–65,536 calibration rows (default 256). Generation
uses a separate `target-calibration-v1` keyed population from the same frozen graph.
The target readouts include the existing converter-capture and structural-noise
ordering. Features, context/query rows, labels and emitted row count do not determine
these statistics. This independent fit is an r0 adaptation, not a claim of exact
first-batch fitting behavior from a paper implementation.

Each target channel receives its independent population's empirical mean and
population standard deviation (`ddof=0`). The graph's conservative moment analysis
must certify finite second moments before normalization. Finite sampled values do
not bypass that mathematical gate. Scaling before moment estimation avoids squaring
extreme raw values. Arithmetic failures raise; there is no clipping or replacement
draw. Constant calibration columns retain identity scaling (center zero, scale one)
and their indices are recorded; constancy in a calibration sample is not a proof
that the population is constant.

The mechanism card retains row count, center, scale, constant-channel indices,
stream/version, a canonical little-endian float64 calibration-population digest,
and a fingerprint of the frozen nodes and target readouts. It does not store the
full calibration population. Seed/task identity plus the frozen recipe and named
stream make that population reproducible. This digest is provenance, not a signature
or proof against coordinated alteration of saved artifacts.

Inspection applies retained constants without refitting and checks the graph
fingerprint, row count, record shape and reconstructed scores. Repeating generation
with the same recipe/seed/task identity deterministically recomputes the independent
fit. Changing the task identity may change that fit; changing emitted row count does
not. Version two does not yet expose reusable sampled-graph construction receipts.

Calibration numeric metadata and its maximum row count are charged to graph
admission. These are conservative declared bounds, not measured process RSS or an
OS memory limit. Calibration runs on CPU; no GPU evidence is claimed.

## Candidate checks and open verification

Public tests cover fixed constants across emitted row counts, selected-row inspection,
finite-second-moment rejection, constant-target identity, malformed records, resource
admission, and cancellation of linear scales at `1e300` and `1e-300`. The shared
session test covers create/save/reopen/replay for both graph versions, all four
current root bases, and regression/binary/multiclass adapters. These are session/API
checks, not browser or hosted Colab checks.

Development checks exposed two decoder regressions: non-object graph records leaked
`AttributeError` after version dispatch, and missing calibration cards leaked
`KeyError` in inspection. Both now reject with `ValueError`; original failures remain
in `/tmp/r0-28-graph-calibration-*-red.log` pending evidence publication.

The full check and independent review are pending. No overall issue-28 completion,
browser/Colab readiness, throughput, scale, scientific or GPU acceptance follows
from these candidate checks. Sampled-graph operation ownership, complete raw-draw
retention, fresh upstream populations for every receiving fit and final atomic
commit remain separate work, guided by
[the ownership research](research/0028-sampled-graph-ownership.md).
