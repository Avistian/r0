# Calibration and shared-scope checkpoints

Historical checkpoints from the [implementation ledger](issue-28-implementation.md).
Their pending statements describe those revisions; see [current integration status](issue-28-integration-status.md) for subsequent progress.

## Standalone Standardize checkpoint

Source `962f0ad` adds immutable population Standardize with epsilon 1e-4 outside
the square root. It supports node input/output placement and ordered pipeline
stages, retains calibration identity/digest, preserves one-row and constant-channel
behavior, and checks finite input second moments before construction. Generation,
original-row inspection and saved replay use the same frozen constants.
[Contract and evidence](issue-28-standardization.md).

Full checks pass **809 tests, one skipped**, all static gates. Independent scoped
review passes 174 checks: 38 numerical comparisons, 58 expected rejections, exact
fitter/equation resource limits, moment/ancestry cases and three saved task types.
All 168 legacy proposals are byte-identical, including four retained failures.
Five exact programs and imported-source custody preserve reproduction. Standards
finds zero hard violations and one nonblocking duplicated-admission-list advisory.
Earlier verifier mistakes remain classified and retained separately.

This advances node standardization; sampled source pipeline construction remains
required. The [integration design](research/0028-pipeline-sampling-integration.md)
recommends one shared preference context, complete transactional binding ownership,
explicit global-mode adaptation, source-name adapters and ancestral fitting against
final parent states. It preserves legacy families. Proposed design oracles are
NOT_RUN. Full #28 scope, browser/Colab, GPU, scale and scientific evidence remain open.

## Shared sampling context checkpoint

Source `662fb89` adds one transactional numerical/categorical scope across named
operations, nested ownership, explicit global policies, forced identities and
aggregate admission before draws/decoding. Existing samplers retain their default
behavior. [Contract and retained evidence](issue-28-joint-sampling.md).

Full checks pass **825 tests, one skipped**, all static gates. Independent review
passes 286 assertions including 165 expected rejections; all 156 legacy pairs are
byte-identical, including three retained failures. Source/import custody verifies
323 entries each. The original `2812d23` decoder REJECT remains retained: a string
path was silently coerced into character components. Repaired decoding rejects
string/dictionary paths. The separate Tensor-output verifier mistake is retained.
Standards accepts the narrow repair/documentation remedy, with no source-level
hard finding and one advisory transaction-copy duplication concern.

This is the shared context/ledger foundation, not the sampled graph integration.
Typed activation/matrix/root/converter plans, semantic destination bindings,
independent ancestral fitting against finalized parent states, source profiles,
all wider options and actual browser/Colab/GPU/scale evidence remain required.
The full #28 goal stays active.
