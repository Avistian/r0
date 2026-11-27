# Frozen network input views

This candidate adds a versioned input view around an unchanged standalone network
archive. It is the library foundation for required irrelevant-feature and
input-sparsity comparisons. Raw pair creation, comparison verification and original-row traces are implemented
alongside opt-in independent variance matching. Explorer/Colab controls remain
required before this feature is delivered.

NetworkInputView stores the complete original NetworkRecipe, explicit task
controls and sorted active input indices. Original network columns retain their
order. Additional observed columns use separately keyed standard Gaussian draws,
one stream per column; increasing their count preserves existing columns. The
original joint root keeps its original width and sampling schedule, including
coupled ball and covariance laws. The network archive is never widened or
resampled.

Inactive coordinates are replaced by zero after the original root transformation,
before evaluating the unchanged network. This is input masking, not connection
sparsity or a causal intervention on the original graph. The default output is
raw: masking does not preserve expected or empirical signal variance. An active
input is a potential dependency; frozen masks or constant functions can still
make it ineffective. No query-dependent calibration is introduced.

The task active count must equal the mask cardinality, signal width must equal
the original network input width, and feature count cannot remove original
columns. Generation, strict decoding and evaluate_signal share the same mask;
signal evaluation consumes supplied observed coordinates without drawing roots.
Task identity and target-adapter random streams stay separately keyed. Admission
includes original generation and simultaneously retained masked-view buffers.

Initial public checks cover unchanged original draws, scores, targets and splits
for Gaussian/uniform/ball roots, common appended columns, exact archive decode
and generation, a literal masked affine oracle, and malformed mask/count rejection.
Those foundation checks are separate from the comparison and explorer evidence below.


## Raw comparisons and original-row traces

The shared paired_recipe/generate_pair/compare_pair APIs now accept feature
extension and input-count changes. Shrinking a mask takes its first sorted active
coordinates; expanding adds the lowest unused original coordinates. Masks are
nested. Extra columns never become network inputs. Removing original coordinates
or claiming a mask count different from the task controls rejects.

Feature-extension pairs require exact common columns, scores, targets,
probabilities and splits. Sparsity pairs require exact observed inputs and splits,
the same frozen network/root and nested masks; score changes are permitted and
zero-dispersion cases remain valid. This raw protocol reports
network_input_mask_uncontrolled; no expected-variance equality is claimed.
Explicit estimated matching is now a separate opt-in protocol; see
[network variance matching](issue-28-network-variance-matching.md). It fits both
masks on one independent retained population, rejecting exact constant channels
and nonfinite dispersion instead of assigning a fallback scale. Raw mode is unchanged.

Pair verification decodes both recipes and regenerates both sides, checking
arrays, identities, mechanism records and declared implementation versions.
The original before archive is retained exactly. Frozen-network identity is
checked independently of view metadata. Original-row traces show observed
coordinates, the explicit zero mask, every network stage and target adaptation,
with repeated row ordering preserved. These are input comparisons, not causal
interventions on a sampled graph.


## Local explorer workflows for raw input views

The local workbench exposes total-feature and active-input comparisons alongside
noise or class separation. Raw sparsity explicitly reports uncontrolled variance.
Extra columns retain common draws, and the original architecture stays fixed.

Replayed input views keep frozen identity controls, construction provenance,
original-row traces, guarded response curves and linked authored architecture
edits. Editing a block preserves the observed width and explicit input mask;
sampled construction evidence remains in the original archive. Standalone input
views show the same causal-intervention applicability notice as ordinary networks.

Browser verification exercises both raw comparison controls, saved pair reopening,
frozen Next, view replay, validated block editing and export interaction. Public
session checks additionally verify a nonprefix mask survives an edit and its
scores equal the independent affine oracle. The estimated matching view has a
separate lifecycle and requires its own integration; this slice does not qualify
hosted Colab or GPU execution.
