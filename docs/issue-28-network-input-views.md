# Frozen network input views

This candidate adds a versioned input view around an unchanged standalone network
archive. It is the library foundation for required irrelevant-feature and
input-sparsity comparisons. Pair creation, comparison verification, original-row
trace integration, explorer/Colab controls and optional independent variance
matching remain required before this feature is delivered.

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
They do not qualify the still-pending comparison or explorer workflows.
