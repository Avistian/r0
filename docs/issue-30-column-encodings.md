# Fixed typed observation encodings

`ColumnEncoding` and `encode_observation_columns` convert finite float64 numeric
columns to binary, categorical, ordinal or count observations. This is an r0
extension component, not a claim of reproducing a complete paper recipe.

Categorical and ordinal maps use explicit strictly increasing cutpoints and
declared vocabulary order. Equality can belong to the lower or upper bin.
Unobserved categories retain their labels; singleton categorical and ordinal
vocabularies are valid. Binary maps require exactly two labels. Cutpoints are
supplied externally: this kernel does not fit them on context or query rows.

Count maps compute `round(max(0, x / scale))`, with floor, ceiling or nearest
(ties to even). Overflow fails instead of silently saturating. Each selected
column has one encoding. Existing categorical columns cannot be silently
reinterpreted as numeric measurements. Unselected values and schemas remain
unchanged and input arrays are not modified.

The result includes values, output schema, the encoding specification and a
SHA-256 binding to the input values, input schema and specification. Binding
does not make lossy maps invertible. Original values must be retained separately
when composing this with observation truth and archive replay.

Bounds are 20,000 rows, 256 columns and 1,023 cutpoints per column, subject to
the supplied metadata and conservative workspace limits. These limits are
admission estimates, not a measured operating-system memory cap.

Validation includes scalar boundary oracles at and immediately around every
cutpoint, rounding examples, vocabulary preservation, value/label binding and
invalid-schema/resource/overflow rejection. Raw logs are gitignored under
`docs/evidence/issue-30-encodings/`. The initial test run exposed an overly
narrow error-message expectation: metadata admission can reject a one-byte
budget before workspace admission. The corrected expectation preserves both
valid rejection paths.

**Remaining:** integration with observation requests, original-schema truth,
archive replay and CLI configurations; sampled/fitted cutpoints and broader
encoding families. This component alone does not complete O30-01 or O30-06.
