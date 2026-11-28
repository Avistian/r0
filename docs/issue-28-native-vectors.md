# Heterogeneous native vector variables

`NativeVectorNode` owns one `FunctionalCausalNode` equation and an ordered tuple
of scalar variable domains. `NativeVectorRecipe` uses `native-vector-causal-v1`.
This is a separate versioned family: scalar native v1/v2 records and RNG keys keep
their original meaning and bytes.

A node evaluates its joint function once on the concatenated encoded states of
its parents. Its joint innovation can use the existing correlated exogenous
factor; any authored score pipeline operates once before emissions. Numerical
readout converters remain incompatible with native variable emissions.

| Logical variable | Joint score slice | State consumed by children | Observed value |
| --- | --- | --- | --- |
| Continuous | 1 | 1 continuous channel | floating point |
| Binary/categorical threshold | 1 | K one-hot channels | integer category index |
| Binary/categorical softmax/argmax | K | K one-hot channels | integer category index |

The ordered `node.spans` property gives each logical variable's score and encoded
state boundaries. Score width, logical-variable count and encoded-state width are
validated separately before evaluation. This is a joint vector mechanism, not
independent scalar nodes presented as a vector. Category blocks may have unequal
cardinalities. Their encoded widths retain all declared categories even when a
finite sample does not emit every category.

Observed references and the target are `(node, logical_variable_index)`. A target
may share its node with observed siblings; the exact target reference cannot be
observed. Hidden siblings still participate in the state available to descendant
nodes. Feature schema, vocabulary, target type, probabilities and query-label
hiding follow each referenced variable. No post-hoc target adapter runs.

## Sampled construction and provenance

`NativeVectorGraphRequest` uses `native-vector-graph-request-v1` and accepts an
ordered tuple of domain priors per node. Each prior describes one scalar logical
variable, so `continuous_width` must be one; several continuous variables are
represented by several entries. Existing symmetric/asymmetric frequency laws,
cutpoint choices and categorical emissions are available per variable. Optional
`composition` enables the existing source-qualified parent composition catalog;
parent slices use complete encoded node widths, not logical variable counts.

`sample_native_vector_graph` shares one named preference scope with per-variable
`node:i/domain:j` ownership and one root/function subtree per node. Compatible
stochastic catalogs share existing preference names. Raw frequency and emission
draws remain separate per variable. Fitted receiving functions see independently
sampled ancestor populations containing actual typed emissions and their one-hot
blocks, never ordinal category codes. No query rows fit mechanisms or emissions.

`sampled-native-vector-graph-v1` retains every variable plan, mechanism fit, raw
record, parent calibration receipt and all four initial/final stream snapshots.
Full construction replay checks all retained fields. Prefix allocation, encoded
width, raw-record and metadata budgets are admitted before materialization, and
failed construction publishes none of the four streams. Underflow, nonfinite
values and budget failures are explicit, without category deletion or retries.

## Exploration and source scope

The explorer includes heterogeneous vector and vector-parent presets with editable
JSON arrays. Generation, per-variable score/emission/state inspection, typed CSV
export, exact frozen reopen, resplitting and linked authored edits use the public
interfaces. Editing the domains array creates a new authored mechanism; the
original sampled archive keeps its construction receipts.

[TabAbax §3.3](https://arxiv.org/html/2611.00562v1#S3.SS3) supports vector latent
nodes, partially observed dimensions, discretization/softmax categoricals and
cardinality/one-hot consistency. Its wording does not disclose the complete
heterogeneous internal layout or resolve its converter-versus-function ambiguity.
The explicit variable layout here is an **r0 extension**, preserving those named
requirements without claiming exact reproduction. Frequency offsets do not
promise empirical class marginals. Undisclosed source laws, source-specific
mixture probabilities and the broader #28 options remain required/unknown as
inventoried; they are not silently replaced by these presets.

[Vector causal interventions](issue-28-vector-interventions.md) now retain joint
mechanisms with explicit logical-variable clamps and domain changes. Adapter-pair
APIs still reject native vector recipes; ordinary mechanism edits are distinct
from causal interventions. Hosted execution and scientific reproduction are unverified.
