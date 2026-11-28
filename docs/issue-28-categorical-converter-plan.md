# Categorical converter pipeline seam

Planning checkout: `/tmp/r0-28-categorical-converter-pipelines`, based on frozen
A `f4bd521`. The plan was approved before source edits. Implementation uses the
owned `/tmp/r0-28-typed-converter-pipelines` checkout from integrated `5c8f6bc`. B/C remain required scope.

## Records and typed output

Introduce a separate `ConvertedCausalRecipe` family rather than expanding the
numeric target-adapter meaning of `FunctionalCausalRecipe`. Each
`ConvertedCausalNode` owns an ordinary `FunctionalCausalNode` equation and ordered
categorical converter records. The existing equation pipeline retains its numeric
stages/converters; categorical records use disjoint `[start,stop)` spans and explicit
vocabularies. Scores, descendant state, numerical readouts and categorical captures
are separate evaluation outputs. Discrete captures remain int64 logical variables;
continuous readouts remain float64. Readout references explicitly select a numerical
channel or a named categorical capture, rather than overloading a channel index.

Reject overlap between categorical spans and numerical converters or other
categorical spans. Numerical observation references cannot select inside a reserved
categorical span; its extracted category has its own reference. Hidden latent
channels remain available to descendant equations. Reject selecting the same target
capture as an observed feature. Native categorical targets derive type/classes from
that capture, prohibit adapter-only controls and return truthful probabilities;
continuous targets use a direct numerical readout. Existing ordinary numerical
adapter recipes remain unchanged and separately available.

A width-preserving state contract is important: a converter consumes d channels and
returns d channels for descendants even though it extracts one categorical column.
All modes preserve the surrounding joint node width. This differs from native
one-hot vector emission and must not reuse its state-width calculation.

## Evaluation and shared inspection

The new evaluator applies a joint equation once, existing numerical standardization,
channel importance and L2 once, then numerical and categorical converters on disjoint
spans. Preserve each numerical converter variant. After all captures, add the full
frozen innovation and apply node importance. Signal-dependent response stays defined
on the pre-pipeline function output as in A. Descendants consume final transformed
states, not captured categories. Softmax emission uniforms have independent keyed
converter streams and original row identity; independent fit populations use the
same emission code with their own captured calibration stream.

Keep numeric and typed family records separate. Factor a shared frozen-function
inspection dispatcher where useful: parent, network and Kumo interior helpers return
actual final dependency IDs; caller owns root/input stages and context-specific ID
prefixes. Preserve existing graph IDs and artifact versions. Root's network fix
`26352a2` currently calls `network_steps(function, inputs, prefix, indices, add,
input_parents)`; implementation will coordinate the accepted integrated base so
new typed traces do not lose network or product internals. This is inspection reuse,
not a new target adapter or a second implementation of frozen function mathematics.

## Source profile and sampling lifecycle

Map to `I2.category.01`–`.20` and `SPEC.D01`–`D03`. Required mode catalogs are single,
neighbor-only, softmax-only and all seven named variants. The pinned source selects
categorical mode and input width before graph width/mechanism construction; retain
that order. Neighbor distance exponent and softmax factor are named logarithmic
requests. A two-way named categorical draw controls all-versus-subset input channels;
subset width uses the source local logarithmic floor/upper-exclusive convention.
Any r0 cardinality prior is separately labeled: the converter itself receives a
fixed vocabulary size from its feature specification.

New request/plans attach one typed feature specification per observed feature and
one target. Plan converter mode/input width first, then use the existing ordered
observation allocator with heterogeneous converter widths, then sample latent widths
and joint mechanisms. Retain full plans/ownership in the shared named context. Old
numerical geometry envelopes keep their dimensions and bytes.

Nearest-prototype fitting retains a permutation of independently generated input
rows, selected center row IDs, centers and exponent. Insufficient calibration rows
for declared category count reject without category deletion. The transformed-center
mode fits a frozen cheap function on the independent population of selected centers,
including multiplicities, not on emitted/query rows. Use the existing cheap catalog
and its distinct named request, not an arbitrary linear alias.

Softmax fitting retains input standardization, positive weight realization and
log(weight+epsilon) biases, scale factor and sampled embedding matrix. Reuse the
source RandomPoints construction through existing root/function plans: transformed
hypercube, volume-ball, Gaussian or covariance bases; do not substitute Gaussian
mixture points for the actual RandomPoints selector. The finite embedding population
has category-count rows; all fit/raw provenance is retained. Unsupported combinations
fail explicitly without retry or a replacement family. Probabilities use frozen
statistics; source nonfinite replacement is intentionally rejected under accepted
r0 failure semantics. Emitted uniforms remain separately recorded.

Source behavior checked against pinned commit9530bd65a7b806f6ad4f9f4335af745577e9d3cc:
[_converter.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py),
[_node_function.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py),
[_base.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py),
[_function.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py),
[_points.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py),
[_weights.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py).
The r0 independent-fit lifecycle is a named adaptation, not exact original batching.

## Capture and intervention semantics

Retain and expose five downstream representation meanings: original latent slice,
selected center, transformed center, selected embedding and broadcast category index.
A capture-only category intervention changes its extracted variable/probability
indicator without pretending to alter the descendant state. A representation-plus-
capture intervention must name the converter and pre-innovation stage explicitly;
subsequent innovation and node importance still run. In unchanged-latent `*_id`
modes there is no unique inverse latent value for a forced category, so reject an
implicit category-to-latent-state mapping. Users can instead choose capture-only
or an explicit numerical state intervention. No hidden one-hot substitution.

Final-state numeric clamps keep earlier captures unless both locations are explicitly
selected. Soft converter replacements preserve input-span geometry and vocabulary
identity, retain original provenance, and require frozen replacement parameters;
queries cannot refit. Matched comparisons retain unaffected root/noise/emission
streams and exact row identities, while population comparisons draw fresh streams.
Trace evaluated/bypassed components: capture-only can still require the original
emission for an unchanged descendant representation; a fully overwritten component
should be bypassed wherever possible. Partial coupled equations retain their explicit
evaluated semantics. New typed-family intervention support is required C work and
must remain visibly unsupported until implemented and verified, not implied by edits.

## Admission, workflow and checkpoints

Admit metadata, N×K×d distance work, N×K probabilities, K×d centers/embeddings,
function and point subplans, all retained fit populations and shared raw costs before
materialization. Certify finite moments where fixed normalization needs them. Converter
captures and states have separate dependence/moment propagation: a bounded category
capture does not make an unchanged heavy-tailed latent state bounded. Four-stream
failures are atomic; every archive binds plans, raw fits and emitted/derived values.

B checkpoint is coherent authored+sampled typed creation, decoder/replay/inspection,
typed save/export and all seven representations with fixed independent calibration.
C completes usable local/Colab family controls, edit/reopen/download and the explicit
intervention semantics above. Shared simple construction surfaces may land with B,
but no missing C integration is called complete. Tests use public APIs, literal NumPy
nearest-center/softmax equations, unequal spans and category counts, ancestor final-
state counterexamples, raw replay/tamper/budget rollback, old-archive identities,
actual Chromium and source-qualified controls. Hosted remains NOT_RUN unless run.
