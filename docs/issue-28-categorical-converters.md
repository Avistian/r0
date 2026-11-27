# Typed categorical converter pipelines

`ConvertedCausalRecipe` declares numerical readout channels and named categorical
captures through `ConvertedReadout`. A `ConvertedCausalNode` retains one joint
`FunctionalCausalNode` equation and disjoint `CategoricalConverter` spans. The
category occupies one typed output column; its descendant representation preserves
the entire converter input width. Numerical readouts cannot select reserved category
channels, and the target capture cannot also be an observed feature.

The stage order is joint function, frozen standardization/channel importance/L2,
disjoint numerical and categorical converters, category capture, structural
innovation, then node importance. Numerical converter variants retain their existing
readout rules. Descendants consume the final state. Neither a category code nor its
probability vector replaces that state implicitly.

| Mode | Category extraction | Descendant representation before innovation |
| --- | --- | --- |
| `neighbor_id` | Nearest retained prototype under declared p-distance | Original input slice |
| `neighbor_disc` | Nearest retained prototype | Selected prototype |
| `neighbor_func` | Nearest retained prototype | Frozen cheap function of selected prototype |
| `neighbor_int` | Nearest retained prototype | Category index broadcast over input width |
| `softmax_id` | Frozen standardized logits, bias and explicit uniform | Original input slice |
| `softmax_disc` | Frozen standardized logits, bias and explicit uniform | Selected embedding |
| `softmax_int` | Frozen standardized logits, bias and explicit uniform | Category index broadcast over input width |

Nearest ties use the first prototype. Softmax bias weights change logits; they do
not guarantee marginal category frequencies. Nonfinite distances, logits or states
reject without replacing values or retrying a different family. The original-input
modes preserve their input moment limitations even though category captures are
bounded. Prototype, embedding and index states have finite support before subsequent
structural innovation. All normalization still requires certified finite moments.

## Sampled construction

`ConvertedGraphRequest` supplies one vocabulary per feature and a target vocabulary.
An empty vocabulary means continuous. Nonempty vocabularies explicitly declare the
labels; target task type/classes follow its vocabulary. `CategoricalConverterPrior`
selects any nonempty subset of the seven modes. The request also retains topology,
latent-width, numerical pipeline/innovation, root, function, and optional parent
composition priors. The existing cheap source function catalog is used for
`neighbor_func`; embedding points use the existing transformed RandomPoints root
construction, including its full function subtree.

`sample_converted_graph` requires distinct named, raw, calibration and fit NumPy
generators. Mode and input-width choices precede observation allocation and latent
width planning. Every function fit and node-pipeline population sees ancestors with
their final representations and innovations. Converter fits use the independent
node-pipeline population after frozen preprocessing. Prototype permutations retain
their complete row order. A transformed-center fit retains selected centers with
row multiplicities. Softmax embedding points have their own independent construction.
Emitted rows never fit or update these records.

`SampledConvertedGraph` retains plans, operation registry, calibration receipts,
raw draws and all four initial/final snapshots. Public decoding and generation check
complete construction replay. `with_task` permits episode resplits while retaining
the source identity. Direct generation, `explain_generation`, typed dataset save/load
and schema-preserving exports share the versioned recipe. Adapter-control task pairs
reject these direct targets.

Admission charges prototype/embedding storage, distance work proportional to
rows × categories × input width, probabilities, all nested function constructions,
retained fit populations and aggregate raw recording before materialization. Failed
sampling leaves all four caller RNGs unchanged. These are conservative allocation
bounds, not measured process-memory guarantees.

## Source qualification and remaining integration

The seven representation modes, input-width choices, p-distance bounds, softmax factor,
positive bias weights and transformed RandomPoints embeddings follow the pinned
[TabICL converter implementation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py).
Independent calibration, immutable archives and explicit nonfinite rejection are r0
adaptations. Explicit vocabularies are caller specifications; no paper cardinality
law is claimed. Optional non-Gaussian innovations remain separately labeled r0
extensions in their pipeline prior.

Checkpoint B provides the typed library construction/replay boundary. New-family
explorer/Colab controls and causal intervention support are required checkpoint C
work, currently pending. Captured-category interventions, pre-innovation
representation interventions and final-state clamps require distinct semantics.
An original-input converter has no unique latent state implied by a forced category.
Authored mechanism edits must not be presented as causal interventions. Hosted
execution is NOT_RUN.
