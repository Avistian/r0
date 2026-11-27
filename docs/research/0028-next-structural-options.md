# Issue 28: next structural options

Research freeze: `e0d19089ace471e24761455fb9aec6b0fa22ef30`, inspected 2026-11-27.
This is a bounded source/code investigation, not implementation or issue acceptance.
The accepted [specification](../specs/0028-causal-network-generators.md), all eight
[atomic inventory files](0028-source-option-inventory.md), and
[progress ledger](../issue-28-option-progress.md) were read. Historical ledger
claims were checked against this source freeze. No tests, training, browser,
Colab, GPU or throughput experiments were run. The typed-policy rejection at
this freeze is unaffected.

## Recommended dispatches

| Order | Exact inventory anchors | Missing delivery |
| --- | --- | --- |
| 1 | `OP.function.08`–`.11` | Finite joint GP feature-column populations with linear, quadratic and neural targets |
| 2 | `P2.graph.04`–`.05`, composing `.01`–`.03` | Explicit log-uniform size and Gamma-derived bounded redirection hyperprior envelopes |
| 3 | `R0.extension.18`, `.16`, `.21`; reuse `I2.root.08`–`.11` | Frozen Gaussian-mixture root parameters with independent row emission and graph/network/prototype integration |

The third item is a broad r0 lifecycle extension, **not four newly missing I2
primitives**. Existing scalar multimodal roots also remain valid implementations;
this dispatch adds the richer multivariate source-derived parameterization.
Each dispatch needs a core author, a whole-workbench integration author and an
independent verifier after a frozen implementation head exists. These are suggested
assignments, not authorization for new agents or paid execution.

## 1. Joint finite GP columns

**Primary source fact.** O'Prior §2.1, fourth SCM family, samples each feature
column with an RBF GP using Cholesky factorization, then constructs the target
with a linear, quadratic or MLP combination. It also permits ordered hybrid
mechanisms. The passage does not specify input locations, cross-column coupling,
kernel hyperpriors, numerical jitter or coefficient distributions. Those details
must be named r0 reconstructions, not historical source settings.
[O'Prior v1, 2026-05-18, §2.1](https://arxiv.org/html/2605.18971v1#S2.SS1).

**Actual gap.** `src/r0/modules/priors/lib/exact_gp.py:generate_exact_gp`
samples ordinary input features and draws GP **target** values, including
independent calibration locations; it uses a symmetric eigendecomposition root.
`gp_conditioning.py:condition_gp` extends those target draws. Neither implements
GP-generated feature columns followed by a distinct target mechanism. Frozen
function machinery likewise does not establish a finite joint column population.
The progress ledger correctly preserves this distinction.

**Proposed API and semantics.** Introduce an explicit request and realized
population record, separate from an evaluable frozen function. Retain row IDs,
latent coordinates, kernel parameters, covariance/factor, column innovations,
column values and the independently frozen target mechanism. A declared shared
coordinate grid with independent column innovations is a reasonable first named
reconstruction. Allow per-column kernels as a separate variant, rather than
silently changing the covariance contract. The three target choices must all be
available through the public construction path. Keep target adapter calibration
independent of query rows; no automatic population standardization.

A larger requested population requires a new construction. Merely slicing or
permuting a retained population preserves its original row IDs and values.
Conditional extension, if added, must be explicit and retain conditioning state;
rerunning Cholesky on a different row count is not a matched extension. For
interventions, keep the original finite column draws and unaffected target
parameters; fresh-population comparisons use separately identified innovations.
Hybrid DAG integration must expose the population as owned root/channel values
and allow existing downstream mechanisms, not pretend each row was independently
drawn from a scalar root distribution.

**Admission and trace.** Reuse covariance computation where appropriate, but
retain Cholesky as the source algorithm and label alternatives. Admit all
simultaneously live covariance/factor copies, per-column variants, retained JSON,
calibration, descendants and comparison populations before allocation, RNG or
replay. Dense storage grows quadratically in rows and factorization cubically;
do not inherit a 20,000-row claim from the overall task envelope. Duplicate
locations need an explicit singular-covariance policy; jitter changes the law
and must be recorded. Failures and valid weak targets must remain visible.

**Whole workbench and independent proof.** Extend generation dispatch, recipe
codec, construction export, archive/reopen, local and Colab editors together.
Show the latent location plot, covariance heatmap, factor, selected column's raw
innovation, emitted values and target equation. Distinguish generation dependence
across rows from temporal dynamics. Save mechanism edits and matched/fresh
comparisons with population identity. An independent two-row oracle can choose
RBF off-diagonal covariance 1/2, verify the Cholesky entries 1, 1/2 and
sqrt(3)/2, multiply retained innovations, then hand-compute all three target
paths. Add tampered row IDs/factors, subset/permutation replay and preallocation
failure probes. Numerical proof, actual browser evidence and hosted evidence
remain separate obligations.

## 2. P2 topology hyperprior envelopes

**Primary source fact.** TabPFN-v2 Methods, “Graph structure sampling”, states
log N is uniform and redirection probability P is Gamma with shape/rate
parameters. It allows one graph or disjoint subgraphs. The disclosed Gamma is
unbounded; the text does not explain its conversion to a probability or integer
rounding of N. It also says smaller P yields denser graphs, a claim not explained
by the implemented one-parent redirection tree. Preserve this discrepancy.
[Published 2025-01-08, Methods](https://www.nature.com/articles/s41586-024-08328-6),
[accessible primary PDF, same section](https://www.nature.com/articles/s41586-024-08328-6.pdf).

**Actual gap.** `causal_graph.py:CausalGraphPrior` has a fixed integer `nodes`
and fixed `redirection_probability`; `sample_causal_graph` implements the
one-parent redirection forest. `causal_sampling.py` chooses redirection from
`(0, 0.25, 0.5, 0.75, 1)`, not Gamma. `structural_graph_prior.py` already supplies
`AttachmentGraphPrior` and `MotifGraphPrior`; neither is a substitute for the
missing P2 laws. Existing graph families must not be rewritten as this dispatch.

**Proposed construction.** Add a versioned P2 hyperprior request containing
log-size endpoints, explicit rounding, shape/rate and bounded-probability policy.
A named bounded transform such as P=G/(1+G), G~Gamma, is implementable without
pretending it is exact P2. A separately named truncated-Gamma law could also be
implemented, with its own algorithm/admission and no silent rejection loop.
Do not clamp unbounded G. Preserve the sampled log-size and G, transformed N/P,
all law settings, stream snapshots, component allocation and the exact resulting
`CausalGraphPrior` inside a construction envelope. A fixed-size override must
be a distinct explicit profile. Any ranges and component-size partition rules
are reconstructed settings, not source facts.

The forest edge count is N minus component count regardless of P. Show degree,
ancestry and depth changes as P varies, rather than asserting increased density.
Keep the paper discrepancy unresolved until another primary source explains it;
do not invent additional edges to make the sentence true.

**Integration and proof.** Feed the realized topology through existing sampled
graph construction/identity/archive paths. Reserve worst-case configured size,
components, graph mechanisms, metadata and raw snapshots before any draw, then
retain the sampled size even if later numerical execution fails. No retries or
usefulness filters. The workbench needs requested law versus realized geometry,
a source/reconstruction badge, editable bounds and an across-seed distribution
view. Reopening must reconstruct the same hyperprior envelope, not only its final
adjacency. Independent proof should hand-transform fixed uniforms/Gamma draws,
exercise endpoint rounding and invalid bounds, verify one-parent ancestry and
N-C edges, and show identical raw records under replay. Statistical histograms
are supporting evidence, not replacements for exact law/identity checks.

## 3. Frozen multivariate mixture roots

**Primary source fact.** At the pinned TabICL revision, `_points.py` defines
`RandomGaussianMixturePoints`: centers, component weights, weighted Gaussian
matrices, shared Gaussian log-scale terms, component assignments and innovations.
`RandomPoints` itself selects only uniform, ball, Gaussian and covariance bases;
therefore promoting the mixture to a general root is an r0 extension, not proof
of that source's root-selection default.
[Pinned `_points.py`, lines 6–20 and 41–63](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py#L41-L63).
The weight factory supports uniform and noisy polynomial-decay choices.
[Pinned `_weights.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py#L5-L52).

**Already implemented.** `joint_point_plans.py:JointPointPlan`,
`joint_point_realization.py:materialize_joint_points` and
`kernel_points.py:KernelPointCloud` retain precisely these component parameters,
assignments and innovations, including shared scale mean/coefficient. They feed
kernel-matrix construction. Thus `I2.root.08`–`.11` have primitive implementations;
reimplementing centers/weights/matrices/scales would duplicate substantial work.

**Actual missing seam.** `JointRootPrior` and `JointRootBase` accept only uniform,
ball, Gaussian and covariance. `source_root_base.py:functional_root_values`,
`NetworkRecipe.root` and `PrototypeRootPlan.base` admit specific root descriptors,
not a frozen mixture-parameter object. `KernelPointCloud` couples parameters to
a fixed point count/assignments/innovations. It cannot itself serve as a reusable
root law with independently emitted row populations.

Separate a retained mixture law from row emission: freeze probabilities, centers,
column-weighted factors and log-scales once; each emission owns assignments and
Gaussian innovations. Use the existing arithmetic/weight definitions and raw
capture machinery. Add a root descriptor and source-qualified sampling profile,
then thread it through graph roots, standalone networks, prototype construction,
codecs, moment validation, interventions and metadata admission. Keep the original
finite point-cloud replay intact. A fixed law has finite conditional moments;
claims about moments integrated over random scale hyperparameters require separate
analysis and must not be inferred from a finite sampled realization.

For normalization, offer none, analytic conditional mixture moments, and named
independent-calibration adaptation where applicable. Retain constants and
calibration provenance, never fit from queried rows. Matched root interventions
retain unaffected row assignments/innovations and mechanisms; fresh-population
comparisons retain the law but resample row exogenous values explicitly. Admit
component-count times width squared, calibration/emission rows, nested prototype
banks, descendants and complete raw/archive copies before allocation or replay.

The workbench should show centers, component probabilities, covariance ellipses,
selected row assignment, weighted factor, innovation and output. Include mixture
root to neural/vector graph and mixture root to prototype-bank examples, with
editing, saved comparisons, reopen and export in both local and Colab paths.
Independent proof should hand-evaluate a two-component two-dimensional example,
verify analytic mixture mean/covariance, and detect a changed factor-column weight,
assignment, normalization constant or row identity. Repeated emissions must change
row noise while preserving the frozen parameter fingerprint.

## Existing root/prototype work and remaining combinations

`SourceRootProfile` already samples P1 normal/uniform/mixed coordinate laws and a
named P2 normal/uniform selector; `SampledSourceRootBase` already carries
independent normalization provenance. `PrototypeWeightLaw` already distinguishes
two multinomial temperature reconstructions and a Dirichlet extension.
`PrototypeRootPlan` supports source-root bases and explicit count rules;
`PrototypeBatchNetworkRecipe` and its realized envelope already bind batch row
geometry and replay identity. These are not greenfield tasks.

Prototype-output normalization is explicitly absent in
`prototype_batch_output.py` (“none at prototype output”); base calibration is not
normalization after convex mixing. This is a further meaningful composition of
`P2.root.04` with `R0.extension.16`, not a missing paper-prescribed P2 normalization
algorithm. Sampled rho/temperature/range profiles likewise require explicit r0
law choices; the source does not disclose their complete distributions. Trace
internal-noise compositions against the actual receiving network/node lifecycle
before dispatch: source-root noise composition already exists. This audit does
not establish that every prototype population, normalization and internal-stage
noise combination works or fails. Keep those combinations PENDING; do not falsely
promote them to the three confirmed missing seams above.

## Delivery boundaries

All three dispatches preserve the complete inventory and broader extension scope.
`OP.selection.01`–`.05`, observation corruption, curricula, production filtering
and mixture selection remain #30; supply interfaces without selecting policy.
Finite GP correlation across exchangeable row locations is not automatically a
temporal model; row-axis convolution and VAR remain #39. Source-private settings
remain UNKNOWN. No full-prior reproduction, scientific utility, full-issue
acceptance or performance claim follows from this report.
