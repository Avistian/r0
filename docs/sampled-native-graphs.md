# Sampled native graphs

`NativeGraphRequest` and `sample_native_graph` construct native mixed-domain DAGs
with one shared `JointSamplingContext`. The request declares parent order,
observed variable/channel references, target reference, row/split controls,
per-node `NativeDomainPrior` records and the existing root/function priors.
Target task type and class count are derived from the sampled target domain.
No class label adapter runs.

Each node owns `node:i/domain:0` followed by its root or function subtree. Named
kind, cardinality, emission, frequency-family, concentration, temperature and
threshold choices use the shared context. Fixed settings own explicit forced
receipts. Cached names retain their contracts: incompatible stochastic catalogs
are rejected rather than opening an unrelated preference scope. Global mode
shares named hyperparameters; raw frequency vectors remain fresh per node.

Cardinality catalogs can include high-cardinality features, bounded by the
request's channel/resource limits. Classification targets remain subject to the
shared task contract's 100-class limit; an incompatible target prior is rejected
before sampling. Children consume exactly K one-hot columns for a K-category
variable. All eight existing frozen function families remain available.

## Frequency and threshold laws

The source inventory identifies discretization, stochastic-softmax categories,
consistent category/one-hot widths and Dirichlet class frequencies in
[TabAbax §3.3](https://arxiv.org/html/2611.00562v1#S3.SS3). Its additional frequency
catalog is undisclosed. [Kumo Appendix A.4](https://arxiv.org/html/2611.23001v1#A1.SS4)
identifies deterministic argmax alternatives. These references motivate named
options, not a claim to reproduce either complete prior.

- Uniform frequencies are an explicit r0 baseline.
- Dirichlet frequencies normalize retained independent Gamma(alpha, 1) draws.
  The symmetric family is source-backed; the configurable alpha interval and
  its log-scale named hyperprior are r0 adaptations.
- Logistic-normal frequencies are an r0 extension, obtained by normalizing
  exponentials of retained Gaussian draws. The concentration control acts as
  their logit standard deviation.

`AsymmetricNativeDomainPrior` opts into `native-asymmetric-domain-prior-v1`.
Its `concentration_vectors` contains exactly one positive finite tuple for every
reachable discrete cardinality: tuple length is the cardinality key; duplicate,
missing and unreachable lengths are rejected. Binary domains require length two.
For example, a categorical prior with `cardinalities=(3, 4)` accepts
`concentration_vectors=((0.5, 2.0, 7.0), (1.0, 2.0, 3.0, 4.0))`.
The explicit `asymmetric-dirichlet` frequency token can be mixed with uniform,
symmetric Dirichlet and logistic-normal tokens. In shared scopes, stochastic
frequency catalogs must still agree; vectors are configured per node/cardinality,
not sampled shared preferences.

These are **absolute per-category concentrations**, not weights multiplied by the
scalar `concentration` hyperparameter. The asymmetric branch does not sample that
scalar; its inherited plan scalar stays the unused sentinel 1.0. Its versioned
plan records the selected tuple plus a forced `configured-cardinality-vector`
receipt. Each category owns a raw Gamma(alpha_i, 1) record; all are normalized by
their common sum. This is the standard [NumPy Dirichlet construction](https://numpy.org/doc/2.3/reference/random/generated/numpy.random.Generator.dirichlet.html).
Admission reserves the per-category records and vector metadata before realization.
The asymmetric family is a named r0 extension, not a disclosed TabAbax law.

Explorer JSON controls select the new prior version and concentration vectors
inside any sampled-native domain, including native parent composition. Saved
records retain those controls and plans; authored runtime recipes reuse the
existing frequency domain. Inspection, typed export and reopen use the existing
native path. Historical prior/plan fields, selector names and symmetric draws
remain unchanged. No query data calibrates these frequency vectors.

`FrequencyVariableDomain` uses `native-frequency-domain-v2`; unchanged authored
`VariableDomain` records retain their v1 fields and bytes. For softmax the adapted
scores are `score / temperature + log(frequency)`. Argmax uses
`score + log(frequency)`. These offsets influence category probabilities; they do
**not** guarantee dataset marginal class frequencies. Ties select the first class.

Threshold choices are also explicit r0 adaptations: cumulative frequency weights
mapped through standard-normal reference quantiles, through uniform [-1, 1]
reference quantiles, or independently sorted Gaussian cutpoints. A positive
sampled/fixed scale multiplies the cutpoints. These are fixed population rules,
not quantiles fitted to context/query rows. In the independently sorted variant,
frequency weights remain recorded but do not determine cutpoints. No finite
sample is forced to contain every declared category.

## Calibration, replay and bounds

Receiving fits draw fresh native ancestor populations. Calibration propagation
uses actual integer categorical emissions and their one-hot child states, never
continuous category codes. The aggregate raw ledger includes root draws,
structural innovations and category uniforms for each receiving population, plus
frequency, threshold, function and fitting draws. All caller streams publish
atomically after successful construction; rejection rolls all four back.

The `sampled-native-graph-v1` archive retains request, domain/mechanism plans,
shared registry, all realizations, raw records, runtime recipe and initial/final
snapshots. Reopening regenerates the complete construction and compares every
retained field. Generation, original-row inspection and group save/load use the
shared public interfaces. Resplitting changes episode identity while preserving
source identity, values and mechanism records.

Aggregate admission reserves numerical mechanism workspaces, typed prefix
buffers, probabilities, retained draws, metadata and replay copies. The raw
ledger receives only the remaining budget. Estimates are not measured RSS.
Nonfinite/underflowed frequencies, reference-quantile endpoint rounding and
out-of-budget constructions fail explicitly without retry or category deletion.

## Remaining scope

Each node currently contains one vector continuous variable or one scalar
binary/categorical variable. The indexed `domain:0` ownership seam leaves room
for ordered heterogeneous variables, but that extension remains required #28
work. Other remaining work includes additional disclosed source-specific frequency laws when available, and native
intervention-pair APIs. The current sampled-native JSON controls and shared
workbench are described in [Sampled native explorer](sampled-native-explorer.md);
hosted notebook execution remains separately unverified. Neither a source's mixture
probabilities nor undisclosed laws have been silently invented as defaults.
