# Sampled pipeline family expansion plan

Base checkpoint: `5a16ae3`. This plan retains the accepted full #28 scope;
individual implementation/review checkpoints do not remove later requirements.

## Source map and distinctions

- Spec13 and `SPEC.R02.01`–`.04` require broad structural innovation laws and
  hidden-channel coverage, distinct from observation/target noise. Existing authored
  `ExogenousDistribution` provides Gaussian, standardized uniform, Student-t,
  symmetric two-Gaussian mixture, conditional truncation, uniform-volume ball and
  explicit linear factors. `NoiseResponse` provides additive and signal-dependent
  scales. Sampling that surface is an r0 extension (`R0.extension.17`–`.20`), not
  a claim that every paper selected these innovations.
- `I2.node.05` is specifically Gaussian after converter capture and before node
  importance. Its existing sigma hyperprior and archive/RNG behavior stay intact.
  P1 shared/per-node noise statistics and TNLU scale law are separately sourced
  (`P1.net.05`, `.06`, `.09`); do not rename an r0 log-uniform law TNLU. P2 Gaussian
  edge innovations (`P2.edge.07`) and L1 feature-distribution scaling
  (`L1.graph.13`) keep their separate geometry/provenance obligations.
- `SPEC.D02`/`D03`, `I2.category.01`–`.20`: seven converter modes, configurable
  catalogs, prototype selection, input-width selection and distinct descendant
  representations. Native one-hot domain emission does not implement this catalog.
- Root-only source obligations stay distinct: P1 Zipf/clipped-Zipf/multinomial,
  P2 normal/uniform and prototype mixtures, I2 multivariate mixtures and transformed
  roots. The existing symmetric mixture must not stand in for those distributions.
  Unnamed distributions remain unknown source choices, not aliases for new defaults.

Verified pinned source: [converter](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py),
[node ordering](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py).
The categorical families select nearest calibrated centers or stochastic softmax;
mode determines whether descendants retain latent inputs, selected centers,
transformed centers, embeddings or broadcast category indices. Neighbor inputs
may reserve fewer channels than category count. Fixed independent fitting is an
explicit r0 adaptation of the source's data-dependent fitting lifecycle.

## Coherent implementation sequence

### A. Sample the complete current authored innovation surface

Add an opt-in versioned innovation prior/plan/realization and v2 pipeline prior,
leaving v1 fields, defaults, named requests, bytes and streams unchanged. The new
prior exposes family catalogs, geometry catalogs, conditional truncation selection,
parameter bounds, structural scale and response laws, and factor choices.
All named choices use the enclosing JointSamplingContext: local/global/meta,
shared/refresh categories and numerical preference identities stay inspectable.

Freeze `InputDistribution`, `ExogenousDistribution`, `NoiseResponse` and the full
per-channel scale vector into the plan/realization. Suggested explicit extension
profile: four base families; location uniform[-0.5,0.5], input scale log-uniform
[0.25,2], Student-t df catalog(0.75,1,1.5,2,3,5,10), mixture gap log-uniform[0.25,4],
optional truncation bounds(2,4,8); independent or ball geometry; identity or sampled
Gaussian-factor geometry; additive or signal-dependent responses with selectable
strength(0,0.1,0.5,1), power(0.5,1,2), offset1; structural scale log-uniform[1e-5,1].
These are documented r0 choices and configurable controls, not source probabilities.
Ball requires standard Gaussian direction inputs; incompatible selected combinations
fail before publication rather than silently ignoring selected laws.

Materialization must place realized innovation on each node before any descendant
fit. The current node's own pipeline fit still sees its bare function/root output;
its later innovation never contaminates its own captured converter calibration.
Replace sigma-only reconstruction in sampled envelopes with exact frozen noise
records. Retain factor draws and all fit/draw custody in the existing ledger and
four-stream transaction; bound metadata, factors, truncation floor and scratch
before materialization. Existing moment analysis must see the realized law and
response. Heavy tails remain available when normalization is disabled; finite
sample values cannot authorize unsupported population normalization.

Deliver this checkpoint through public decoder/generate/inspection, geometry and
attachment/motif composition, explorer JSON controls/presets and shared Colab
transport. Show family and parameters, source label and failure receipts. Verify
actual browser save/reopen/edit and matched draw semantics using public APIs.

### B. Typed categorical converter runtime and source profiles

Use a separate typed converter/recipe version rather than inserting ordinal codes
into the existing numerical-only output contract. Records explicitly name input
`[start,stop)` spans, logical output reference/vocabulary, state representation,
fit matrices/center indices, probability law and fresh emission stream. Preserve
numerical converter variants and all state/capture distinctions.

Support all seven source modes and single/neighbor/softmax/full selectors. Nearest
prototype fitting retains a permutation of independent calibration rows and
log-uniform distance exponent[0.5,4]; category count cannot silently shrink when
fit rows are insufficient. Softmax uses frozen standardization, log-positive-weight
biases and explicit temperature/factor; draw uniforms per converter. Retain sampled
embeddings and transformed-center functions using existing source-qualified point,
weight and cheap-function plans, including independent fit/raw records. Never
repair nonfinite logits silently. Exact discrete source parameter laws and ownership
will be verified against pinned functions/points/weights before coding these plans.

Reserve converter input spans before latent width/mechanism sampling. Categories
remain separately typed captures even when their descendant state is unchanged or
continuous. Descendant equations receive the declared representation with its exact
width. Later innovation/importance affects state only; target/observed captures
retain declared categorical values and truthful probabilities. Extend certificates,
admission, actual traces, feature schema, native target isolation, persistence,
recipe decoder and intervention location semantics together. Do not imply one-hot
states cover the other five representation requirements.

### C. Shared workflow and acceptance

Controls expose the law catalogs and categorical profiles, state/readout/target
semantics and independent-fit provenance. JSON remains a complete escape hatch,
not a substitute for usable family controls. Save/reopen/download/inspect and
local Colab callback paths use the same public decoder. Hosted remains NOT_RUN
until actual hosted evidence exists. Source-specific unfinished roots, parameter
laws, residual families and other inventory options stay required.

For each checkpoint: public RED/GREEN, independent NumPy arithmetic, unequal-span
and parent-order counterexamples, exact raw replay and untouched-stream failures,
malformed/tamper/admission checks, old-archive compatibility and actual Chromium.
No full suite from this worker; parent owns combined verification. Evidence lives
under ignored `docs/evidence/issue-28-pipeline-family-expansion/`.
