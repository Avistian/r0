# Issue #28 newer-source option check

Checked 2026-11-26 using the research workflow. This bounded extraction supplements
the initial audit; it is not implementation or checkpoint-reproduction evidence.
The accepted specification remains authoritative. Every semicolon-separated choice
below requires its own inventory entry. Numeric recipe tuning remains delegated.

## Source register

| ID | Pinned public source | Version date shown by source |
|---|---|---|
| I2 | [TabICLv2](https://arxiv.org/html/2602.11139v1), Appendix E; source commit `9530bd65a7b806f6ad4f9f4335af745577e9d3cc` | 2026-02-11 |
| M2 | [MITRA-v2](https://arxiv.org/html/2609.04540v1), §§2.2–2.3 | 2026-09-03 |
| L2 | [LimiX-2](https://arxiv.org/html/2609.17488v1), §§4.1–4.5 | 2026-09-15 |
| A1 | [TabAbax](https://arxiv.org/html/2611.00562v1), §3.3 | 2026-10-31 |
| K1 | [Kumo Tabular](https://arxiv.org/html/2611.23001v1), Appendix A | 2026-11-17 |
| G1 | [TabFM](https://arxiv.org/html/2609.37959v1), §3.2 | 2026-09-29 |
| P35 | [TabPFN-3.5](https://arxiv.org/html/2609.17895v1), §3.3 | 2026-09-15 |

I2 code was inspected in primary-source copies fetched by the source-inventory
agent into `/tmp/r0-28-source-read`. Permanent citations below point to the exact
commit, not that temporary directory. No upstream code was copied into r0.

## Pinned I2 options that broad labels can hide

### Functions

Source: [_function.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py).

- I2-F01: Eight dispatch choices: MLP; tree; nearest-prototype discretization;
  linear; quadratic; GP; EM-style cluster assignment; product. Presets: all eight;
  `tabpfnv2` subset MLP/tree/discretization; explicitly supplied family list.
- I2-F02: MLP independently enables beginning and ending activations. Linear
  mappings themselves have no bias; activation rescaling supplies shifts.
- I2-F03: Oblivious tree ensemble, standard-deviation-weighted split dimensions,
  thresholds sampled from input rows, Gaussian leaf vectors, mean ensemble output.
- I2-F04: Nearest-prototype discretization uses observed centers and linear-mapped
  center outputs; distance exponent spans 0.5–4.
- I2-F05: GP has axis-aligned product-spectrum and radial-spectrum options; the
  latter adds a weighted random input projection. Both use random Fourier features.
- I2-F06: Quadratic includes constant and linear terms; source subsamples at most
  20 inputs. Preserve that reference limit separately from broader exploration.
- I2-F07: EM-style assignment independently varies distance exponent, powered
  distance and log-width spread, then linearly maps soft assignment probabilities.
- I2-F08: Product children are GP/tree/discretization/linear/quadratic only; two
  children receive the same input. MLP/cluster/recursive-product children are absent.

### Matrices and positive weights

Sources: [_matrix.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py),
[_weights.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py).

- I2-W01: Gaussian; positive-importance-weighted Gaussian; Gaussian-factor product
  with weighted inner dimension; signed Laplace-kernel matrix; activation-transformed
  matrix. Top-level `default` and Gaussian-only selections are separate recipes.
- I2-W02: Kernel matrices use Gaussian-mixture points in three dimensions,
  Euclidean distances, exponential distance decay and independent signs.
- I2-W03: Activation matrices use a nonrecursive four-family matrix selector,
  plain activations and additive Gaussian jitter. Top-level matrices add further
  tiny Gaussian jitter and normalize each row.
- I2-W04: `SimpleRandomWeights` combines shuffled polynomial decay with Gaussian
  perturbations in log space, softmax and norm scaling. `UniformRandomWeights`
  samples independent uniforms. These are distinct from Dirichlet preferences.
- I2-W05: The named singular-values class uses Gaussian, not orthogonal, factors;
  its weighted inner coefficients are not exactly the resulting singular values.

### Roots, channel state and aggregation

Sources: [_points.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py),
[_node_function.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py),
[_multi_function.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_multi_function.py).

- I2-R01: Root base choices: uniform hypercube; uniform-volume ball; Gaussian;
  random covariance. `RandomCirclePoints` is a ball distribution, not a sphere.
  Every `RandomPoints` base is subsequently transformed by a random function.
- I2-R02: Covariance construction chooses uniform or Gaussian base and applies a
  weighted Gaussian matrix. The separate Gaussian-mixture sampler varies centers,
  mixture weights, per-component matrices and component scales.
- I2-N01: Latent width adds reserved converter channels. Optional per-channel
  importance scaling, whole-node RMS-vector normalization and whole-node scalar
  importance are distinct operations.
- I2-N02: Optional Gaussian internal noise is added **after** converter readout,
  so descendants receive it while already extracted columns do not. Preserve this
  source ordering as a named variant.
- I2-C01: Concatenate parents and apply one function; or transform each parent
  separately then sum; product; maximum; logsumexp. Restrict-to-concat and
  restrict-to-aggregation selectors are documented source options.

### Converters and activations

Sources: [_converter.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py),
[_activation.py](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py).

- I2-D01: Numerical identity; Kumaraswamy warp; activation warp. Historical
  behavior warps hidden state and retains original readout; corrected behavior
  warps readout and retains hidden state. Target warping can be disabled.
- I2-D02: Seven categorical modes: `neighbor_id`; `neighbor_disc`;
  `neighbor_func`; `neighbor_int`; `softmax_id`; `softmax_disc`; `softmax_int`.
  There is no `softmax_func`. Preserve single-mode, neighbor-only, softmax-only and
  all-mode selection recipes.
- I2-D03: Neighbor conversion chooses all category-count channels or a sampled
  subset; softmax requires category-count channels. State choices are unchanged
  vector, selected prototype/embedding, transformed prototype, or broadcast index.
- I2-D04: Softmax standardizes logits, adds log-positive-weight biases and samples
  categories. Its nonfinite-logit replacement is a source safeguard, not authority
  for silently repairing r0 numerical failures.
- I2-A01: Fixed activation list matches A01–A04 in the initial audit. Additional
  semantic details: sign(0)=+1; Heaviside(0)=1; modulo is fractional part;
  rank/argsort are channel operations; is-maximum includes all tied maxima.
- I2-A02: Standard activation samples fixed/ReLU-power/signed-power only. Plain
  activation additionally samples stabilized reciprocal powers and integer powers.
  GP activation is commented out, not selectable in this pinned file.
- I2-A03: Standard activation wraps input standardization, sampled rescale/shift,
  activation and output standardization. Shifts use sampled input values.
  L2Normalize uses a batch-calibrated scalar, not per-row unit normalization.

The accepted independent calibration rule must be used for training-ready r0
variants. All input-row-dependent fitting above needs retained independent
calibration constants, clearly labeled as an adaptation.

## Newer-paper findings and exact boundaries

### M2: Hybrid SCM

Required independent option records: MLP; decision tree; ExtraTrees; random forest;
GP via RFF. The paper also names 1D convolution and VAR: preserve convolution-axis
semantics and route row dependence to #39. Graph is a chain with independently
sampled skip parents and variable-width vector nodes. Multiple-parent nodes mix
the nonlinear mechanism output with direct parent projections using mean;
softmax-weighted sum; MLP projection; product; maximum. Output treatment includes
standardization; feature-importance scaling; L2 normalization. Feature/target
selection has **k-means; farthest-point; random** variants. Classification uses
quantile bucketing; some features become categorical. Independent calibration
is an r0 adaptation. Standalone outer decision tree, ExtraTrees, gradient boosting,
random forest and directly sampled forest belong to #29; outer mixture policy to
#30. The paper does not specify all tree fit/sampling internals or selection details.
[M2 §§2.2–2.3](https://arxiv.org/html/2609.04540v1#S2.SS3).

### L2: Motifs, rewrites and compositional mechanisms

Required records: normal/uniform/beta hyperparameter-distribution families; chain,
confounding and collider motifs; recursive motif expansion; edge redirection;
local-path replacement; node structural transformation. Edge functions: MLP;
CNN; decision tree; linear; kernel; piecewise; periodic; multiplicative; compositions.
Aggregators: averaging; weighted; neural. Root distribution types/parameters vary,
but their exact catalog is unspecified. Feature/target selection uses subgraph and
redundancy descriptors followed by multi-objective filtering; #28 must expose the
descriptors while #30 owns selection policy. Observation linear/monotone/log/exp
warps and compositions belong to #30. Random-interval classification with varied
class frequencies needs a target-adapter record. CNN axis, motif/rewrite samplers,
kernel catalog and filtering algorithms remain unspecified.
[L2 §§4.1–4.5](https://arxiv.org/html/2609.17488v1#S4).

### A1: Typed readout and prior variations

Preserve Cauchy/vector-node base; independently selected per-node function and
conditional parameter family; partial latent-channel observation; discretization
and stochastic softmax categoricals; exact agreement among declared cardinality,
emitted categories and one-hot width; Dirichlet class frequencies; no post-hoc
target warp. Its additional class-frequency distributions and richer function
families are unnamed: UNKNOWN, not invented alternatives. Final signal rejection
is #30. Caption language describes richer converter families while prose describes
per-node function changes; retain that ambiguity rather than claiming a precise
release algorithm. [A1 §3.3](https://arxiv.org/html/2611.00562v1#S3.SS3).

### K1: Correlated choices and conversion differences

Required: separate per-table Dirichlet vectors for function, activation and
converter choices; graph retention despite no feature/target shared ancestor;
cluster L1/L2 distance and variable log-width spread; products admitting all seven
non-product families; deterministic argmax categorical readout. Readout-only
Kumaraswamy warp is distinct from the historical hidden-state warp. Numerical
measurement noise, shared-noise blocks, feature coarsening, rank-based
high-cardinality coding, heavy-tail target transforms and missingness are #30.
Target recomputation from selected features uses linear/MLP/GP mechanisms: expose
those #28 mechanisms while #30 owns postprocessing selection. Learnability
rejection must remain separate from valid disconnected-task generation.
[K1 Appendix A](https://arxiv.org/html/2611.23001v1#A1).

### G1 and P35: retain explicit unresolved records

G1 adds a concrete **dual classification/regression target per table** construction
to the graph-and-readout interface. Its §3.2 describes SCM nonlinear transformations,
algebraic aggregation and mixed data but supplies no new exact function catalog.
Missingness/noise/class-balance policy belongs to #30. Record dual-target algorithm
details as UNKNOWN, with a named r0 reconstruction required before coverage claims.
[G1 §3.2](https://arxiv.org/html/2609.37959v1#S3.SS2).

P35 §3.3 documents high-cardinality categoricals, larger feature counts and
train/test group shifts. The first two require scalable #28 typed-channel support;
group shifts need #39/#30 interfaces. No exact additional generator algorithms are
provided. This is an explicit disclosure gap, not a new named mechanism family.
[P35 §3.3](https://arxiv.org/html/2609.17895v1#S3.SS3).

## Discrepancies requiring named variants

1. **Cluster widths:** K1 A.1 describes I2 log-width spread as fixed 0.1; pinned
   `_function.py::RandomEMAssignmentFunction` samples the spread from 0 to 1.
   Preserve the K1-described historical I2 setting and pinned source setting.
2. **Kernel-matrix points:** I2 Appendix E.10 says random covariance points;
   pinned `_matrix.py` uses Gaussian-mixture points. Keep both constructions.
3. **Converter correction:** historical I2 hidden-state warp and corrected
   readout warp coexist in the pinned source. Kumo's paper comparison is about
   historical pretraining behavior, not proof the source lacks a corrected option.
4. **Activation semantics:** retain paper interval [0,1] versus code absolute-value
   interval; paper one-hot argmax versus code tied-maximum mask. The existing
   audit already records these and they remain unresolved by broad family labels.

These are source-level observations. They do not identify which complete source
configuration produced any published checkpoint. Code-only choices, published
recipes and r0 adaptations must remain distinguishable in the finite inventory.
