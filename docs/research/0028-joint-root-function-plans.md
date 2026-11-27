# Issue 28: joint root bases and function plans

Design only, 2026-11-27. Preflight: `/tmp/r0-28`, `feat/issue-28-generators`,
SOURCE `10d9b6c55107ff55c7e710f9adc01f96fc3c491b`, untracked `.venv`; four live
slots. Only this note authored; no executions/tests/downloads/cloud/installs.
Checks **NOT_RUN**, resources **NOT_MEASURED**; full [scope][S] remains required.

## Provenance and implementation boundary

Join root/function trees to `JointSamplingContext`/`JointMatrixPlan`; every root
requires RandomFunction. Reuse [integration][I], [matrix design][D], [node laws][L]
and [checkpoint][C]. [P] [F]

Points/base/etc. cache: `/tmp/node-pipeline-sampling-research`; function copy:
`/tmp/r0-28-source-read/I2func.py`. [Earlier extraction][E] attributes it to
TabICL `9530bd65a7b806f6ad4f9f4335af745577e9d3cc` (links below). Current **local**
SHA256, not fresh upstream verification:

| Copy | SHA256 |
| --- | --- |
| `I2func.py` | `ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c` |

Checkpoint configuration UNKNOWN.

## All root bases

For R rows, width d>0: `random_base_points` ordered **uniform, circle, Gaussian,
covariance**. Select base; construct unfitted RandomFunction(d,d); realize base;
fit/call function. Covariance choices precede function selector. Source rebuilds
base/function per sample call; freezing per task is decision-12 adaptation. [P] [B] [S]

| Base | Exact raw construction, output (R,d) |
| --- | --- |
| uniform | U (R,d) in [0,1); B=-1+2U, covariance I/3. [P] |
| circle | Gaussian G (R,d), normalize rows; independent U (R,1), multiply by U^(1/d). Uniform **unit ball**, covariance I/(d+2), not sphere. Nonzero direction plus U=0 gives valid origin. Zero direction divides by zero in source; explicitly reject without retry as existing r0 does. [P] [exo] |
| Gaussian | Independent standard normals (R,d); no extra normalization/jitter. [P] |
| covariance | Gaussian A (d,d), RandomWeights (1,d) w, F=A diag(w); then `random_covariance_base_points` ordered **uniform, Gaussian**; draw B (R,d), output B F^T. Factor shared across rows, zero center. No matrix selector/wrapper, mixture count/assignments/log scales. [P] |

F is a factor: Gaussian covariance FF^T, uniform FF^T/3. Singular/zero F is valid.
Moment formulas are deductions from these laws. [P]

Weights share `simple_random_weights_decay_rate` (log
[0.1/log(1+d),6], boundary mass enabled) and `simple_random_weights_std_scale`
(log [0.0001,10], no boundary mass), source shape (1,). Source `simple` means r0
`polynomial`: Gaussian log perturbations, shuffled decay, softmax, RMS-one output.
Uniform means unchanged U[0,1), no numerical events or node RMS normalization.
Zero factor weights are valid, unlike zero multinomial mass. [W] [jw] [wv]

## Exact function requests and shared scopes

`random_function_fct_type` expands tokens in supplied order:
`default` = **mlp, tree, disc, lin, quad, gp, em, prod**;
`tabpfnv2` = **mlp, tree, disc**; explicit comma-separated names allowed. Preserve
duplicates with distinct slot IDs. No source trimming; unknown tokens reject.
Product's fixed selector
`cheap_random_function_fct_type` = **gp, tree, disc, lin, quad**, regardless of
outer preset; exactly two children, no MLP, EM or recursive product. [F] [Q]

Requests are scalar/meta/no boundary mass except weights. `randint(lo,hi)` floors
then clips [lo,hi-1]; log interpolates in log space. `use_int=True` rounds with
NumPy rint/ties-to-even. Booleans are categorical false/true. [B]

| Function | Named request sequence before lazy children |
| --- | --- |
| MLP | `nn_width`: log randint(1,128), 1..127; `nn_n_layers`: log randint(1,4), 1..3; booleans `nn_start_act`, `nn_end_act`. [F] |
| tree | `rt_n_trees`: log numerical [1,128], **round**, 1..128; `rt_depth`: linear randint(1,8), 1..7. [F] |
| discretization | Optional `rand_discr_n_centers`: log randint(2,min(R,C)), C=`max_discretization_cardinality`, default 256; `cat_disc_p`: log [0.5,4]. [F] [Q] |
| linear/quadratic | No own scalar request; reach matrix subtree. Quadratic feature subset uses raw randomness. [F] |
| GP | `gp_a`: log [2,20]; boolean `generalized_gp_use_product_kernel`; radial only: weight decay/scale then `rand_gen_gp_scale`: log [0.5,10]. [F] |
| EM | `random_em_components`: log randint(2,max(16,2o)+1), 2..max(16,2o); `gen_em_assignment_std_factor`: linear [0,1]; `gen_em_assignment_p_inv`: log [1,4]; `gen_em_assignment_q`: linear [1,2]; lazy output linear. [F] |
| product | Entire first cheap selector/subtree, then second cheap selector/subtree. [F] |

Discretization R=1: one center, no count/permutation. For R>=2, H=min(R,C)=2
returns **1** (source clips floor(2) to [2,1]); H>2 gives 2..H-1; C<2 fails bounds.
Explicit positive constructor count skips named count, caps at R, still permutes
for R>=2. Tag these cases; existing floor validator cannot silently replace H=2.
Training-ready R is independent fit rows, not emitted/query rows. [F] [B] [numeric] [S]

Standard activation requests: `act_type` has four distinct fixed slots,
relu-power, signed-power; lazy order then `rr_scale` log [1,10], then
`acts_torch` or `relu_random_power_exponent`/`random_power_exponent`, log
[0.1,10]. Fixed order (r0 names): tanh, leaky_relu, elu, identity, selu, silu,
relu, softplus, relu6, hardtanh, sign_positive, heaviside_positive, gaussian_bump,
exp, interval_abs_le_1, sin, square, abs, log_clip_abs, rank, sigmoid, round,
modulo, softmax, tied_max, argsort, log_sigmoid. Matrix subtrees additionally
reach `act_type_plain`: eight fixed slots plus relu/signed/negative/integer
powers. `neg_random_power_exponent` is positive log [0.1,10] applied negatively;
`rand_int_power_exponent` is randint(2,6). Reuse the existing catalogs and
JointActivationPlan order, not a new independent sampler. [A] [ja] [al]

Keep all five RandomMatrix/four SimpleRandomMatrix families and kernel requests
from [D]/[jm]. Raw covariance/GP factors bypass wrappers. Root/function/matrix/
node/converter names share outer preferences; categorical meta's fresh
SimpleRandomWeights(Context()) scope stays separate. Meta shares hyperparameters,
local is independent, global blindly reuses first values. Preserve strict-batch
and scalar-quantile adaptations, stable catalogs and dynamic count/width contracts;
never split source names silently. [B] [j] [jr] [jw]

## Function constructions and lazy fitting

Here i/o are widths (root i=o=d), K=256 frequencies by default, t trees, h depth,
c centers, k=min(i,20)+1.

| Variant | Raw order, shapes and equation |
| --- | --- |
| linear | JointMatrix(1,o,i), realized M; f(X)=X M^T, no bias. Normalized M rows become W columns; sampling (i,o) changes law. [F] [M] |
| MLP | L linears, widths [i,w,...,w,o], L-1 hidden widths. L-1+start+end standard wrappers; width sampled even for L=1. Lazy order: optional start activation, linear 0, next activation if present, linear 1, etc. Share preferences, not fit constants. [F] |
| tree | Feature masses max(population std(X_j),1e-8), replacing nonfinite std by 1e-8 in source. Draw t*h features with replacement, t*h rows, thresholds at coordinates, Gaussian leaves (t,2^h,o). Shared split per depth. Bit l=(X_j>threshold); index sum(bit_l*2^l); average leaves. Reject nonfinite inputs before source fallback. [F] |
| discretization | Centers X[permutation[:c]] or sole row; p; linear map fitted on centers using matrix (1,o,i), targets centers @ M^T. Evaluate p-distance and select first minimum. p<1 is source quasi-distance, not norm. [F] |
| quadratic | If i>20 raw permutation selects 20 inputs, else ordered all inputs without randomness. Append constant 1 last. One JointMatrix(batch=o, rows=k, columns=k); f_j=z^T Q_j z. Q need not be symmetric. Includes cross/linear/constant terms; kernel batch shares one cloud. [F] [mr] |
| GP product spectrum | U (i,K), clamp [1e-6,1-1e-6]; F=U^(1/(1-a))-1, nonnegative, no signs or input matrix/scale. Then phase=2pi U (1,K), coefficients=Gaussian(K,o)/sqrt(K). f(X)=cos(XF+phase) coefficients. Finite RFF, not exact GP. [F] |
| GP radial spectrum | U (K), same clamp/inverse for radii; Gaussian directions (i,K), normalize columns, multiply by radii. Raw Gaussian A (i,i), weights (1,i), scale s: F=(s diag(w) A) directions. **Row** weights, unlike root covariance. Direct RandomGaussianMatrix has no selector/jitter/row normalization. Then phases/coefficients above. Zero direction rejects without retry. [F] [M] |
| EM | c input rows with replacement plus Gaussian jitter (c,i), scale 1; spread b then Gaussian z(c); sigma=exp(bz); constants=-log(2pi sigma²)/2, **not multiplied by i**. Instantiate output linear; sample p,q. First transform computes logits=constants-(distance_p/sigma)^q, softmax over c; only then fit output linear (1,o,c). Despite name `p_inv`, use p, not 1/p. [F] |
| product | Two complete cheap children on identical X, left before right; outputs (R,o), coordinatewise multiply. No outer averaging/rescale; right never fits on left output. [F] |

Standard wrapper: population Standardize (epsilon 1e-4), s*(x+bias), primitive,
Standardize; bias uses negative per-column sampled row values. Fit each stage on
its actual preceding output. Freeze thresholds/centers/anchors/constants on
independent rows, never refit emission. Constructors do not sample children. [A] [B] [F] [fit]

## Reuse and irreducible work

| Local owner | Reuse / boundary at SOURCE |
| --- | --- |
| InputDistribution / ExogenousDistribution | Gaussian/ball available. Uniform [-sqrt(3),sqrt(3)) needs scale=1/sqrt(3). Factor applies B @ factor: store F^T; retain raw provenance separately. [inputs] [exo] |
| RootInputCausalNode / causal_values | Separate root/innovation streams; new function attachment needed, root_scale ones/bias zero. Primitive activation cannot represent all functions. [nodes] [eval] |
| KernelPointCloud(random-covariance) | Reuse factor/base arithmetic; finite cloud, not reusable law. One zero-centered component, no log scale; required assignment/mass scaffolding is deterministic, not source draws. [points] |
| JointMatrixPlan / Realization | Reuse all linear/quadratic branches/batches with transpose. Raw covariance/GP require raw Gaussian/weights, not normalized Gaussian branch. [jm] [mr] [wc] |
| JointActivationPlan / calibrated records | Reuse primitives/wrappers. Caller samples do not prove ancestry; exact-type checks exclude general functions. [ja] [fit] [nodes] |
| CausalNodePipeline | Reuse post-function normalization/importance/L2/converter capture/noise; function belongs upstream. [pipe] [eval] |
| fit_source_nodes | Gaussian affine fitter; plain bypasses ancestry; no root-function/pipeline input. Attaching them after fitting gives stale descendants. [oldfit] [I] |
| Fourier / quadratic / diagnostic MLP/tree | Generators resample different laws/normalizations; quadratic basis differs, tree is not oblivious. Fourier evaluator accepts supplied arrays with identity normalization; add typed provenance, never relabel independently generated records. [fourier] [quad] [diag] [tree] [signals] |

Required: eight typed records/evaluators/decoders/requests, root attachment,
provenance/fit, shape/ancestry/moments, graph transaction/admission. Paper trees,
parent multifunctions, categorical states/converters, profiles and workbench/
local/Colab remain required. [F] [nodes] [ancestry] [moments] [S]

## Minimal typed tree, manifest and transaction

Proposed frozen tagged records:

```text
JointRootBasePlan(operation, width, family, covariance_weights?, covariance_base?)
JointRootPlan(base, function, source_identity, policy_versions)
JointFunctionPlan(operation, input_width, output_width, branch):
  Linear(matrix_plan, transpose=True)
  Sequence(ordered_linear_and_activation_plans)
  Quadratic(selected_width, matrix_batch_plan)
  RFF(spectrum, a, frequency_count, radial_weights?, radial_scale?)
  ObliviousTrees(count, depth)
  Discretization(count_mode, fit_rows, center_count, p, linear_plan)
  EMAssignment(component_count, spread, p, q, linear_plan)
  Product(left_cheap_plan, right_cheap_plan)
FrozenRootFunction(base_construction, function_construction_tree, fit_receipts)
RawManifestEntry(owner_path, role, stream_id, distribution, shape, destination)
```

Derive exact expected events/catalogs/values/shapes on decode; sequences compose,
products agree, roots map d to d. Register every operation once, including
event-free uniform weights/raw factors/deterministic selections. Current context
marks completion only with descendant events. Each event/raw field has one leaf
owner; parent containment allowed, aliases/orphans/duplicate paths rejected. [j] [jr] [jm]

Manifest owns every raw draw listed above, including matrix/vector internals,
shift indices and fit inputs/innovations. Retain raw and derived fields needed
for reconstruction; deterministic stages name formula/version/constants/inputs,
not invented draws. Seeds alone do not prove ownership. [P] [F] [A] [C]

Separate streams: named preferences/decisions; raw construction; fit inputs;
fit row/shift choices; fit innovations; emitted root inputs; emitted innovations.
Freeze law/factor/function; matched interventions retain rows. Plan-before-realize,
NumPy streams/float64 are named adaptations, not Torch bit identity. [B] [C] [eval] [S]

Stage context/all streams/registry/tree in one transaction: plan/admit, realize,
fit topologically, validate, commit. Root function then pipeline/noise/importance
must precede descendant fit. Receipts bind calibration identity/input digest,
ancestors and pipeline variant. No query rows/labels, retries or private `_rng`.
Helper atomicity is not graph atomicity; finite samples do not prove population
moments at actual normalization inputs, including products. [j] [fit] [moments] [I]

## Decoder and aggregate bounds

Before conversion bound bytes/depth/nodes/containers/strings/paths/numeric leaves/
axes. Reject unknown fields/versions, bool indices/dimensions, ragged/object
arrays, nonfinite values/bad references. Check 2^h/i*K/products before allocation.
Source/exploration limits stay distinct; caller caps override payload. Per-matrix
preflight is not graph admission. [decode] [limits]

Count unique shared preferences/clouds, ledger/plans/manifest/receipts and all
stored copies. Retained terms: covariance d²+weights; ball R*d+R; tree t*h
indices/thresholds+t*2^h*o leaves; discretization c*i+c*o; EM c*i+c widths/constants
+output matrix; RFF i*K+K+K*o+radial i²/weights; quadratic o*k²+indices. Add raw
copies/base samples. [P] [F] [mr]

Peak adds live ancestors/concat/fit arrays, RFF R*K, tree R*t*h and R*t*o,
disc/EM R*c (R*c*i if broadcast), quadratic R*k/contractions, MLP intermediates,
matrix buffers/pipeline clones. Chunk or admit all buffers, label rounding-order
adaptations. Admit aggregate before children; counts/estimates/RSS differ. [F]
[pipe] [limits] [C]

## Independent literal oracles and adversarial cases

Oracles **NOT_RUN**; arithmetic deductions. [S]

| Case | Discriminating expectation |
| --- | --- |
| Covariance | A=[[1,2],[3,4]], w=(1/2,1/4), B=(2,-4): F=[[1/2,1/2],[3/2,1]], B F^T=(-1,-1). Gaussian covariance [[1/2,5/4],[5/4,13/4]], uniform divides by 3. [P] |
| Ball | d=2, direction (3,4), U=1/4 gives (0.3,0.4). U=0 gives origin for valid direction; zero direction rejects. d=1 gives interval, not two endpoints. [P] |
| Quadratic | z=(2,1), Q=[[1,3],[5,7]] gives 4+6+10+7=27, including asymmetric cross terms/constant. [F] |
| Tree | h=2, dims (0,1), thresholds (1,2), leaves (10,20,30,40): X=(1,3) gives bits (0,1), index 2, value 30. Two trees yielding 30,6 average 18; equality left. [F] |
| Integers | Tree-count raw 2.5/3.5 rounds 2/4; endpoint 128 stays 128. Depth endpoint 8 clips 7. Discretization R=C=2 count=1; R=1 omits count/permutation. [B] [F] |
| Discretization | Centers (-1),(1), targets (-2),(2), X=0 selects first target -2; reversing center order changes tie result. [F] |
| GP | a=2,U=1/2 gives radius 1. Radial direction (3,4) normalizes (3/5,4/5); A=[[1,2],[3,4]], w=(1/2,1/4), s=1 maps to (11/10,5/4), exposing row scaling. [F] |
| EM | i=2, equal zero distances, sigmas (1,2): constants differ by log 2, softmax (2/3,1/3), not (4/5,1/5). Identity output exposes missing source dimension factor. [F] |
| Ancestor | Root output (-1,1), post-converter noise (2,-2), importance 3 gives descendant state (3,-3); next inverse std 1/(3+1e-4), not 1/(1+1e-4). Readout may stay (-1,1). [pipe] [I] |

Cover every branch, MLP flags/L=1, cheap children in both positions, all modes/
policies, duplicate slots and outer/nested scopes. [F] [B]

Reject altered events/dimensions/values/order, duplicate event-free paths, wrong
shapes/transpose, radial GP selector, recursive product, invalid indices, stale
derived values/digests/ancestry and aggregate overflow despite valid children.
Preserve finite zero/singular factors, constants and saturated/weak draws. [F] [jr] [S]

Fail after selectors/weights/child/matrix/direction/tree/EM/pipeline/freeze: roll
back all streams/caches/ledger/paths/registry/graph; retain separate reason.
Check legacy bytes/seeds, new count/query/chunk independence at fixed fit/geometry.
GPU/browser/Colab/scale/scientific evidence remains pending. [j] [C] [S]

[S]: ../specs/0028-causal-network-generators.md
[I]: 0028-pipeline-sampling-integration.md
[D]: 0028-joint-matrix-plans.md
[L]: 0028-node-pipeline-sampling-laws.md
[C]: ../issue-28-joint-matrices.md
[E]: 0028-newer-source-check.md
[F]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py
[P]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py
[B]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py
[A]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py
[W]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py
[M]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py
[Q]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py
[inputs]: ../../src/r0/modules/priors/lib/inputs.py
[exo]: ../../src/r0/modules/priors/lib/exogenous.py
[nodes]: ../../src/r0/modules/priors/lib/causal_nodes.py
[eval]: ../../src/r0/modules/priors/lib/causal_evaluation.py
[points]: ../../src/r0/modules/priors/lib/kernel_points.py
[wc]: ../../src/r0/modules/priors/lib/weight_constructions.py
[wv]: ../../src/r0/modules/priors/lib/weight_vectors.py
[jw]: ../../src/r0/modules/priors/lib/joint_weight_plans.py
[jm]: ../../src/r0/modules/priors/lib/joint_matrix_plans.py
[mr]: ../../src/r0/modules/priors/lib/joint_matrix_realization.py
[ja]: ../../src/r0/modules/priors/lib/joint_activation_plans.py
[al]: ../../src/r0/modules/priors/lib/source_activation_law.py
[fit]: ../../src/r0/modules/priors/lib/joint_plan_calibration.py
[pipe]: ../../src/r0/modules/priors/lib/node_pipeline.py
[oldfit]: ../../src/r0/modules/priors/lib/source_causal_calibration.py
[ancestry]: ../../src/r0/modules/priors/lib/pipeline_ancestry.py
[moments]: ../../src/r0/modules/priors/lib/causal_moments.py
[fourier]: ../../src/r0/modules/priors/lib/fourier.py
[quad]: ../../src/r0/modules/priors/lib/quadratic.py
[diag]: ../../src/r0/modules/priors/lib/smooth_neural.py
[tree]: ../../src/r0/modules/priors/lib/structured.py
[signals]: ../../src/r0/modules/priors/lib/signals.py
[j]: ../../src/r0/modules/priors/lib/joint_sampling.py
[jr]: ../../src/r0/modules/priors/lib/joint_sampling_records.py
[numeric]: ../../src/r0/modules/priors/lib/parameter_preferences.py
[decode]: ../../src/r0/modules/priors/lib/joint_matrix_decoding.py
[limits]: ../../src/r0/modules/priors/lib/causal_limits.py
