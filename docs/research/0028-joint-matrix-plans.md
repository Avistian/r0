# Issue 28: joint matrix, weight and point plans

Design only, 2026-11-27. Preflight confirmed `/tmp/r0-28`, HEAD
`800af3e4cd0888fcea5232df452490b65a7a3f0f`, parent-owned edits and live slots.
This note is the only authored output; no tests, upstream execution, downloads,
cloud or dependency changes. Proposed checks: **NOT_RUN**. Full #28 scope stays
broad; five-family integration is component work, not issue acceptance. [spec]

## Source identity

Cached primary files: `/tmp/node-pipeline-sampling-research`. [Retained metadata][laws]
associates them with TabICL `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`.
Its `_node_function.py` checksum matches current bytes. [Prior review][review]
records live matrix/points inspection at that commit. No upstream checksums for
these five files were found in scoped metadata: provenance is retained, not a
fresh remote comparison. Current SHA256:

| File | SHA256 |
| --- | --- |
| `_matrix.py` | `cfb1d766bbe0db14ae8987b8570e5f9392904ed32f63d1af206811882aaf4c41` |
| `_weights.py` | `5e8ca75184a84ce34337a8b90dbc0ce199395fe03ba51c95c8b622438b85d2d8` |
| `_points.py` | `04dcbd7664004f6acc8453ca866caba0879d87d4ecbcdad08dd79f9ca2ff7a04` |
| `_base.py` | `065ed93d700fb125222ee60b68656eb419a39c630645d1b8f44fe9d6baf718d7` |
| `_activation.py` | `11b8bca1e34d11725bbe2a19a229059840fe14ca0f604cbe864c07987bb9f51a` |

Extends [the integration design][integration].

## Exact requests and sharing

One `JointSamplingContext` must own matrix/mechanism/converter/node/root requests;
sharing only an RNG is insufficient. Derived source contexts share their sampler.
Meta shares Beta location/concentration; local is independent; global blindly
caches first realization. Categorical shared fixes the historical refresh cache
bug. Its nested `SimpleRandomWeights(Context())` remains a separate scope. [B]
[joint] [records]

| Request | Exact contract and consumers |
| --- | --- |
| `random_matrix_type` | Ordered default catalog Gaussian, weights, singular-values, kernel, activation; gaussian configuration uses singleton Gaussian. Bind to existing family/version mapping without reordering. [M] |
| `simple_random_matrix_type` | Ordered Gaussian, weights, singular-values, kernel, always; excludes activation. Source ignores gaussian-only configuration here. Distinct name/catalog from outer selector. [M] |
| `simple_random_weights_decay_rate` | Log `[0.1/log(1+w), 6]`, boundary mass enabled, source shape `(b,)` for b vectors of width w. Matrix rows, factor diagonal, mixture masses, mixture factor columns, covariance roots and node importance vectors share this name. [W] [P] |
| `simple_random_weights_std_scale` | Log `[0.0001,10]`, no boundary mass, same source shape `(b,)`; same consumers. [W] |
| `random_gmp_n_centers` | Log randint low=1, high=16: `clip(floor(exp(log(16)*u)),1,15)`, after mode sampling; no boundary mass. Scalar; every directly sampled Gaussian-mixture cloud. [P] [B] |
| `random_kernel_matrix_dist_factor` | Scalar log `[0.1,10]`, no boundary mass; requested after point generation. [M] |
| `random_base_points` | Ordered uniform, circle, Gaussian, covariance; root RandomPoints only, followed by a RandomFunction. No kernel request. [P] |
| `random_covariance_base_points` | Ordered uniform, Gaussian; covariance points only, after factor/weight sampling. No Gaussian-mixture request. [P] |
| `act_type_plain` | Eight distinct fixed slots, relu-power, signed-power, negative-power, integer-power. Transformed matrices use the parent's plain activation planner; no rescale/standardization requests. [A] |
| `acts_torch`, power names | Shared with reached mechanism/converter consumers. Reuse source activation request factories. Positive `neg_random_power_exponent` magnitude is applied negatively by the primitive; `rand_int_power_exponent` is integer 2..5. [A] [activation] |

Weights use task configuration `simple` (r0 polynomial) or `uniform`, with no
family selector. Uniform has no numerical events and keeps U[0,1) coordinates.
Polynomial logits `-decay*log(j)+scale*z_j` are permuted, softmaxed and rescaled
to RMS one; innovations/permutations stay fresh. Catalogs are task-wide immutable
ordered tuples, including refresh. Gaussian-only does not complete five-family
coverage. [W] [B] [vectors]

Global `scalar-quantile-v1` scalarizes each row request and remaps a shared unit
into width-specific bounds: it ties row parameters, unlike source arrays. Name
`scalarized-weight-batches-v1`: all row decays then all row scales, not alternating
pairs. Meta/local scalarization preserves conditional laws, not RNG consumption.
A batched strict-contract variant must reject incompatible shape/bounds, never
split source names silently. [W] [B] [records]

## Branch semantics and complete order

Let `R(X)=X/(1e-6+row_L2(X))`. Every RandomMatrix and SimpleRandomMatrix wrapper
adds fresh `1e-6*G` and applies R, even Gaussian-only. The source five branches,
with b batch members, n rows and m columns, are: [M]

| Family / existing raw record | Branch sequence before outer jitter/R |
| --- | --- |
| Gaussian / `GaussianWeights` | Fresh Gaussian `(b,n,m)`. |
| weights / `ImportanceWeights` | RandomWeights `(b*n,m)`; reshape; fresh Gaussian `(b,n,m)` multiply; inner R. There are two normalizations including the wrapper. |
| singular-values / `ImportanceFactorizedWeights` | k=min(n,m); Gaussian U `(b,n,k)`; RandomWeights `(b,k)`; Gaussian V `(b,k,m)`; `(U*diag)@V`. Factors are Gaussian, not orthogonal/SVD factors. |
| kernel / `LaplaceKernelWeights` | One Gaussian-mixture cloud of `b*(n+m)` points in d=3; reshape/split; Euclidean distance; named log distance factor; `exp(-factor*distance)*sign(G)`. No squared-distance/RBF substitution. |
| activation / `TransformedWeights` | Entire SimpleRandomMatrix child including its jitter/R; flatten each member to `(b,n*m)`; plain activation; fresh `1e-3*G`; outer jitter/R. Rank/softmax/argsort operate over all n*m entries within each member. |

**Transformed order:** constructor creates no activation decisions; Python
samples the child argument before lazy activation `__call__`. Named order:
outer family, complete child subtree, plain selector, primitive/exponent.
`MatrixPlanner.plan` instead samples activation first. [M] [A] [B] [planner]

Gaussian-mixture point order (q points, dimension d): c; weight batch `(1,c)`;
assignments with replacement; Gaussian centers `(c,d)`; Gaussian factors `(c,d,d)`;
weight batch `(c,d)` multiplying factor columns; scalar normal a, scalar normal h,
c normals z; multipliers `exp(a+h*z_i)`; q Gaussian innovations. Point = center +
scaled factor @ innovation. Signed h is not abs(h). Assignments are fresh
multinomial draws, not outer categorical preferences. All-zero mixture mass
rejects without retry; zero individual mass cannot receive assignments. [P] [points]

Kernel directly calls mixture points: no point-law, covariance or root-function
selector. Preserve paper covariance kernels as a distinct adaptation, as [prior
review][review] requires. Covariance points draw Gaussian d*d factor, `(1,d)`
weights, then `random_covariance_base_points` and innovations; zero center, no
log multiplier/count/mixture-mass draw. [P] [M]

Full RandomPoints requires a base plan plus RandomFunction hook: base selector,
base realization/subdecisions, function lazy fit. Circle means uniform unit-ball
points: Gaussian direction times U^(1/d), not sphere-only. These root/function
options remain required integration work beyond kernel mixtures. [P] [spec]

## Minimal frozen plans and ownership

New records; preserve every legacy recipe, sampler and decoder contract. Exact
types and tagged branch invariants avoid irrelevant optional fields. [spec]

```text
JointMatrixPrior(version, source_commit, top_catalog, simple_catalog,
                 vector_family, scalarization, point_variant)
JointWeightPlan(version, operation, width, batch, family, decays, scales)
JointPointPlan(version, operation, count, dimension, law,
               components?, mixture_weights?, factor_weights?, base_law?)
JointMatrixPlan(version, operation, rows, columns, family,
                weights? | diagonal? | points+distance? | child+plain_activation?)
JointMatrixRealization(version, plan, construction, realization_manifest)
```

One weight plan owns its batch: roles `decay:0` ... then `scale:0` ... preserve
source parameter order without reopening row operations. Uniform plans have no
named events but require unique operation identity and realization ownership.
Do not invent `forced_identity` for arrays or geometry. Current context marks
completion only when descendant events exist; an aggregate plan registry must
also reject duplicate event-free operations. [joint] [records]

Each sampler opens a transaction, freezes metadata and validates before commit.
Transformed child precedes parent's plain activation planner. Derive complete
expected events from branch/dimensions and compare exact ordered paths, roles,
kinds and draw references. Binding helpers alone do not prove ownership.
Validate request name/mode/shape/bounds/law and value against each field, plus
record version, vector geometry, point geometry, distance, selector and catalogs.
Every event has one reachable leaf owner; reject orphan/duplicate operations,
recursive transformed children and standard scales inside matrices. Parent containment is not double ownership. Recompute destinations on decode. [draft]
[bindings] [records] [contracts]

Named events exclude raw randomness. A separate ordered realization manifest
binds every Gaussian/uniform/permutation/multinomial draw, jitter and deterministic
stage to role, stream, shape and stored field; validate complete ownership.
Assignments never use outer categorical preferences. [materializer] [points]

Plan all named decisions in source order, admit aggregate, then realize: this
explicit two-phase RNG adaptation differs from source interleaving. Old
materialization also draws importance Gaussian before vectors and factorized
U,V before diagonal. Version that order or preferably add a small source-order
adapter using existing arithmetic; never change legacy sampling. [B] [materializer]

Context has no public realization RNG: use a narrow staged callback or separate
identified realization stream cloned/committed with the outer transaction.
Never expose `_rng` or cache samplers. Admission, realization, ancestral fit,
freeze and validation all precede commit. [joint] [integration]

## Reuse and mismatches

Reuse the frozen vector/matrix/point record arithmetic listed in the branch table:
float64 stability is an adaptation of source float32. `MatrixPlan` itself is an
unvalidated DTO without semantic bindings. New plans can lower to it after
validation if the legacy realization order is declared. Supply every polynomial
parameter, mixture law/count, component/factor vector and distance; omissions
otherwise trigger independent draws. [vectors] [points] [materializer]

Current records represent b=1. Source b>1 kernels share one cloud's mixture and
centers; b independent clouds change the law. A batch container must share its
point subplan; activation flattening remains per member. [M] [P] [transforms]

Preserve old sampler ownership and names `kernel_point_law`,
`kernel_covariance_base`, `kernel_mixture_components`, `kernel_distance_scale`.
New context plans use source names. Existing matrix validators require all draws
to be matrix draws: reuse field-mapping ideas, not whole-ledger checks. [planner]
[prior] [contracts] [preferences]

## Raw decoder and aggregate admission

Before conversion/materialization scan raw structure under caller-owned limits:
reject bool dimensions/indices, non-arrays, unknown fields/versions, ragged shapes,
oversized lists/paths/names and recursive children. Count all leaves/containers
with checked integer arithmetic. Existing raw preference counters help; nested
weight/point decoders cannot replace aggregate preflight. [records] [decoder]

Count preferences/nested categorical owners once, plans/bindings/manifests,
realized leaves, duplicate node weights/point geometry, scalars and fit constants.
Peak work includes factors, points, distance/sign/transform buffers, innovations
and ancestors. Admit metadata incrementally and exact component geometry before
allocations. Individual caps do not prove aggregate safety; parameter counts are
not measured bytes/RSS. [integration] [materializer] [limits]

Record-only formulas, V(w)=w uniform or 2w+2 polynomial with permutation,
q=n+m, k=min(n,m), c=mixture count, d=3:

- Gaussian normalized: `2nm+2`.
- Importance normalized: `2nm+n*V(m)+3`.
- Factorized normalized: `nk+km+V(k)+nm+2`.
- Kernel: cloud `cd+cd²+c*V(d)+V(c)+q+qd+2+c`, plus retained points `qd`,
  signs `nm`, distance `1`, wrapper `nm+2`.
- Transformed: normalized-child count + primitive parameter count + inner jitter
  `nm+1` + wrapper `nm+2`.

Add ledger, plans and stored node matrices separately. [materializer] [points] [matrices] [transforms]

## Independent literal oracles and rejection cases

Checks: **NOT_RUN**. Use literal independent arithmetic, not implementation
helpers as the expected-value oracle. [spec]

| Case | Discriminating literal expectation |
| --- | --- |
| Weight arithmetic | w=2, decay=1, zero Gaussian logits, permutation `(1,0)` gives weights `(sqrt(2/5),2*sqrt(2/5))`, squared sum 2. Uniform `(0.25,0.75)` stays unchanged. [W] |
| Shared global remap | Unit u=0.5 gives width-w decay `sqrt(6*0.1/log(1+w))`, scale `sqrt(0.0001*10)`; equal unit/preference identity across node, matrix, mixture and covariance consumers, different width decay values. At u=0/1 decay is exactly its endpoint. [B] [W] |
| Center bound | Units 0,0.5,1 produce 1,4,15 under log integer conversion. Last fixture checks high-exclusive clamp against floating endpoint rounding. [B] [P] |
| Factor orientation | U=`[[1,2],[3,4]]`, diag=`[0.5,1]`, V=`[[1,0],[0,1]]` yields `[[0.5,2],[1.5,4]]` before wrapper. Distinguishes weighting U rows or V columns. [M] |
| Point factor columns | center `(1,2,3)`, raw factor identity, uniform weights `(0.5,0.25,0.75)`, a=h=z=0, innovation `(2,4,-4)` gives `(2,3,0)`. Non-diagonal factors should additionally distinguish transpose and column scaling. [P] |
| Laplace distance | Points `(0,0,0)` and `(3,4,0)`, distance scale 0.1, sign -1 give `-exp(-0.5)` before jitter/R. [M] |
| Two normalizations | Importance raw row `(3,4)` gives `(3,4)/(5+1e-6)`, then wrapper adds its independent recorded jitter and normalizes again. Transformed fixture must retain separate child 1e-6, activation 1e-3 and outer 1e-6 draws. [M] |
| Flattening | For a 2x2 child, softmax denominator includes all four entries; rank/argsort fixtures use distinct entries spanning both rows. Batched fixture normalizes separately per batch member. [M] [A] |
| Exact ownership | Force all five families and all four simple children. Expected literal events: family; child subtree; plain activation. Kernel: family; count; mixture decay/scale; factor decays then scales; distance. Uniform vectors omit numerical events. [M] [W] [P] |

Also verify all modes/policies and nested-scope separation. Reject wrong catalog,
width/mode, recursive transformed child, extra kernel point-law selector,
missing/orphan/duplicate/reordered events, stale values, changed stage constants,
mismatched point geometry and zero-mass assignments. Raw over-budget payloads
must fail before materialization; test aggregate overflows despite individually
admissible matrices/samplers. [records] [contracts]

Inject failures after selector, child, count, weights, admission, realization,
jitter, fit and final freeze. RNG, caches, ledger, completed paths and separate
realization stream must equal pre-call state. Retain attempted identity/reason
separately; no retries. Preserve valid weak/zero/saturated draws. Check legacy
recipe bytes, seeded outputs and malformed rejection behavior. Full graph,
roots/functions, workbench, Colab/GPU and measured resources remain pending.
[joint] [spec]

[spec]: ../specs/0028-causal-network-generators.md
[laws]: 0028-node-pipeline-sampling-laws.md
[review]: 0028-matrices-source-review.md
[integration]: 0028-pipeline-sampling-integration.md
[M]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py#L9-L97
[W]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py#L7-L56
[P]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py#L7-L70
[B]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py#L25-L197
[A]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L82-L173
[joint]: ../../src/r0/modules/priors/lib/joint_sampling.py
[records]: ../../src/r0/modules/priors/lib/joint_sampling_records.py
[activation]: ../../src/r0/modules/priors/lib/source_activation_law.py
[draft]: ../../src/r0/modules/priors/lib/joint_activation_plans.py
[bindings]: ../../src/r0/modules/priors/lib/joint_plan_bindings.py
[planner]: ../../src/r0/modules/priors/lib/preferred_matrix_sampling.py
[prior]: ../../src/r0/modules/priors/lib/matrix_preference_prior.py
[materializer]: ../../src/r0/modules/priors/lib/causal_matrix_sampling.py
[contracts]: ../../src/r0/modules/priors/lib/preferred_contract.py
[preferences]: ../../src/r0/modules/priors/lib/matrix_preferences.py
[vectors]: ../../src/r0/modules/priors/lib/weight_vectors.py
[points]: ../../src/r0/modules/priors/lib/kernel_points.py
[matrices]: ../../src/r0/modules/priors/lib/weight_matrices.py
[transforms]: ../../src/r0/modules/priors/lib/transformed_weights.py
[decoder]: ../../src/r0/modules/priors/lib/weight_constructions.py
[limits]: ../../src/r0/modules/priors/lib/causal_limits.py
