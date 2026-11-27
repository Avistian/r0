# Issue 28: independently fitted node-pipeline sampling laws

Read-only preflight: `/tmp/r0-28`, 2026-11-27,
`ae9b3b8b64c226d1532e7cb7939ab99b907fc2ad`. Research/proposal only; no upstream code
copied into r0 or numerical fixture executed. Examples are deductions.

Authority: [accepted specification, decisions 10–13 and 18–19][spec]. This extends
the [wrapper audit][wrappers] and preserves the three [authored converter variants][pipeline].
All upstream links pin `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`. Line numbers
count actual raw-file blank lines, not browser-normalized text. Retained local
`/tmp/node-pipeline-sampling-research/_node_function.py` has 73 lines, SHA256
`3b2345f23d0013442f7390c907454dfd4c6f4b43836a1bd210423242a34964c6`.
Key raw anchors: Standardize 176–182, L2 185–190, Kumaraswamy 205–217 in activation;
converter numerical branches 46–73; node slice assignment 61–63. [A][A] [C][C] [N][N]

## The critical source/adaptation distinction

Standardize/L2/converter fits freeze on first use; channel weights, noise scale
and whole-node importance are instead requested **on every node transform**.
The noise tensor is fresh. Global mode may reuse named scalar/array draws, but
channel Gaussian perturbations/permutations or uniform weights still redraw.
[node20–73][N] [first-call172–197][B] [weights][W]

Default `ensure_iid=False` fits on the first emitted `n_train+n_test` rows.
`ensure_iid=True` discards a graph pass of that size before emission. Combining
that loop with transform-time draws shows that importance/noise parameters can
change between fit and emission (deduction, not executed comparison). [dataset][D]
[defaults][Q] [node][N]

Proposed r0 adaptation: freeze these mechanism parameters once before independent
ancestral calibration, varying only row exogenous draws afterward. Explicit
calibration size/seed must not depend on emitted rows or labels. Keep source
first-emitted-batch and per-pass-redraw references separately named. [decision12][spec]

## Root input is distinct from later innovation

Source roots call `RandomPoints.sample` **before** Standardize/importance/L2 and
converters; optional Gaussian innovation is added **after** converters. RandomPoints
selects `random_base_points` among uniform/circle/Gaussian/covariance points and
passes them through a newly constructed RandomFunction. Its covariance path also
uses the shared RandomWeights scope. This per-call law/function construction must
be frozen explicitly for r0's fixed-mechanism adaptation. [node43–71][N]
[points7–23][P]

Current authored after-output roots begin with bias before the pipeline and add
innovation afterward; this is not the full source RandomPoints root. Add a
separate root-input law upstream of normalization/conversion, with independent
post-converter innovation and separate stream/binding identities. Never relabel
one existing innovation as both. An upstream random node is a distinct topology adaptation.
[current evaluator80–106](../../src/r0/modules/priors/lib/causal_evaluation.py)
[authored contract][pipeline] [source order][N]

## Exact arithmetic and sampling requests

For an m-by-d calibration matrix C, `Standardize` keeps a 1-by-d mean and inverse
scale. Its population standard deviation uses **correction=0, axis 0**:
`mu_j = mean_i C_ij`, `s_j = sqrt(mean_i (C_ij-mu_j)^2)`,
`S_j(x) = (x_j-mu_j)/(s_j+1e-4)`. The epsilon is outside the square root.
[Standardize][A]

Deduction: constant-channel inverse scale is 10000, mapping the fitted constant
to zero but future x to `(x-c)*10000`; a one-row fit has the same convention.
These methods do not validate empty/nonfinite/overflowing inputs. Proposed r0
policy: nonempty finite fit input, stable equivalent arithmetic, explicit genuine
numerical failure. [Standardize176–182][A] [spec][spec]

When feature importance is enabled, the node standardizes **all channels**,
including latent and reserved readout channels, requests one positive vector of
shape `(1,d)`, squeezes it to `(d,)`, then multiplies each standardized column by
`v_j = w_j/sqrt(mean_k w_k^2)`. This second RMS normalization has **no epsilon**.
Turning feature importance off skips both this standardization and the weights;
L2 is independently switchable. [Node lines 48–54][N]

The configured weight family is a fixed config choice, not a named categorical
draw: default `simple`, alternatively `uniform`; other strings raise. [W][W] [Q][Q]

| Family | Exact law and shape | Source |
| --- | --- | --- |
| `simple` | For each batch row b, draw decay `p_b` using `simple_random_weights_decay_rate`, logarithmic bounds `[0.1/log(1+d),6]`, boundary mass enabled, size `n_batch`; draw `sigma_b` with `simple_random_weights_std_scale`, logarithmic bounds `[0.0001,10]`, size `n_batch`. Build logits `-p_b log(k)+sigma_b Z_bk`, k=1..d, independent standard-normal Z; independently permute each row; softmax along channels; multiply by `sqrt(d)/L2_norm` along channels. Node calls use `n_batch=1`. | [W][W] |
| `uniform` | Independent Torch uniform entries in shape `(n_batch,d)`; this method itself does not normalize them. The node subsequently applies its RMS normalization. | [W][W], [Nweights][N] |

Deductions: finite simple logits yield positive weights already having RMS one;
node renormalization is algebraically redundant, not necessarily bitwise so.
Uniform weights can contain zeros; neither family samples signs. A nonzero
singleton normalizes to 1; all-zero weights make division undefined, with no
fallback law. Softmax underflow can produce represented zeros. Reject zero width,
all-zero/nonfinite vectors or unsupported families explicitly; preserve authored
signed factors as extensions. Uniform omits an explicit device argument; GPU
behavior is NOT_ESTABLISHED. [weights][W] [node48–52][N] [authored][pipeline]

L2 then fits one whole-node scalar
`f = 1/(sqrt(mean_i sum_j C_ij^2)+1e-8)` and applies `f*x`; it neither centers
nor divides by d. Here C is **after** the standardization/channel-weight stage.
Deduction: zero C gives f=1e8, retained for future inputs. [L2][A]

After converters/readout capture, optional structural noise is
`sigma*Z` with Z shaped like the full node matrix and standard-normal entries;
`gaussian_noise_sigma` has logarithmic bounds `[1e-5,1]`. Finally all propagated
channels are multiplied by the positive scalar `node_importance`, logarithmic
bounds `[0.1,10]`. Both requests use the ordinary default numerical mode, with
no boundary-mass flag and scalar shape. Defaults: noise disabled, node importance
enabled. Deduction: conditional on these scalar draws, final added innovation
has standard deviation `node_importance*sigma`, not sigma. No source
signal-dependent noise appears at this placement. [Noise/order][N] [Q][Q]

## Names, sharing, and collisions that the sampler must not hide

For a numerical name in source meta mode, one location U uniform on `[0,1)` and
one concentration H log-uniform over `[0.1,10000]` are cached via the suffixes
`__meta_beta-loc` and `__meta_beta-sum`. Each request draws
`V ~ Beta(U*H,(1-U)*H)` with its requested shape. Local mode instead draws fresh
uniform V. Boundary mass, when enabled, maps V to `clip(1.2*V-0.1,0,1)` before
linear or logarithmic interpolation. Global mode caches a realized result by
name; bounds/shape are not checked against later requests. Consequently these
default log requests are **Beta-hyperprior log mappings**, not simply log-uniform
draws. Endpoint/invalid-Beta numerical robustness is UNKNOWN here. [Bnum][B]

Categorical local mode chooses a uniform index; global caches an index by name.
Meta mode constructs polynomial positive preference weights using a **fresh
`Context()`**, normalizes them to sum one, and chooses an index with ambient
NumPy randomness. Corrected mode caches under `name__meta_weights`; historical
default checks `name` but stores under the suffixed key, normally regenerating
weights each call. Its fresh-context weight hyperpriors do **not** share the
outer node/matrix weight scope. Descendant component contexts otherwise retain
the same sampler. [Bcat][B] [Bcontext][B] [Q][Q]

| Names | Sharing boundary (same outer sampler unless stated) |
| --- | --- |
| `simple_random_weights_decay_rate`, `simple_random_weights_std_scale` | Node and matrix positive vectors overlap; nested categorical vectors use fresh scopes. Width/batch contracts differ. [weights][W] [matrices][M] [sampler][B] |
| `act_type`, `act_type_plain`, `acts_torch` | Standard and plain top selectors differ; both share fixed-catalog `acts_torch`. Converter `act` is standard; matrix activation is plain. [activation][A] [converter][C] [matrix][M] |
| `relu_random_power_exponent`, `random_power_exponent` | Same names in both selectors, log `[0.1,10]`; plain additionally uses `neg_random_power_exponent` (positive draw then negated) and `rand_int_power_exponent` (floor/high-exclusive 2..6). [activation144–173][A] [sampler77–92][B] |
| `rr_scale` | Shared scalar log `[1,10]` in standard activations. Each channel shift separately selects a calibration row and negates its standardized value; indices are unnamed Torch draws. [activation193–202][A] |
| `num_conv_mode`, `kumaraswamy_a/b` | Converter selector and two separate shape names; no matrix-selector collision. Activation children share names above. [converter][C] [activation205–217][A] |
| `node_n_latent_features` | Log/floor 1..32 high-exclusive: 1..31 latent channels plus reserved converter widths. [node32–34][N] [sampler][B] |
| `gaussian_noise_sigma`, `node_importance` | Distinct scalar names shared across nodes/passes, separate from matrix/activation parameters. [node66–73][N] [sampler][B] |

r0 already declares separate numerical/categorical namespaces, injected RNG,
shared/refresh categorical policy and global quantile remapping across matrix
widths; its activation sampler owns separate scopes. Concatenating independently
sampled matrix/activation/node records would not establish shared outer
preferences: inject compatible scopes or name the separation adaptation.
Source global caching does not validate changed bounds/shapes; retain explicit
quantile remapping or reject incompatibility, never silently reinterpret a cached
array. [r0 matrices][shared] [r0 activation][activations] [source25–75][B]

## Numerical converter selection and target restriction

The ordered catalog is `id`, then `kumar` if allowed, then `act` if allowed.
`num_conv_mode` is sampled even for the singleton identity catalog. Defaults give
`[id,kumar]`, historical conversion, and target warping allowed. With target
restriction enabled, **only numerical converters with feature group exactly `y`**
receive the disallow flag: their catalog becomes `[id]`. This does not disable
upstream mechanisms, node standardization, L2, or the propagated state's later
noise/importance. Categorical conversion is a different branch. [C][C] [Nfit][N]
[Q][Q]

For `kumar`, the fit stores channel extrema and scalar a,b; their independently
named meta preferences are shared across converters but individual realizations
need not match. Each numerical converter has one channel; a general multi-channel
Kumaraswamy transformer shares its scalar a,b across its channels. The normalized
argument is clamped to `[0,1]` with denominator `(max-min+1e-30)`, followed by
`1-(1-u^a)^b`. Identity never calls this transform, so it does not draw a/b merely
because a transformer object exists. [C][C] [K][A]

For `act`, the converter uses the full **standard** activation pipeline: input
standardization, sampled rescale/shift, selected primitive, output
standardization. It does not use the plain matrix selector. Both nonidentity
converter types obey the same corrected/historical state/readout split.
[C][C] [Astandard][A]

Preserve `historical-tuple` (transformed state/copied raw readout),
`historical-graph` (transformed state/readout via the pinned alias consequence)
and `readout-only` (raw state/transformed readout), captured before noise/importance.
Both `kumar` and `act` share this branch: activation ends with out-of-place
Standardize, so the same tuple/graph distinction applies. Earlier alias-fixture
qualification remains unchanged; no new execution occurred here.
[tuple63–71][C] [assignment61–63][N] [Standardize181–182][A] [audit][wrappers]

**Deduced source corner case:** restricted target and unrestricted feature
converters can request different ordered catalogs under the same `num_conv_mode`
name. Corrected shared meta reuse can supply a probability vector of the wrong
length; global reuse can supply an index outside the singleton catalog. The
historical refresh path normally avoids stale weights. This is a source-level
inference, NOT_EXECUTED here. Do not reproduce unchecked cache failure or silently
renormalize shared labels. Proposed explicit r0 policy: forced-identity target
decision with a retained restriction reason and no categorical draw; ordinary
converters share one stable ordered catalog. This changes source draw consumption
and must be versioned as an adaptation. Alternatively use explicit catalog-keyed
scopes, also labeled as an adaptation. [C][C] [Bcat][B] [Nfit][N]

## Proposed sampling and binding order for r0

Proposal under [decisions 10–13,18–19][spec]; not implemented evidence.

1. **Declare/version/admit.** Retain source commit, root/mechanism law, all switches,
   preference mode/policy, global quantile policy, independent calibration identity
   and size, and converter semantics. Admit dimensions, complete parameter costs
   and buffers before allocation/drawing. Order nodes topologically and readouts
   deterministically; reserve converter/latent widths before matrix dimensions.
   Source extracts every feature through a converter; the authored surface permits
   final-state readouts. Omitting even an identity converter can therefore change
   noise/importance exposure. [node][N] [authored contract][pipeline]
2. **Sample/freeze/bind.** Plan compatible outer scopes jointly with matrices and
   activations, preserving fresh nested categorical scopes. Retain each decision's
   scope, draw, intended field and first-use order. Save vector decay/log-noise,
   innovations, permutations and normalized channel factors; retain uniform
   realizations. Freeze root-input law, sigma and node importance before fitting;
   freeze converter a/b or activation primitive/scale before their fit. Do not
   draw unused parameters. Transactional admission/rejection and this draw-order
   change are explicit r0 adaptations, not source RNG parity.
3. **Fit ancestors.** Draw an independent root-input sample; non-roots consume
   finalized parent **state**. Fit mechanism-local stages in place, then node
   Standardize, apply frozen channel factors, fit L2 on that result, and fit each
   converter on its post-L2 slice. Capture readouts by value; update state under
   the declared variant; add independent calibration innovation with frozen sigma;
   multiply by frozen importance. Pass that final state to children. Never use
   parent readout/pre-noise state as a substitute. Source propagates state; fixed
   independent fitting is the accepted adaptation. [node][N] [graph][G] [spec][spec]
4. **Gate moments in place.** Require certified finite second moments at each
   Standardize/L2 input and at standard-activation output before its final
   Standardize. Finite samples alone do not certify population moments. Powers,
   exponentials/products and dependent parents require conservative certificates;
   unknown remains uncertified. Bounded conversion can establish later moments,
   not justify an earlier normalization. Finite frozen weights, positive bounded
   importance and Gaussian innovation preserve second moments. Track state and
   readout separately; target normalization uses target readout. These are
   mathematical deductions/conservative proposals, not an exact integrability
   solver. [decision 13][spec] [existing moments][moments] [pipeline][pipeline]
5. **Emit/replay/account.** Save fit constants, stage order, widths, stream identity,
   calibration rows/input digests and complete bindings. Use disjoint emitted-row
   streams; query values/counts, chunks and labels cannot refit. Decode with
   missing/duplicate/orphan/reordered-binding and scope checks. Keep all attempted
   draws, explicit failures and valid weak/disconnected/saturated/constant cases.
   [accepted policy][spec]

Proposed checks (NOT_RUN): independent Standardize/RMS oracles; shared/nested
scopes; root/innovation separation; sigma/importance order; restricted catalogs;
three converter variants; downstream calibration on final parent state;
chunk/query invariance; bindings and complete attempted-seed census.

## Inventory mapping and unresolved boundaries

Existing records remain unchanged/open as applicable. [records 06][I6] [07][I7]

| Records | Required sampling/binding coverage |
| --- | --- |
| `I2.node.01`–`.05` | Reserved channels, positive channel importance, actual-input whole-node L2, positive node importance, post-readout Gaussian noise. “RMS” in `.03` must not imply division by channel count. |
| `I2.matrix.15`–`.17` | Reuse positive-vector laws and outer hyperpriors; arbitrary authored signed factors are not source sampling. |
| `I2.activation.18`–`.24` | Selector/primitive/power sharing, rescale/shifts, input/output Standardize and L2 at actual placements. |
| `I2.converter.01`–`.03` | Identity, historical Kumaraswamy **and activation** conversion. Preserve tuple-intent reconstruction; propose separate `.02.graph` and `.03.graph` records for pinned alias behavior. |
| `I2.measurement.01`–`.03` | Corrected readout Kumaraswamy/activation and target-warp restriction; preserve #30 interface boundary. |
| `I2.root.*`, `I2.category.01`–`.20`, `I2.node.06`–`.13` | Root-point/function sampling, categorical converters and parent aggregation remain separately tracked; this note does not close them. |

UNKNOWN: checkpoint prior, failure rates, GPU behavior and RNG-order equivalence.
No throughput, browser/Colab, scientific or full-issue acceptance evidence.


[spec]: ../specs/0028-causal-network-generators.md
[wrappers]: 0028-wrapper-source-laws.md
[pipeline]: ../issue-28-node-pipelines.md
[shared]: ../issue-28-shared-matrix-preferences.md
[activations]: ../issue-28-source-activations.md
[moments]: ../issue-28-exogenous-laws.md#moments-and-normalization
[I6]: 0028-source-options-06.json
[I7]: 0028-source-options-07.json
[N]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_node_function.py#L20-L73
[W]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py#L7-L56
[Q]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py#L10-L41
[C]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_converter.py#L46-L73
[D]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_dataset.py#L59-L67
[G]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_graph_function.py#L32-L42
[A]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L15-L217
[B]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py#L25-L197
[M]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py#L59-L97
[P]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py#L7-L23
