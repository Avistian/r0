# Issue #28 exogenous laws and structural-noise checkpoint

This is partial implementation of the [accepted scope](specs/0028-causal-network-generators.md),
particularly decisions 10, 12, 13 and 18. Root/noise families, broader weights,
function families, typed variables, interventions and workbench/environment
acceptance remain required. This document does not close #28 or all paper options.

## Frozen laws, geometry and replay

`StochasticCausalNode` extends `ActivatedCausalNode` with an
`ExogenousDistribution` and a `NoiseResponse`. It uses causal-v2 recipes and the
same public decoding, generation and row-inspection interfaces. Nodes and their
shared decoder now live in `causal_nodes.py`; existing public imports remain.
Old nodes retain their fields, evaluation order and RNG keys. New-node evaluation
has an explicit separate equation, avoiding changes to legacy floating arithmetic.

`ExogenousDistribution.inputs` reuses #27's `InputDistribution`:

| Family | Standardized draw before location/scale |
|---|---|
| gaussian | standard Gaussian |
| uniform | uniform on `[-sqrt(3), sqrt(3)]` |
| student_t | Student-t with declared positive `df`, without variance rescaling |
| mixture | standard Gaussian plus independent equiprobable `+mixture_gap` or `-mixture_gap` |

`inputs.location + inputs.scale * draw` is applied channel-wise. The node's bias
and noise scale remain separate structural parameters. For example, location 0.5
and scale `1/sqrt(12)` express Uniform[0,1] before node scaling/activation. The
symmetric mixture chooses its sign independently per channel; it is not a shared
multivariate prototype mixture or native categorical node.

An optional positive `bound` is the existing conditional truncation law on the
standardized draw, before location/scale. Values outside the bound are rejected
within the declared 128-batch budget, using batches of at least 256 values. This
is explicit distributional truncation, not clipping an arbitrary generated task.
Budget exhaustion is an observable failure; no task is retried or substituted.
Gaussian, Student-t and mixture draws are not silently standardized by generated
rows. Per-channel streams are independent before explicit vector geometry.

Geometry is either independent or ball. Ball geometry requires default standard
Gaussian inputs: normalize a Gaussian direction and multiply by an independent
radius `U**(1/d)` in dimension `d`. This is uniform volume, not uniform radius or
surface sampling. A frozen optional square `factor` is then applied as `draw @ F`.
Factors may be singular or zero. They must match the node width, and their stored
entries count toward admission. Gaussian and uniform bases therefore support
correlated roots; ball factors produce ellipsoidal images. Transform overflow
fails explicitly. Factors are not asserted to be covariance matrices: for a
standard Gaussian base the covariance is `F.T @ F`.

Original-row inspection exposes `exogenous-i`, `noise-scale-i`, `innovation-i`
and the causal node. The exogenous value is the transformed distribution draw
before structural scaling. The row-wise scale and resulting innovation are
separate tensors. Changing the noise response with the same seed and task index
preserves the exogenous draw, while the edited recipe gets a different task
identity. This is a prerequisite for matched comparisons, not a complete
intervention interface. Row-count changes preserve the same stream prefixes;
inspection reconstructs selected original rows, not newly sampled rows.

Channel dependence follows nonzero factor columns and shared ball geometry.
Zero structural-noise scales/responses remove their random paths. This remains
structural potential dependence, not a symbolic cancellation solver or measured
predictive usefulness. Correlated channels can share an exogenous cause even
when there is no graph edge between them. Ball coordinates are dependent even
when their covariance is zero.

## Structural-noise equation

For concatenated parent channels `x`, let `a = f(x) @ W + bias`; roots use only bias.
The recorded response is

```text
s(a) = noise_scale * (offset + strength * abs(a)**power)
innovation = s(a) * exogenous
before_output: g(a + innovation)
after_output:  g(a) + innovation
```

Offset/strength are finite nonnegative scalars; power is finite and positive.
Defaults `(offset=1,strength=0,power=1)` recover constant structural scaling.
Each channel responds to its own pre-noise affine value; this does not depend on
batch statistics, labels or the output activation. A zero strength skips the
power calculation, avoiding an irrelevant overflowing intermediate. Nonfinite
scale/value failures are explicit. Existing Gaussian observation/target-adapter
noise is separate and is not replaced by this structural mechanism.

An after-output response can carry parent dependence even when `g` is constant.
The retained worked check uses `W=2`, bias 1, base scale 0.5, offset 1, strength 0.25
and power 2: each row's scale is `0.5*(1+0.25*abs(2*x+1)**2)`. Its trace verifies the
scale and innovation using the independently written scalar equation.

## Moments and normalization

Unnormalized heavy-tailed draws remain available regardless of population
variance. Recipes containing stochastic nodes must certify finite target second
moments before opting into fixed independent normalization. A moment descriptor
records bounds: positive absolute moments strictly below each numeric bound are
certified finite; `all` means all positive polynomial moments; zero means unknown.
This is a sufficient certificate, not an exact integrability solver or a sample
estimate. Failing certification rejects the requested normalization, rather than
silently changing the distribution. `normalize=False` remains available.

Untruncated Student-t starts with bound `df`; bounded and other current base laws
start with all moments. Finite linear sums take the minimum parent bound.
Positive powers divide it by their growth exponent. Signal-dependent noise uses
the minimum of the exogenous bound and affine bound divided by response power,
using independence of the node's exogenous stream from its ancestors. Bounded
activations reset to all moments. Stabilized absolute logs have all polynomial
moments when their input has some certified positive moment. Strict log near
zero, reciprocal powers without a safeguard, and exponential growth require
stronger information than polynomial-moment bounds, so are conservatively unknown
unless their input is known constant. This can decline some mathematically valid
normalizations; it does not mark the unnormalized family invalid or complete the
future analytic-compatibility catalog.

Constants are propagated separately from moment bounds. Valid constant targets
retain identity normalization and a zero-dispersion flag. An identity Student-t
with df 2 or squared Student-t with df 4 fails second-moment certification; a bounded
tanh descendant passes. Generation still checks actual numerical/domain validity.
No certificate authorizes replacing nonfinite numerical outputs with clipped data.

## Admission and sampling profile

Distribution/factor/response scalars count toward the parameter estimate. Stochastic
nodes add eight times their largest width to the temporary working-column estimate;
conditional truncation raises the admitted row floor to 256. These are conservative
array/storage estimates, not measured peak RSS or supported-maximum claims.
Legacy estimates remain unchanged. Actual resource qualification remains pending.

`profile="causal-exogenous-v1"` is a versioned r0 exploration profile, not a source
reproduction. It retains the three current graph families and Gaussian projections,
and independently chooses node widths from 1,2,4. It reuses the 26-choice activation
preference profile. Each root chooses ball geometry with probability 0.2; other
nodes use independent geometry. Independent geometry chooses the four input laws
uniformly, location uniform [-0.5,0.5], scale log-uniform [0.25,2]. Student-t df is
uniform over 0.75,1,1.5,2,3,5,10; mixture gap is log-uniform [0.25,4]. Independently,
truncation is selected with probability 0.25 and a bound uniform over 2,4,8.
Ball geometry uses its required default direction law.

Each node independently selects a Gaussian factor with probability 0.5; entries
are standard Gaussian divided by `sqrt(width)`. Otherwise the factor is identity.
Roots retain constant noise response. Other nodes choose strength uniformly from
0,0.1,0.5,1 and power from 0.5,1,2, with offset 1. All choices are frozen before task
generation, including failed attempts. There is no success-conditioned resampling.
The new construction name is `activation-exogenous-v1`; older profiles are unchanged.

## Source-to-implementation scope

The [pinned I2 point source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py)
distinguishes hypercube, uniform-volume ball and Gaussian points, Gaussian/uniform
covariance bases and further random-function transforms. Current authored laws
cover the base definitions and explicit frozen factors relevant to
`I2.root.01`–`.04`,`.06`–`.07`. The source's complete matrix sampler and random-function
transformed-root pipeline (`.05`) remain pending. Reading source did not copy it.

Gaussian and uniform family definitions also address parts of `P1.root.01`,
`P1.code.02`–`.04` and `P2.root.01`–`.02`; this is not an assertion of identical
published parameter/hyperprior or calibration laws. Heavy-tailed, multimodal and
bounded inputs cover parts of `R0.extension.17`–`.19`; the explicit response is an
r0 implementation of `.20`. Zipf/clipped-Zipf, native multinomial roots, shared
multivariate Gaussian/prototype mixtures, source-specific mean/scale hyperpriors,
the underspecified multivariate catalog and further structural noise/weight laws
remain required. In particular, the symmetric two-Gaussian mixture is not silently
substituted for P2's normal/uniform mixture or I2's multivariate mixture.

## Development and verification

| Behavior | RED | GREEN |
|---|---|---|
| Declared exogenous law and trace | `20f3883` | `7c931c8` |
| Frozen factor and cross-channel dependence | `c3b39b9` | `59af2d2` |
| Uniform-volume ball and prefixes | `a4d5f40` | `cd4fa65` |
| Signal-dependent noise and matched draws | `1a3d820` | `0d2ada7` |
| Heavy-tail normalization contract | `00dd3cf` | `9c8ea92` |
| Zero-response dependence | `0a580c4` | `8bf83e3` |
| Sampled save/reopen/replay | `ef46df5` | `a05838f` |
| Constant transformed targets | `eeda953` | `7c524cf` |
| Stochastic-buffer admission | `6e2bd38` | `b86ff16` |

Focused source at `b86ff16` passes 21 tests. The retained output is
[complete focused output](evidence/issue-28/legacy-research/0028-exogenous-focused-b86ff16.txt); command:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_exogenous.py \
  tests/explorer/test_causal_session.py --tb=short
```

Initial commit attempts hit assertion-formatting, Pyright set/norm annotations
and one long metadata string; these were corrected before their recorded commits.
They are not passing revision evidence. The first uniform-root RED test was
observed before implementation; its initial commit failed lint. The corrected
test-only commit retains the missing-decoder failure, followed by its GREEN.

Full checks, independent review and a bounded census are recorded below. Actual browser, Colab, GPU, complete source-family coverage and measured
production envelopes remain PENDING. Backend workflow checks do not substitute
for those requirements. The full implementation goal stays active.


## Author census

At frozen `f75c9cd`, the retained procedure attempted 64 tasks per topology family,
192 total, using seed 28 and task IDs 0 through 63. Each task has 16 rows, eight
context rows and four observed features; regression, binary and three-class
adapters cycle with task ID. Each recipe is captured in memory before generation; JSONL and saved groups are
written after each family finishes.

| Family | Attempted | Valid | Failed |
|---|---:|---:|---:|
| forward | 64 | 63 | 1 |
| layered | 64 | 64 | 0 |
| redirection | 64 | 64 | 0 |

All four input laws, both geometries and all sampled response strengths occur.
The [summary and artifact hashes](evidence/issue-28/legacy-research/0028-exogenous-census.json) count laws,
geometry and response strengths across every sampled node, including the failed
task. Ball directions count as Gaussian law entries. The stored `0` and `0.0`
strength keys both denote zero. Valid task flags overlap: disconnected 62,
inactive ReLU 19, saturation 23 and zero target dispersion 6. No valid flagged
task was removed. These counts are not measured functional utility or coverage
of the full paper inventory.

The [failed example](evidence/issue-28/legacy-research/0028-exogenous-failed-example.json), forward task 50,
raises `nonfinite square activation output`; its full frozen recipe remains in
the original JSONL and is reproducible from the recorded profile/seed/task ID.
No failure was retried or substituted. Raw records and successful saved groups
remain in `/tmp/r0-28-exogenous-census-f75c9cd/`; the published summary retains
hashes for those local artifacts. They are not embedded in the repository.

Reproduce at the recorded revision using a new destination:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/causal_topology_census.py \
  /tmp/r0-28-exogenous-census-reproduction --profile causal-exogenous-v1
```

Timing and peak process memory remain NOT_MEASURED. The census is author evidence,
separate from independent review. It does not establish browser, Colab, GPU or
supported-maximum behavior.

Independent review, retained failures and repaired check results are recorded in
[the verification record](issue-28-exogenous-verification.md).
