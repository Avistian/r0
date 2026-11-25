# Replayable sampling profiles

`sample_recipe(family, task=..., seed=..., task_id=..., profile=...)` chooses
mechanism settings and returns an ordinary recipe for `generate_task`. It does
not generate data, retry failures, resize tasks, choose between families or filter
outputs. The caller selects the family and the `LinearRecipe` task controls:
rows, features, active count, context size, task type, class count, noise,
separation and class offsets are preserved. Vary those controls explicitly when
measuring coverage. Production mixture weights and training-stream policy remain #30.

```python
from dataclasses import asdict
from r0.modules.priors import LinearRecipe, generate_task, sample_recipe

controls = LinearRecipe(
    rows=512, context_rows=384, features=64, signal_features=32, active_features=16
)
recipe = sample_recipe("polynomial", task=controls, seed=27, task_id=7)
record = {
    "profile": "broad-v1",
    "family": "polynomial",
    "task": asdict(controls),
    "seed": 27,
    "task_id": 7,
    "recipe": asdict(recipe),
}
# Persist record before generation so a numerical failure retains the exact request.
task = generate_task(recipe, seed=27, task_id=7)
```

A profile is a versioned distribution over settings, not a bulk-training
eligibility claim. Save the selected recipe and the profile request together.
The selected recipe contains all concrete settings; generated task cards separately
retain the sampled mechanism and execution environment. JSON round-trips through
`recipe_from_dict` preserve the selected recipe. Sampling is independent of call
order: its explicit NumPy generator is keyed by profile, family, complete task
controls, root seed and task identifier. Changing any of these may change settings.
Changing a profile's distributions requires a new profile name/version.

## Broad profile: `broad-v1`

All listed discrete choices are uniform. Integer intervals include both endpoints.
`log-uniform(a,b)` means exponentiating a uniform draw between `log(a)` and
`log(b)`; continuous intervals use NumPy's half-open uniform sampling semantics.
These ranges are explicit exploration choices, not attributed to a paper.

| Family | Sampled settings |
| --- | --- |
| `linear` | Preserves the task controls; the existing generator samples active columns and normal coefficients. Standard Gaussian inputs only. |
| `grouped` | Group size chosen from all common divisors of signal and active feature counts. |
| `polynomial` | Maximum degree uniformly 1–16. The first active column is forced to that degree; other active columns sample uniformly from 1 through that maximum. |
| `quadratic` | Rank 1–256, interaction strength uniform(-2,2), linear strength log-uniform(0.01,4). |
| `rff` | Terms from 64,128,256,512,1024,2048,4096; length scale log-uniform(0.05,20). Standard Gaussian inputs only. |
| `periodic` | Terms uniformly active-count–64; length scale log-uniform(0.05,20). Standard Gaussian inputs only; active count above 64 is rejected. |
| `radial`, `sinusoidal` | Additive or ridge projection; terms from 1,4,16,64,256,1024,4096, restricted to at least the active count for additive projection. Length scale log-uniform(0.05,20), width spread log-uniform(1,16). |
| `gp` | 1–64 kernels, sum or product, kernel/geometry settings below. Exact reference workloads only, at most 1,024 task rows. |
| `composition` | 1–64 components, sampled expression depth limit 1–8, sums and products, settings below. |

Analytic families and compositions choose independent calibration sizes from
256,1,024,4,096. Exact GPs choose 32,128,256,512. These are independent calibration
samples, not context/query labels. Estimated normalization is not a guarantee
that a finite sample accurately estimates heavy-tail population variance.

### Input distributions

Grouped, polynomial, quadratic, smooth and exact-GP settings choose uniformly
among Gaussian, uniform, Student-t and a symmetric two-Gaussian mixture.
Location is uniform(-2,2), scale log-uniform(0.25,4), Student-t degrees of freedom
from 0.5,1,2,3,5,10,30,100, and mixture gap uniform(0.5,4).
The underlying `inputs-v1` definition determines how location and scale apply.
All fields are recorded even when a selected distribution does not use them.

For Student-t inputs, grouped linear needs moment order 2, polynomial needs twice
its maximum degree, and quadratic needs order 4. If the sampled degrees of freedom
do not exceed the required order, the recipe explicitly conditions the standardized
input draw on `abs(x) <= 4`. The input sampler uses bounded rejection sampling;
it never clips observations. Smooth profiles use the conservative order-2 rule;
exact GPs use order zero. Broader manually authored smooth recipes may also use
unbounded heavy tails with lower degrees of freedom.

Plain linear and Fourier implementations currently accept standard Gaussian inputs.
This profile records that exclusion; it does not silently replace them with another
family. Grouped linear with group size one supports the other input distributions
through its own explicitly named recipe and estimated calibration.

### Exact-GP settings

Each component chooses RBF, Matérn-1/2, Matérn-3/2, Matérn-5/2, rational-quadratic,
spectral or Gibbs covariance. The first covers all active inputs; later components
choose a uniform subset size from 1 through the active count, then select that many
axes without replacement. Each selected axis gets a log-uniform(0.25,4) scale;
component length scale is log-uniform(0.05,20). Sum weights are log-uniform(0.1,10);
product weights are exactly one, as required by the kernel contract.

Rational-quadratic alpha is log-uniform(0.1,10). Spectral frequencies are independent
uniform(0,4) cycles per input unit. Gibbs centers are uniform(-2,2), transition
scales log-uniform(0.25,4), length slopes uniform(-log(4),log(4)), and amplitude
slopes uniform(-log(2),log(2)) divided by the number of selected axes. These choices
allow feature-specific variation and spatially varying smoothness/amplitude.
No approximation replaces an exact GP. A numerical failure remains a failed task.

### Function compositions

With probability 1/4, inputs are standard Gaussian; otherwise use the input
sampling rules above. For Student-t compositions, the conservative required moment
order is 2×16×64, so the listed degrees of freedom all receive explicit bound 4.
The analytic child pool is grouped linear, polynomial, quadratic, radial and
sinusoidal. Standard-Gaussian compositions additionally include linear and RFF;
periodic is included only when the parent active count is at most 64.

The first child covers all parent active inputs. Later children choose a uniform
active count from 1 through the parent's count and a uniformly selected subset.
Each child uses the broad family settings, shares the parent's input distribution,
and has zero observation noise, unit separation and no class offsets. These target
controls apply once at the parent. Component strengths have equiprobable signs and
log-uniform(0.25,4) magnitudes.

The expression is built by randomly splitting the ordered component indices until
one remains or the sampled depth limit is reached. At that limit, all remaining
indices become siblings. Each combination independently selects sum or product,
with the same signed-strength distribution. Every component is used once; actual
depth may be less than the sampled limit. There is no silently selected simpler
replacement when a product is numerically invalid. GP children remain excluded
because they are joint reference draws, not analytic functions.

## Adapted reference: `tabiclv2-adapted-v1`

The accepted source revision is
[`9530bd65a7b806f6ad4f9f4335af745577e9d3cc`](https://github.com/soda-inria/tabicl/tree/9530bd65a7b806f6ad4f9f4335af745577e9d3cc).
This profile preserves selected disclosed settings while using r0's independently
implemented functions, normalization and target adapters. It does not reproduce
TabICLv2's entire sampler, random-matrix distributions or causal graph prior.

| Name | Preserved setting and explicit adaptation |
| --- | --- |
| `power` | Uniform integer exponent 2,3,4,5. Applies the power to a sampled normalized linear projection using one repeated child. The source applies integer powers as activations; r0's projection and calibration are adaptations. |
| `quadratic` | At most 20 active inputs; requests above 20 fail. Uses r0's rank-20 quadratic construction, not the source's random affine quadratic matrix distribution. |
| `rff` | Exactly 256 frequencies. Uses r0's Gaussian spectral frequencies and log-uniform(0.05,20) length scale; it does not reproduce the source's generalized random kernel. |
| `product` | Exactly two analytic children, no nested products. Uniform child pool: r0 linear, adapted quadratic, adapted RFF, radial and sinusoidal. Smooth children use 64 additive terms and log-uniform(0.05,20) length scale. Requires at most 20 active inputs so every child choice is valid. |

All reference inputs are standard Gaussian and calibration size is 4,096. Function
coefficients, feature selection, label adapters and other settings follow the
existing r0 recipe definitions, not inferred unpublished paper settings. Broad
profiles keep their larger ranges; the reference restrictions do not change library
limits. Sources: [integer powers](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L168-L173),
[quadratic, Fourier and product functions](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py#L206-L313).

The sampler is a public selection function in this slice. Integrating per-task
selection into resumable runs and the explorer is subsequent #27 work; callers
can already save and generate any selected recipe through the existing interfaces.
No profile is yet marked bulk-training eligible by this document.

## Reproduce the settings and examples

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python3 scripts/run_check.py python scripts/sampling_walkthrough.py \
  --out docs/evidence/issue-27-sampling-profiles
```

Use a new output directory for each run. The script retains 128 settings draws per
family/profile combination (1,792 total), complete request/recipe records, summary
histograms, and one generated example per family/profile/task-type combination
(42 total). It checks recipe JSON round-trips, deterministic sampling, exact task
replay, finite scores and generation-graph reconstruction for rows 0,1,3.

The local run at `a9490e120e5f0066222bf9a2d46a26f440e985a7` passed all 42 examples
with zero signal/target graph reconstruction error. Its settings draws covered all
polynomial degrees 1–16, every listed RFF term count, composition component counts
including 1 and 64, and actual expression depths 0–8 (zero means one component).
They also included one- and 64-kernel GPs. These are observed settings counts;
not every selected setting was generated, nor is this the full diversity or
resource qualification matrix. Raw records remain in the ignored local directory.
The runtime/source version is recorded independently of later documentation edits.
