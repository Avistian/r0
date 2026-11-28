# Issue 28: shared matrix preferences

This increment connects retained numerical and categorical hyperpriors to all five
implemented matrix families. It extends the preceding
[numerical sampler](issue-28-parameter-preferences.md) and
[matrix constructions](issue-28-matrix-families.md); it does not complete #28 or
reproduce the complete source joint prior. Source activation weighting, additional
network/tree/GP and structural options, typed states, interventions, and actual
browser/Colab delivery remain required.

## Reference and adaptations

The reference is TabICL commit `9530bd65a7b806f6ad4f9f4335af745577e9d3cc`:
[`GlobalSampler`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py)
and [positive vectors](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py).
Implementation is independent; upstream code was not copied.

The source exposes local/global/meta numerical and categorical choices. Corrected
categorical meta sampling caches a distribution; its legacy lookup normally
refreshes it. Source categorical sampling also uses ambient NumPy/Torch randomness,
including a fresh context whose NumPy seed is obtained from the ambient RNG.
Here all draws use the caller's injected Generator, and `shared` versus `refresh`
is explicit. Numerical and categorical names have separate namespaces. Accidental
cross-kind cache collisions are not reproduced. These are declared reproducibility
adaptations, not bitwise source RNG parity.

The positive-vector family still uses polynomial decay plus log-Gaussian noise,
permutation and RMS normalization. Categorical preferences always use this family,
matching the source's direct SimpleRandomWeights construction; changing the matrix
vector family to uniform does not change the categorical hyperprior. Categorical
vectors retain fresh numerical Beta hyperpriors for their decay/noise settings.
Matrix vectors instead share numerical hyperpriors across requests by name.

Categorical arithmetic uses binary64 vector values and exact rational inverse-CDF
comparisons. Source float32 arithmetic and ambient choice streams are not copied.
Scalar numerical requests replace source vectorized batch draws in the matrix
planner. They preserve shared Beta hyperparameters and independent conditional
draws, but use a different RNG call order. Global matrix preferences use explicit
quantile remapping across width-dependent bounds, not unchecked reuse of a final
value from incompatible bounds or shape.

## Public records and sampling

All interfaces are exported through `r0.modules.priors`.

`CategoricalParameter(name, choices, mode)` declares 1..256 distinct nonempty string
labels. `CategoricalSampler` supports local/global/meta fallback like its numerical
counterpart. Explicit local/global requests override the default. Repeated names
must retain the same ordered labels within their effective mode; a changed contract
is rejected before RNG consumption. Named scopes for different modes are separate.

- Local choices use a new uniform unit and equal category masses on each call.
- Global choices cache the first uniform unit and reuse its selected category.
- Meta choices use a recorded PolynomialWeightVector. Shared policy retains one
  vector per name; refresh policy retains a new vector for every request.

Every CategoricalDraw retains its request, preference index, unit, selected index
and label. CategoricalPreference retains its effective mode, any cached unit,
weight vector and the numerical record that produced its hyperparameters. Freeze
produces CategoricalPreferences; `categorical_preferences_from_dict` checks scope,
first-use order, label contracts, reference coverage, cache policy, vector/hyperprior
agreement and every inverse-CDF realization. Changing refresh to shared cannot
silently reinterpret multiple distributions under one name.

The categorical numerical requests use singleton shape `(1,)`: decay is
log-interpolated from `0.1/log(1+n)` to 6 with boundary mass, and log-noise scale is
log-interpolated from 0.0001 to 10. Each has a freshly sampled Beta law. These
hyperparameters and all Gaussian values/permutation indices remain in the record.
Stored weight values round to binary64 before categorical mass comparisons. Exact
rational cumulative comparisons cannot select a zero-weight bin through summation
rounding. Positive binary64 masses remain selectable even when their normalized probability
view would underflow. `log_probabilities` retains those values in log form; asking
for an unrepresentable binary64 `probabilities` view fails explicitly without
invalidating the record or its exact inverse-CDF choice.

PreferenceLimits includes nested numerical draws and values, not only returned
category labels. For a new three-choice meta preference, 22 retained numeric
parameters describe two singleton Beta requests, eight describe the polynomial
vector and three describe the category draw: 33 total. A shared second category
draw adds three; a refreshed second distribution adds another 33. The sampler
admits these costs before consuming randomness or allocating the weight vector.
The raw JSON decoder admits the full nested numeric cost before reconstruction.

## Matrix prior and bindings

`CausalMatrixPrior` configures effective mode, categorical shared/refresh policy,
polynomial/uniform matrix vectors, top-level and simple-child family allowlists,
kernel point laws and covariance innovation bases. Defaults expose all five top
families, all four non-transformed child families, both point laws and both
covariance bases. A transformed matrix cannot recursively transform another
transformed matrix. Point dimension remains the earlier explicit r0 value of three.

`sample_preferred_matrices(base, rng=..., prior=...)` takes a base
SampledCausalRecipe with `gaussian-activation-prior-v1` construction. It preserves
its task, topology, readouts, node activation stages, noise order and base Gaussian
node settings. It replaces non-root projections with planned normalized matrices.
An already exogenous/factorized/matrix-specialized recipe is rejected instead of
silently discarding its specialized construction.

Named categorical decisions choose the top matrix family, simple child family,
kernel point law and covariance base. Numerical decisions choose polynomial decay
and log-noise scale, mixture component count (log/floor in [1,16), hence 1..15), and
kernel distance scale (log range 0.1..10). Gaussian matrix entries, vector log-noise
innovations/permutations, point centers/factors/scales/innovations, signs and jitter
remain explicit construction records as before. Node and matrix activation choices
still use the earlier recorded r0 activation prior; complete source activation
weighting is not claimed by this increment.

Planning freezes all numerical/categorical decisions before matrix array draws.
A PreferenceBinding links each top-level draw to exactly one controlled recipe
field, including a length binding for mixture component count. The constructor
requires the complete field catalog for the actual matrix families, the exact
named parameter requests, matching values, and one-to-one coverage of draws and
bindings. It rejects dropped or duplicate bindings, orphan decisions, changed
category contracts and relabelled vector families. These are consistency checks;
they do not prove that authored Gaussian values came from a random sampler.

PreferredCausalRecipe preserves the matrix prior, both preference records and
bindings in JSON and the existing generation inspection card. Resource admission
includes all retained preference numeric parameters and binding indices, in
addition to the earlier geometry and matrix-construction records. The ordinary
CausalRecipe still adds zero preference cost; existing profile schemas remain
unchanged. Record metadata contributes to the conservative working-memory estimate;
this is not measured process RSS or an OS-enforced cap.

## Exploration profiles and delivery

| Profile | Numerical/category mode | Categorical policy |
| --- | --- | --- |
| `causal-shared-matrices-v1` | meta | shared |
| `causal-refreshed-matrices-v1` | meta | refresh |
| `causal-local-matrices-v1` | local | shared scope, independent uniform draws |
| `causal-global-matrices-v1` | global | fixed uniform choices; numerical quantile remapping |

All four use polynomial matrix vectors and support forward, layered and
redirection topologies. Uniform vectors and restricted family/point/base choices
are available through CausalMatrixPrior and their saved recipes. New profiles have
separate stream identities. Earlier profiles retain their sampling paths and
serialization; exact legacy verification is required for acceptance of this delta.

The shared explorer backend can generate, save, reopen, inspect original rows and
replay these recipes. Automated backend cases are not browser or Colab proof.
Targeted RED/GREEN logs cover categorical laws/admission, all five matrix families
and the twelve new profile/topology save/replay combinations. Full checks, independent scoped review and unfiltered census results appear below.

## Reproduction of development checks

Retained raw logs, exact argument lists and environment selection:
[categorical laws/admission](evidence/issue-28/legacy-research/0028-shared-development-categorical.json),
[matrix bindings](evidence/issue-28/legacy-research/0028-shared-development-bindings.json), and
[profile save/reopen/replay](evidence/issue-28/legacy-research/0028-shared-development-profiles.json).
These include uncommitted test additions during development; they do not claim
immutable-revision suite coverage. Full-suite evidence is retained separately.

All commands run from the working checkout with the explicitly reused environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_categorical_preferences.py --tb=short
```

The binding check substitutes `tests/priors/test_preferred_matrices.py`. The profile
check substitutes `tests/explorer/test_causal_session.py` and adds
`-k 'shared or refreshed or local or global'`. Final targeted results are three
categorical checks, five matrix-family binding checks and twelve saved-session
profile/topology checks. Actual browser/Colab execution is still outstanding.

## Frozen-checkpoint verification

At `47630e8`, [full repository checks](evidence/issue-28/legacy-research/0028-shared-check-47630e8.txt) pass
**648 tests, one skipped**, including all static gates. The skip is the existing
CUDA case on this CPU-only environment. Smoke/eval-smoke remain NOT_AVAILABLE:
the checkout exposes no train/eval CLI. This is the initial, subsequently rejected source checkpoint; the repair and final
verification appear below. Evidence and census-program changes are separate.

The original [standards review](evidence/issue-28/legacy-research/0028-shared-standards-initial.md) identified
one P2 missing evidence-publication path and a nonblocking duplication heuristic
for scalar versus singleton-shape weight-prior requests. Its
[documentation-only follow-up](evidence/issue-28/legacy-research/0028-shared-standards-delta.txt) confirms
that all ten development logs match the originals and resolves the publication-path
finding. Original historical source snapshots are not represented as verified.
The follow-up does not rerun or broaden the source review.

### Unfiltered author census

Each profile attempted the same 192 seed-28 tasks: 64 per topology, cycling
regression/binary/multiclass. No failed or degenerate task was replaced or filtered.

| Profile | Valid / attempted | Retained failures |
| --- | --- | --- |
| Shared | 192 / 192 | None |
| Refreshed | 189 / 192 | Forward 18/63 integer-power overflow; layered 38 signed-power overflow |
| Local | 192 / 192 | None |
| Global | 191 / 192 | Redirection 41 integer-power overflow |

All five raw matrix families, both point laws and both covariance base laws occur
in each profile's attempted recipes. Component counts 1..15 occur across the pooled
census; that is not a claim that every count occurs in every individual profile.
Disconnected targets, inactive/saturated activations and zero target dispersion
remain in the valid denominator. These are coverage observations, not predictive
quality or source-joint-prior acceptance.

Published summaries:
[shared](evidence/issue-28/legacy-research/0028-shared-census-shared.json),
[refreshed](evidence/issue-28/legacy-research/0028-shared-census-refreshed.json),
[local](evidence/issue-28/legacy-research/0028-shared-census-local.json), and
[global](evidence/issue-28/legacy-research/0028-shared-census-global.json).
Each manifest retains every attempt's header/recipe hash, all complete failed
records in ordered reconstructable byte parts, and hashes of the local raw JSONL
and saved task archives:
[shared manifest](evidence/issue-28/legacy-research/0028-shared-census-artifacts-shared.json),
[refreshed manifest](evidence/issue-28/legacy-research/0028-shared-census-artifacts-refreshed.json),
[local manifest](evidence/issue-28/legacy-research/0028-shared-census-artifacts-local.json), and
[global manifest](evidence/issue-28/legacy-research/0028-shared-census-artifacts-global.json).
Successful raw recipes and archives remain local, explicitly LOCAL_ONLY in those
manifests. Source/profile/seed/task controls and the census program reproduce them.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/path/to/pinned-checkout/src python scripts/causal_topology_census.py \
  /path/to/new-output --profile causal-shared-matrices-v1
```

Repeat with the other three profile names and distinct new output directories.
Use source revision `47630e8` with the published census program; its hash is in each
manifest. Timing, peak RSS, GPU, browser/Colab and scientific acceptance are not
established by this census. The complete issue #28 goal remains active.

### Preserving subnormal categorical mass

The initial source at `47630e8` is **SCOPED_REJECT** despite its passing full suite
and broad compatibility checks. The [independent review](evidence/issue-28/legacy-research/0028-shared-spec-reject.md)
found that construction unnecessarily required a representable binary64 probability
view. Its [input](evidence/issue-28/legacy-research/0028-shared-subnormal-input.json) has positive masses
`[1.2247448713915892, 5e-324, 1.2247448713915892]`. An exact unit of 0.5 falls strictly
inside the middle bin: the exact halfway mass is the first mass plus half of the
positive middle mass. The [original rejection](evidence/issue-28/legacy-research/0028-shared-subnormal-reject.json)
and [source/verification receipt](evidence/issue-28/legacy-research/0028-shared-spec-reject-receipt.json) remain
retained; earlier passing checks do not erase this finding.

Repair `639dd2a` removes the display requirement from construction. Exact mass
selection and JSON replay preserve the middle bin, `log_probabilities` exposes a
finite log probability for every positive represented mass (negative infinity for
zero mass), and only the unrepresentable binary64 probability view raises an
explicit error. [Regression RED](evidence/issue-28/legacy-research/0028-shared-subnormal-red.txt) and
[GREEN](evidence/issue-28/legacy-research/0028-shared-subnormal-green.txt) are retained. Independent repair
verification and final full-check results are retained separately below.

All **768 retained census recipe identities** resample identically at `639dd2a`.
The [identity-only receipt](evidence/issue-28/legacy-research/0028-shared-census-identity-receipt.json),
[individual outcomes](evidence/issue-28/legacy-research/0028-shared-census-identity-artifacts.json), and
[portable identity program](../scripts/verify_shared_census_identity.py) qualify
reuse of the earlier census recipes, including the four failed-generation recipes.
This is not a fresh generation census. The sole source-file delta from the census
is the categorical probability-view repair; generation/matrix code is unchanged.


### Final repaired-source verification

At `639dd2a`, [full repository checks](evidence/issue-28/legacy-research/0028-shared-check-639dd2a.txt)
pass **649 tests, one skipped**, including all static gates. The skip remains the
CUDA case. Smoke/eval-smoke are NOT_AVAILABLE, as above.

The [independent repair review](evidence/issue-28/legacy-research/0028-shared-spec-repair.txt) reports
**PASS_SCOPED_REPAIR**: 33 categorical/admission/malformed probes, eight boundary
and log-view probes, two extreme constructor/decoder checks, and 124 matrix probes
(**167 total**). Matrix coverage comprises 96 controlled configurations, twelve
profile replay cases, four actual backend save/reopen workflows, nine binding
attacks and three specialized-input overwrite guards. All **504 legacy outcomes**
match frozen `0341b4c`, including four failed draws on each revision. No failed
generated draw was retried. The receipt verifies all 141 tracked source files and
63 imported module paths/hashes; the independent Decimal log-probability oracle's
maximum observed absolute error is zero within these fixtures.

[Raw evidence](evidence/issue-28/legacy-research/0028-shared-repair-artifacts.json) includes all final raw
outputs, complete receipt, saved synthetic backend archives and original harness
error logs in bounded ZIP bundles. Adjacent manifests contain exact member hashes;
extract the bundles into one new directory to reconstruct the retained run layout.
Source archives are excluded and can be reconstructed from the pinned Git revision.
[Reproduction commands](research/0028-shared-spec-reproduction.md) use the six
`scripts/shared_preference_*.py` programs. Three published copies add one import
separator blank line required by repository Ruff; all six ASTs match the executed
programs, whose exact bytes and both hashes are retained in the evidence manifest.
Published copies were not rerun. Historical absolute paths
identify actual runs; use new output directories for reproduction. The original
SCOPED_REJECT and harness errors remain distinct from the repaired result.

Publication adds documentation, evidence and portable verification programs only;
source/test bytes retain the checked and reviewed `639dd2a` state. This establishes
the scoped shared-matrix increment, not complete source activation weighting,
browser/Colab delivery, scientific efficacy or completion of issue #28.
