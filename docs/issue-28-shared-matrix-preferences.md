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
rounding. A positive binary64 weight whose displayed normalized probability would
underflow is an explicit failure, not silently clipped or retried.

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
and the twelve new profile/topology save/replay combinations. Full checks,
independent review and unfiltered census results will be recorded after execution.
