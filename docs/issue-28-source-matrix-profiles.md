# Issue 28: joint source activation and matrix profiles

This increment integrates the [source activation profiles](issue-28-source-causal-profiles.md)
and [recorded matrix preferences](issue-28-shared-matrix-preferences.md). It retains
their source references and declared r0 adaptations. The implementation is
independent; upstream code was not copied. This is one required integration toward
#28, not a complete published joint-prior reproduction or whole-issue acceptance.

## Public contract and selection scope

`with_source_matrices(base, prior=SourceActivationPrior(...),
matrix_prior=CausalMatrixPrior(...), seed=..., task_id=...)` returns a
`SourceMatrixCausalRecipe`. The base must be an ordinary sampled Gaussian-affine
recipe. Specialized recipes are rejected to preserve their provenance. Both priors
must agree on local/global/meta mode and shared/refresh categorical policy.

One source activation sampler owns selections for **both** transformed matrices
and node stages. Matrix transformations use the plain selector. Node stages use
the declared standard or plain selector. Named fixed-primitive preferences and
power exponents are therefore shared wherever their names overlap. In global
plain mode, every transformed matrix and active node stage uses the same retained
primitive, including any exponent. Standard and plain selector-slot preferences
have different names, while the fixed-primitive catalog has one shared name.

Sampling order is explicit: plan every non-root matrix in node order, selecting
its plain activation when needed; then select active node input/output stages in
node order and their standard-wrapper rescale draws; retain noise-order units;
admit the combined recipe; materialize matrix draws; fit independent ancestors.
Matrix family choices, vector hyperparameters and kernel-point choices use the
existing named matrix scopes. Numerical exponent, rescale and matrix scopes have
disjoint names. Categorical hyperparameter vectors retain fresh nested numerical
scopes, as documented in the preceding matrix checkpoint.

`matrix_activation_nodes` binds the ordered prefix of source selections to all
transformed matrices exactly once. Remaining source selections bind to every
active node stage. Matrix field bindings retain every categorical and numerical
decision, including child constructions and mixture component counts. Loading
checks both complete binding catalogs, all selected values, prior policies and
the fitted ancestral replay. No orphan or silently discarded selection is allowed.

The seed derives from `source-matrix-prior-v1`, both prior records, seed and task
id. Task rows/context and admission budgets are excluded from that identity.
Changing observations cannot refit the mechanism. Different profile names are
explicit exploration recipes, not bitwise upstream RNG reproductions.

## Calibration and admission

All five matrix families remain available: Gaussian, importance, factorized,
Laplace kernel and transformed. Transformed matrices have a nonrecursive simple
child, flattened plain activation, retained jitter and row normalization. The
matrix prior retains polynomial/uniform positive vectors, both point-cloud laws
and both covariance bases. Existing Gaussian roots and Gaussian additive structural
innovations remain the base for this integration. Broader exogenous integration
is still required; these APIs do not silently reinterpret a specialized base.

Standard node wrappers calibrate through the materialized matrix equations in
parent-before-child order. Root arithmetic retains its existing Gaussian path.
Matrix nodes compute bias plus activated-parent projection **before** adding
innovation for a before-output stage, matching their generation equation. An
after-output stage adds innovation only after the frozen output wrapper. Each
stage retains independent stream identity, sample digest, shifts and fitted
constants. Plain profiles do not allocate an ancestral calibration sample.

Metadata, matrix constructions, Gaussian stochastic descriptors, fitted wrappers
and their duplicated stored realizations all count toward the combined parameter
limit. Before matrix arrays or fitted tensors, admission includes the maximum
of task/calibration rows and the largest matrix-noise or fitted-stage channel
buffer. Limits remain conservative array estimates, not measured process RSS or
hard operating-system enforcement. Numerical and categorical preference limits
apply separately to each explicit scope; the causal limit admits their total.

## Exploration profiles and delivery

Eight profiles support forward, layered and redirection graphs:

| Profile | Selector | Mode | Categorical policy |
| --- | --- | --- | --- |
| causal-source-standard-matrices-v1 | Standard | Meta | Shared |
| causal-source-standard-refreshed-matrices-v1 | Standard | Meta | Refresh |
| causal-source-standard-local-matrices-v1 | Standard | Local | Shared |
| causal-source-standard-global-matrices-v1 | Standard | Global | Shared |
| causal-source-plain-matrices-v1 | Plain | Meta | Shared |
| causal-source-plain-refreshed-matrices-v1 | Plain | Meta | Refresh |
| causal-source-plain-local-matrices-v1 | Plain | Local | Shared |
| causal-source-plain-global-matrices-v1 | Plain | Global | Shared |

They retain exploration widths 1/2/4, input/output/both activation placement,
noise-after probability 0/0.5/1 and 256 independent calibration rows. These are
explicit r0 exploration settings. Valid disconnected, saturated or constant draws
are retained and flagged. Numerical failure is explicit; no retry or replacement
is introduced. Shared generation and saved explorer paths use the same recipe
decoder and sampler. Actual browser/Colab acceptance remains separate and pending.

## Verification state

Implementation and verification are in progress. Development RED logs retain the
missing joint API and unsupported profiles. The first global plain joint replay
passes. Complete repository checks, independent numerical/binding verification,
compatibility evidence and an unfiltered census are required before publishing
this increment as checked. No whole-issue completion is claimed.
