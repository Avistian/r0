# Issue #28 activation preferences and parameter priors

This checkpoint implements activation sampling within the
[accepted scope](specs/0028-causal-network-generators.md), especially decisions
11–13 and 19. Function-family, converter-family and weight-family preferences,
source selectors, calibration/scaling references and the complete workbench are
still required. It is not full #28 or whole-paper acceptance.

## Public interface and frozen records

`sample_causal_recipe(..., activations=CausalActivationPrior(...))` returns a
`SampledCausalRecipe`, extending `CausalRecipe` without adding fields to existing
causal-v1/v2 recipes. New recipes retain the complete activation prior and its
realized categorical probabilities. Their construction is
`gaussian-activation-prior-v1`, using causal-v2 evaluation. The nodes themselves
retain the actual activation kind, numeric parameters and structural-noise order.
`recipe_from_dict`, `generate_task`, selected-row inspection and saved task groups
share this representation. The mechanism card exposes `activation_sampling`.

Sampling reads no generated rows, context/query labels or calibration values.
Changing only task row counts does not resample topology, weights, readouts or
activations. The sampler takes explicit seed/task identity and a versioned prior;
it does not use global random state. Replay consumes the frozen recipe rather
than resampling its preferences. Existing affine/activated convenience profiles
are unchanged. The new subclass preserves their serialized field layouts.

`choices` contains 1–256 `CausalActivation` descriptors; duplicate kinds with
different parameters are allowed. `weights` is empty for uniform weights, or a
matching finite nonnegative tuple with positive total. An explicit zero excludes
that choice. Normalize weights using their maximum before summing so, for example,
`(1e308,1e308)` remains `(0.5,0.5)`. Positive weights that become unrepresentable
probabilities fail explicitly rather than silently excluding their choices.

## Three distinct laws

Let `w` be the normalized base weights and `c` total concentration:

| Mode | Table-level probabilities | Within-table choices |
|---|---|---|
| independent | `p=w`, unchanged across tasks | independent categorical draws from `p` |
| dirichlet | one `p ~ Dirichlet(c*w)` per task on positive-weight support | conditionally independent draws sharing `p` |
| homogeneous | one `k ~ Categorical(w)`; retained `p` is one-hot | one fully sampled activation descriptor shared at every eligible site |

Concentration defaults to 1 and must be finite/positive; a nondefault value
requires Dirichlet mode. It is **total** concentration, not concentration per
choice. For two equally weighted choices and `c=2`, `p[0]` is Uniform(0,1), with
mean1/2 and second moment1/3. Two node choices therefore agree with probability
2/3, versus1/2 for independent fair choices. These identities supply an oracle
separate from the sampling code.

The implementation draws independent Gamma components and normalizes them.
Nonfinite Gamma draws, underflowed positive components or probabilities reject
explicitly without retry or replacement. This finite-precision restriction is
not a choice to filter weak-signal tasks. The rejection remains observable to the
caller; a runner must retain the failed attempt. Very small concentrations can
therefore fail numerically even though their ideal distribution is defined.

`placement` is input, output or both. Roots have no input stage, so it remains
identity. Inactive stages are identity. Homogeneous mode shares the entire
parameterized descriptor across both eligible stages, not just its name.
`noise_after_probability` is in[0,1] and selects each node's noise position.
Probabilities0 and1 are deterministic orders; the structural innovations themselves
remain independently keyed Gaussian draws. This checkpoint does not add noise laws.

## Numeric parameter distributions

`parameter_priors` is empty or one tuple of `ActivationParameterPrior` records per
choice. Each record specifies `name`, `low`, `high` and `distribution`:

- `uniform`: real uniform on the declared interval; equal bounds are fixed.
- `log_uniform`: uniform in log space with positive bounds; equal bounds are fixed.
- `randint`: discrete uniform over both endpoints, requiring signed64-bit integer
  bounds. Retained sampled values remain Python integers.

Frozen parameters and sampled parameter names must be disjoint. All names and
both interval endpoints must satisfy the selected activation's own contract.
An integer-power exponent must use an integer distribution or a fixed whole
value. Positive-power, epsilon, beta and period restrictions remain intact.
The generated descriptor is validated again. Ordinary finite-precision failures
remain explicit; there is no domain-changing fallback or silent clipping.

Independent and Dirichlet modes draw new numeric parameters at each selected
activation site. Homogeneous mode draws them once for the whole graph. Numerical
parameter correlation beyond this homogeneous case remains pending; this does
not claim I2's correlated numerical sampler. Stored probabilities must match the
independent law or homogeneous one-hot law as appropriate, and cannot enable
zero-weight choices. Decoder validation checks node activations against the saved
support/ranges, placement and deterministic noise-order restrictions. Editing a
mechanism beyond its prior requires an appropriately changed recipe, rather than
keeping false sampling provenance.

## Source distinctions

[Kumo Appendix A.1](https://arxiv.org/html/2611.23001v1#A1.SS1) specifies separate
per-table Dirichlet vectors reused for function, activation and converter choices.
This implements the activation preference structure (`K1.structure.02`) with
explicit configurable concentration and support; that text does not provide a
complete numerical concentration setting. Our defaults are r0 choices, not an
exact reconstruction of unpublished settings. `K1.structure.01` and`.03` remain
pending. Independent categorical and homogeneous modes implement distinct
accepted extensions; they are not synonyms for Dirichlet draws.

The [pinned I2 activation source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py)
uses positive power exponents on a log scale. The generic parameter interface can
express these ranges, but does not yet reproduce the source's hierarchical
selectors (`I2.activation.18`–`.19`), correlated numerical sampling or
standardization/rescaling (`.20`–`.24`). Reading its source did not copy its code
into r0. The existing [operation definitions](issue-28-nonlinear-mechanisms.md)
retain paper/source differences and named r0 conventions.

## New exploration profile

`sample_recipe(..., profile="causal-preferences-v1")` retains the three existing
topology families and Gaussian projections/readouts. It uses its own RNG namespace
and chooses modes uniformly among the three above. Dirichlet total concentration
is log-uniform[0.5,50]. Placement is uniform over input/output/both; noise-after
probability is uniform over0,0.5,1. These are declared r0 exploration laws.

Its26 activation choices are the previous23-choice exploration catalog plus
ReLU power, integer power and stabilized reciprocal absolute power. Positive
ReLU/signed/reciprocal powers use log-uniform exponents[0.1,10]; integer powers use
inclusive integers2–6. Negative LeakyReLU slope is uniform[0,1]; ELU alpha,
softplus beta and modulo period are log-uniform[0.1,10]. Other parameters retain
the operation defaults. This profile can generate numerical failures in extreme
compositions and never retries. It does not stand for the full authored42-choice
catalog or complete source-family coverage.

## Retained development evidence

| Behavior | RED | GREEN |
|---|---|---|
| Homogeneous descriptor and replay | `4f0edd9` | `50f96cb` |
| Independent versus Dirichlet law | `91714d7` | `671a0dc` |
| Numeric parameter draws | `baa26ed` | `b02d747` |
| Sampling provenance consistency | `f6fca78` | `7fc4101` |
| Positive-weight underflow | `83a2e4b` | `f43c0fe` |
| Explorer profile save/reopen/replay | `a807fc9` | source revision recorded below |

The first missing-interface probe failed collection before `4f0edd9`; that commit
adds typed descriptors and retains the behavioral failure. Parameter RED
`baa26ed` also exposes the newly added tuple field's missing JSON conversion;
both failures are repaired at its GREEN revision. A failed pre-commit typecheck
for the integer-sampler argument was repaired before committing `b02d747`.

The focused public-seam check passed17 tests after the new profile implementation:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_activation_sampling.py \
  tests/explorer/test_causal_session.py
```

Actual browser and Colab interaction, GPU agreement, final workload/resource
measurements and the remaining source families remain PENDING. Backend replay
is not browser evidence. Full checks and independent review are recorded below
once executed; this checkpoint does not close the full active goal.
