# Issue #28 nonlinear structural mechanism checkpoint

This is partial implementation of the [accepted scope](specs/0028-causal-network-generators.md).
It does not close the activation-source inventory or the wider network, tree,
intervention, typed-state and workbench requirements.

## Interface, equations and compatibility

`ActivatedCausalNode` extends the affine node with explicit `input_activation`,
`output_activation` and `noise_position`. It requires a `CausalRecipe` with
`version="causal-v2"`. `causal-v1` retains its original fields, evaluation order,
RNG identity and serialized recipe/mechanism content. The legacy sampled-task
identity in `test_affine_v1_retains_its_pre_extension_task_identity` was captured
at `23f0302` before this extension.

For concatenated parent channels `x`, projection `W`, bias `b` and saved scaled
innovation `e`, the two orders are:

```text
before_output: g(f(x) @ W + b + e)
after_output:  g(f(x) @ W + b) + e
```

Roots omit `f(x) @ W`; a non-identity input activation on a root rejects because
there are no parent channels. Every node can choose different input/output
functions. Scalar/vector dimensions and the shared target adapters remain intact.
The noise domain remains independently keyed Gaussian innovations; new noise laws
and signal-dependent noise are not implied by this checkpoint.

`CausalActivation(kind, parameters)` stores named numeric parameter pairs as an
immutable tuple. JSON uses pairs of arrays; the shared recipe/inspection decoder
reconstructs tuples without changing the caller's record. Duplicate, unknown,
inapplicable, nonfinite or malformed parameters reject. All functions accept
optional `input_scale`, `input_shift`, `output_scale`, `output_shift`, implementing
`output_scale * h(input_scale*x + input_shift) + output_shift`. Their defaults are
1, 0, 1, 0. These are frozen constants, not statistics fitted to task rows.

Activation-parameter entries contribute to the existing parameter and estimated
storage admission. Code-defined defaults do not add stored free parameters.
Resource estimates remain estimates; no measured maximum-envelope claim follows.

## Declared operations

There are 42 named operation choices, including deliberately separate boundary
conventions and aliases. This is not a count of 42 unique algorithms.

- Fixed elementwise functions: identity, tanh, sigmoid, leaky_relu, elu, selu,
  silu, relu, softplus, smooth_relu, relu6, hardtanh, sign, sign_positive,
  gaussian_bump, exp, sqrt_abs, interval_abs_le_1, interval_0_1, sin, square,
  abs, log, log_abs, log_sigmoid, round, step, heaviside, heaviside_positive,
  modulo and gelu.
- Power/log variants: power, integer_power, relu_power, signed_power,
  reciprocal_abs_power and log_clip_abs.
- Within-row channel functions: softmax, tied_max, argmax, rank and argsort.

All evaluations preserve float64 and device. Declared hardtanh/ReLU6 bounds are
part of those functions. They are not general clipping of arbitrary values.
Nonfinite inputs, intermediate rescaling and final results fail explicitly.
Strict log requires positive arguments. Fractional powers of negative arguments
and negative powers of zero reject; neither is silently replaced by an absolute
or signed variant. Gaussian-bump and softplus formulas allow mathematically
appropriate tail underflow, without replacing an overflowing unbounded function.

| Choice | Definition / controls |
|---|---|
| leaky_relu | negative branch slope `negative_slope`, default .01 |
| elu | negative branch `alpha*(exp(x)-1)`, alpha default 1 |
| selu, silu, gelu | PyTorch named definitions; GELU uses the exact-erf form |
| softplus / smooth_relu | `log(1+exp(beta*x))/beta`, beta default 1, positive; stable algebra. Smooth ReLU is an explicitly named softplus reconstruction |
| gaussian_bump / sqrt_abs | `exp(-x*x)` / `sqrt(abs(x))` |
| sign / sign_positive | sign(0)=0 / sign(0)=1 |
| step / heaviside / heaviside_positive | first two use `x>0`; positive variant uses `x>=0` |
| interval_abs_le_1 / interval_0_1 | inclusive `[-1,1]` / `[0,1]` |
| round | nearest integer, ties to even |
| log_abs / log_clip_abs | `log(abs(x)+epsilon)` / `log(max(abs(x),epsilon))`, epsilon default 1e-6 and positive |
| modulo | nonnegative remainder for positive `period`, default 1 |
| power / integer_power | `x**exponent`, default 2; integer variant requires a whole exponent |
| relu_power / signed_power | `relu(x)**exponent` / `sign(x)*abs(x)**exponent`, positive exponent, default 2 |
| reciprocal_abs_power | `(abs(x)+epsilon)**(-exponent)`, positive exponent default 2, positive epsilon default 1e-3 |
| softmax | normalized exponential across channels of each row |
| tied_max / argmax | all tied maxima receive 1 / only first maximum receives 1 |
| rank / argsort | ordinal rank / sorted channel indices; ascending, zero-based, stable original-channel order for ties |

The fixed catalog follows the relevant named families in
[TabICL Appendix B](https://arxiv.org/html/2502.05564v1#A2.SS1).
The [pinned I2 activation source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py)
distinguishes positive-zero sign/Heaviside, clipped-absolute log, tied maxima and
stabilized reciprocal powers. The additive log variant is an r0 extension.
Stable rank/argsort ties are an explicit deterministic r0 convention, not a claim
of identical upstream tie behavior. Source code was read, not copied into r0.

Input and output channel transforms change dependency structure. The ancestry
descriptor accounts for both stages, not just nonzero matrix entries. Explicit
zero-scale and exponent-zero constant transforms stop parent dependence. This
remains a structural potential-dependence descriptor, not measured usefulness or
a symbolic proof that all possible cancellations were eliminated. Channel indices
and one-hot values here are continuous mechanism outputs; native categorical
state/readout semantics remain outstanding.

## Worked checks

With parent `A=-2`, weight 2, bias 1 and innovation `.5`, ReLU before-noise gives
`max(0,-3)+.5=.5`, whereas noise inside ReLU gives `max(0,-3+.5)=0`.
The actual row trace retains the innovation and selected order so a reviewer can
repeat the scalar calculation. Input activation is separately positioned before
the projection; it cannot be conflated with the output activation.

For channels `(0, log(3))`, softmax returns `(.25,.75)`. The second input channel
therefore influences the first output even with a diagonal projection. For
`(3,1,3,-2)`, stable ranks are `(2,1,3,0)`, sorted indices `(3,1,0,2)`, tied maxima
`(1,0,1,0)` and first-argmax `(1,0,0,0)`. Public generation/inspection tests check
these input- and output-stage cases.

## Diagnostics and sampling

Generation records per-stage, per-channel boundary fractions for tanh, sigmoid,
hardtanh, ReLU6 and ReLU. Tanh/sigmoid use tolerance 1e-6 against their base-function
boundaries; hard bounds are exact. A fraction at least .95 flags saturation or
inactive ReLU. Zero scale flags a constant activation transform. These author-set
diagnostic thresholds never reject/filter a valid draw. Fractions describe the
base activation after input transformation, before output scaling and later noise.
They are computed from the generated population for inspection only. Independent
calibration remains separate, and constant targets retain identity normalization.

`profile="causal-activated-v1"` samples the existing three topology families and
Gaussian projections, then independently chooses each node's output function
uniformly from the 23 entries in `_EXPLORATION_ACTIVATIONS`. Half the nonroots also
sample an input function. Noise position is uniform over the two orders. Roots
can transform their bias/innovation and have no input stage. All parameter values
and selected functions are retained in ordinary v2 recipes. The profile changes
its RNG namespace; the older affine profile is unchanged. There are no retries,
quality filters or fallback substitutions.

This finite convenience profile deliberately does not stand for all source
samplers: shared Dirichlet preferences, homogeneous graphs, parameter hyperpriors,
Fourier/GP activations, independently calibrated per-node standardization,
batch-dependent reference modes, per-parent aggregation and the remaining
matrix/tree/network constructions are still required. Strict log, channel
operations and custom powers are available for authoring, not silently counted
as sampled by this particular scalar convenience profile.

## Retained verification checkpoints

| Behavior | RED revision | GREEN revision |
|---|---|---|
| Structural noise order and trace | `fcdbbe2` | `b8fc553` |
| Target-adapter RNG and legacy identity | `94db320` | `b8fc0cc` |
| Fixed catalog and frozen parameters | `ec4836f`, `00b9f26` | `8e052e7` |
| Power and stabilized log variants | `ac821ed` | `b782d76` |
| Channel ties and ancestry | `5cdff35` | `97833a0` |
| Saturation retention | `f9e51bb` | `2cf58b3` |
| Sampled explorer save/reopen | `75f9785` | `1ed90b2` |
| Post-output innovation independence | `fb6efd4` | `28a478f` |
| Extreme sums and integer parity | `d6d368e` | `fbb1aa8` |

`b8a5940` was an intermediate implementation commit still failing shared trace
dispatch; it is not GREEN evidence. The fixed-catalog working-state check before
parameter addition was not committed separately; use the retained revisions above.

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_activations.py \
  tests/explorer/test_causal_session.py
```

The backend integration covers generation, saved recipes, row graphs, save,
reopen and replay. It does not establish actual browser or Colab behavior, native
categorical support, GPU agreement, all source settings, finite population moments
for arbitrary compositions, or measured production resource limits. Those remain
pending requirements of the full goal, not exclusions from it.

## Source-option mapping and remaining work

All 27 `ACT.*` inventory entries have an explicit named authored operation in
this checkpoint. This statement covers the function definitions only; it does
not close their sampling distributions, source-specific scaling, population
coverage or delivery evidence. `smooth_relu` uses the named softplus
reconstruction; stable `rank` ties are an r0 convention. Strict `log` enforces its
domain and is separate from the two safe-absolute variants.

| Inventory records | Current authored mechanism | Outstanding source behavior |
|---|---|---|
| `I2.activation.01`–`.08` | clipped-absolute log, log-sigmoid, round, softmax, tied maximum, first argmax, rank and argsort | Source sampling/scaling; upstream rank-tie equivalence is not claimed |
| `I2.activation.09`–`.12` | positive ReLU/signed powers, stabilized reciprocal and integer powers | Source parameter hyperpriors and selector law |
| `I2.activation.13`–`.17` | Both interval definitions, positive-zero sign/Heaviside and modulo | Source selection law |
| `I2.activation.18`–`.19` | Constituent authored operations | Named source selectors and their preference distributions |
| `I2.activation.20`–`.24` | Frozen input/output scaling constants only | Source standardization, sampled rescaling/shifts, batch-scalar normalization; separately named batch-dependent references |

These are partial mappings, not completion marks on the inventory. The scalar
23-choice exploration profile cannot stand in for the last column.

## Author nonlinear census

The [retained summary](research/0028-nonlinear-census.json) identifies source head
`fbb1aa8`. All 192 attempted tasks generated successfully: 64 per topology, with
22 regression, 21 binary and 21 multiclass per family. Each has 16 rows and four
observed features. No filtering or retries occurred. Flag counts overlap:

| Topology | Disconnected target | Inactive ReLU | Saturated activation | Zero target dispersion |
|---|---:|---:|---:|---:|
| forward | 33 | 11 | 13 | 0 |
| layered | 36 | 10 | 10 | 2 |
| redirection | 4 | 10 | 8 | 1 |

This is bounded author evidence, separate from the independent mathematical
verification. Timing remains NOT_MEASURED; no production-envelope claim follows.
The executable keeps every recipe/status and successful task archive under a new
destination. Original local artifacts are in
`docs/evidence/issue-28/nonlinear-census/`. Reproduce from the checkout:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/causal_topology_census.py \
  /tmp/new-causal-nonlinear-census --profile causal-activated-v1
```

## Independent review and numerical repair

### Standards

At `28a478f`, independent standards review found zero documented breaches and
one low-priority duplication heuristic: four fixed default evaluators were
unreachable because parameter-aware branches already handled those functions.
Scoped recheck at `fbb1aa8` confirms those duplicates were removed, with **zero
new documented breaches or heuristic concerns** in the repair delta. This is
source review, not numerical execution evidence.

### Spec

The first independent verdict at `28a478f` was **REJECT**, despite its full suite
passing 562 tests with one skip. The independent verifier reported **81 PASS,
3 FAIL**, with these retained P2 findings:

1. `reciprocal_abs_power(x=1e308, epsilon=1e308, exponent=.001)` returned zero
   through intermediate overflow. An 80-digit Decimal calculation gives
   `0.49169859803820276`.
2. `integer_power(-1, exponent=9007199254740993)` returned +1 instead of -1.
   The generic power path shared this loss of integer parity.
3. Additive `log_abs(x=1e308, epsilon=1e308)` rejected an intermediate overflow,
   although the result `709.889355822726` is representable. This was an explicit
   failure, unlike the two silent incorrect results above.

Committed RED `d6d368e` retains five extreme-value/parity regression cases. Repair
`fbb1aa8` evaluates the additive logarithm in the log domain, reuses that stable
sum for reciprocal powers, and retains exact integer parity independently of the
floating magnitude kernel. Python integer conversion overflow becomes an explicit
parameter validation failure. No silent clipping or retries were added.

Scoped independent recheck at `fbb1aa8` reports **84 PASS, 0 FAIL** on the original
probe set and **212 PASS, 0 FAIL** on neighboring Decimal-oracle extreme cases.
The original REJECT is not replaced by the later PASS. Eighteen affine-v1 cases
exactly preserve serialized metadata, identities and all tensor bytes against
base `23f0302`. The independent probes cover all 42 authored operations, parameter
boundaries, domain rejection, noise order, ancestry, replay, calibration and
resource admission. This is acceptance of the reviewed partial slice only.

Author `make check` at `fbb1aa8` passed **567 tests, 1 skipped**, plus preflight,
formatting, lint, typing, module and architecture gates. The skip is GPU numerical
agreement because CUDA is unavailable. The local complete log is
`docs/evidence/issue-28/nonlinear-repaired-make-check.log`. Neither the suite nor
the independent probes establish browser/Colab behavior, GPU agreement, source
hyperprior fidelity or full #28 completion.

The [original review result](research/0028-nonlinear-original-review.json) and
[scoped recheck](research/0028-nonlinear-recheck.json) retain exact heads, check
names, script hashes and failures. The identical portable program independently
reproduced the original 81/3 result in an isolated local clone before passing
84/0 on the repaired source. Raw reports remain in
`/tmp/verify-causal-nonlinear-28a478f7b73d-pq2_eowx/verification.json`,
`/tmp/verify-causal-nonlinear-fbb1aa883dba-qh49sk_7/verification.json` and
`/tmp/nonlinear-neighbors-fbb1aa883dba-809_a400/neighbors.json`.
These local paths are retained artifacts, not portable dependencies.

The two published verification programs use public generation/inspection seams
and independently specified numerical expectations. Run from the checkout with
its selected environment (each creates a new temporary evidence directory):

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_nonlinear.py

UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_nonlinear_neighbors.py
```

To reproduce the original REJECT, make an isolated local clone at `28a478f`, set
`PYTHONPATH` to that clone's `src`, and invoke the retained main verifier with the
selected Python interpreter and `--repo` pointing to that clone. Keep this current
verifier outside the old checkout; it was published after that historical commit.
The verifier checks its import location and requires assertions enabled. It
extracts base `23f0302` locally for the exact affine compatibility comparison.
No download or cloud service is required. Subsequent checkpoint publication edits
only documentation and evidence scripts; package/test source remains identical
to independently reviewed and fully checked `fbb1aa8`.

Publication note: the repository hook required first-party import grouping in the
retained main verifier. Its only change from the independently executed portable
program is moving the `torch` import above the `r0` group. The author reran this
published form at unchanged `fbb1aa8`: all 84 named statuses match the independent
recheck. Both script hashes and the separate author replay are recorded in the
recheck JSON; the independent original/repaired records remain intact.
