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
| Sampled explorer save/reopen | `75f9785` | subsequent nonlinear-profile implementation |

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
