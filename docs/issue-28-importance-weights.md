# Issue 28: recorded importance-weighted projections

This increment adds authored and sampled positive importance vectors and Gaussian
projections. It is partial issue 28 delivery; the scope ledger remains authoritative.
The prior factorized checkpoint is `977461c276c722592cac7bdd271d9250b93db972`.

## Frozen mathematical contract

`UniformWeightVector` retains an explicit nonempty tuple of draws in `[0,1)`;
its realized values are those draws. Zero is allowed, consistent with a uniform
floating-point generator's endpoint. It does not normalize the vector itself.

`PolynomialWeightVector` retains finite Gaussian realizations `z`, nonnegative
finite decay `d` and log-noise scale `s`, and a permutation (empty means identity).
For zero-based index `i`, compute `a_i = -d*log(i+1) + s*z_i`, permute the scores,
and return `sqrt(n)*exp(a)/sqrt(sum(exp(2*a)))`. This is mathematically softmax
followed by RMS normalization; the vector has squared norm `n`. Realized draws
are replay inputs, not assertions that an individual tuple passed a normality test.

The implementation centers scores before exponentiation. It computes products,
subtractions and centering using Decimal arithmetic with enough precision for
exact binary64 products; `log(i+1)` itself is a binary64 approximation. This avoids
losing small decay differences under a huge common perturbation. The declared
`exact_scores_limit` (default 1,000,000) bounds the number of scores before Decimal
work, and is resource control rather than a model parameter. Underflow in final
float64 values can produce zero. Epsilon-zero matrix normalization centers over
active Gaussian coordinates to retain a direction even when those weights alone
would underflow. Internal negative infinity for a zero weight is not serialized.

`ImportanceWeights` retains a finite rectangular Gaussian realization `G`, one
matching importance vector per row, and finite nonnegative `epsilon` (default
`1e-6`). For each row `r`, the projection is
`W_r = (G_r * v_r) / (epsilon + ||G_r * v_r||_2)`.
An all-zero numerator gives an all-zero row, including when epsilon is zero.
Scaled evaluation avoids overflow in the norm; extreme products use logarithms.
These are row-normalized projections, not orthogonal or singular-value-certified
matrices. Biases, activations, exogenous defaults, target controls and shared
causal generation semantics remain as recorded elsewhere.

A `MatrixCausalNode` stores both the realized projection and its construction;
recipe admission precedes construction materialization and rejects disagreement.
Parameter accounting adds all Gaussian entries, vector draws, polynomial decay
and scale, explicit permutation indices, and epsilon, as well as the existing
node and dense-projection parameters. Construction records travel through JSON,
row traces, saved archives, reopen and replay. A sampled factorized record must
contain actual factorized projections; importance provenance has its own guard.

## Source and local sampling choices

Primary references are pinned TabICLv2 graph-lib
[`_weights.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py),
[`_matrix.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py),
and [`_base.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_base.py).
The vector decay/log-noise/shuffle law and Gaussian product with epsilon-plus-norm
follow those mathematical definitions. The implementation is independent code.

`causal-importance-v1` / `activation-importance-v1` is an r0 exploration profile:

- Forward, layered and predecessor-redirection graph families; widths 1, 2, 4;
  existing 26-choice activation preference law, including its retained modes.
- Each nonroot projection independently chooses uniform or polynomial vectors
  with probability one half. Each row has independent vector draws. Gaussian
  matrix entries are independent standard normals. Roots retain their existing
  activated Gaussian form; nonroots use default Gaussian exogenous/constant noise.
- For polynomial width `n`, decay lies between `0.1/log(1+n)` and 6. Draw local
  `u ~ U[0,1)`, transform `clip(1.2*u-.1,0,1)`, then interpolate logarithmically.
  Under the ideal continuous law this gives endpoint masses 1/12 each and interior
  mass 5/6. Log-noise scale is locally log-uniform from `1e-4` to 10. Each row gets
  its own standard-normal tuple and uniform permutation. Epsilon is `1e-6`.
- Admission accounts for retained arrays before allocating Gaussian/vector arrays.
  Sampling, row execution and saved replay do not retry or discard failed draws.

The one-half family mix, graph/width choices, local parameter independence and
stable float64 arithmetic are explicit adaptations. This does **not** reproduce
source GlobalSampler shared meta/global numerical preferences, recursive matrix
mixtures, kernel/transformed matrices, jitter wrappers, masks, full networks or
joint source priors. These remain required issue 28 work, not unsupported options
silently treated as complete. The old five causal profile streams must remain
unchanged, including their failures.

## Verification status

Implementation head `fc0265b7706112a0b674a2c85078f177fb265b83`:
- Uniform authored projection: RED `e5b4be7`, GREEN `c13a2ed`.
- Factorized family provenance: RED `f36065f`, GREEN `ff0a09f`.
- Polynomial laws, permutations and huge common offset: RED `879faea`, GREEN `534b3cb`.
- Saved explorer workflow on three graph families: RED `671a7e5`, GREEN `fc0265b`.
- Repository check, independent review, census and publication: PENDING.

These are CPU software and retained-example checks. Scientific efficacy, speed,
CUDA behavior and complete browser/Colab acceptance are NOT_MEASURED/NOT_RUN.
