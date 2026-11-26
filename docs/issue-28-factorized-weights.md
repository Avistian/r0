# Issue #28 factorized structural projections

This is a partial weight-family checkpoint under accepted decisions 10, 13 and 18
of the [full specification](specs/0028-causal-network-generators.md). It does not
complete weight sampling, neural architectures or the paper-option inventory.

## Frozen construction and evaluation

`FactorizedWeights` records finite rectangular matrices `left`, `right` and a
nonnegative `diagonal` vector. The inner dimensions must agree. Its equation is

```text
W = (left * diagonal[None, :]) @ right
node_affine = input_activation(parent_channels) @ W + bias
```

`MatrixCausalNode` extends the existing stochastic activated node and stores both
the realized dense `weights` and its `weight_construction`. Ordinary causal nodes
keep their existing fields and RNG identities. A matrix node must have parents;
its recorded projection dimensions must match the dense weights and output width.
Recipe admission counts the construction scalars in addition to the dense matrix.
Only after admission does recipe validation materialize the factors and require
exact agreement with the stored dense projection. Generation uses the admitted,
frozen dense weights; it does not resample or refit the matrix from task rows.
Nonfinite construction results fail explicitly without retry or clipping.

The public decoder restores nested construction records. Original-row inspection
shows both the effective weights and their factors. Saved recipe replay retains
these fields and identities. Removing a required projection record from the
`activation-factorized-v1` sampled construction is rejected. Existing Gaussian
innovations, signal-dependent responses, activation stages and moment checks
remain available on authored matrix nodes.

A worked example is

```text
left = [[1, 2], [3, 4]]
diagonal = [2, 0.5]
right = [[5, 6], [7, 8]]
W = [[17, 20], [44, 52]]
first output, zero bias/noise = 17*x0 + 44*x1
```

The retained test also checks the complete parameter count: four root scalars,
eight dense child scalars, seven default exogenous/response scalars and ten
factor scalars, 29 total. These are admission/storage estimates, not measured
peak memory. The factors are not required to be orthogonal, and `diagonal` is not
claimed to contain actual singular values. Zero coefficients and rank-deficient
projections remain valid.

## Versioned exploration profile

`causal-factorized-v1` uses forward, layered or predecessor-redirection topology,
node widths uniformly drawn from 1, 2, 4, and the existing 26-choice activation
preference law. Roots retain the existing Gaussian innovation law. For each
nonroot with `n` incoming channels and `m` output channels:

- Choose inner dimension `k` uniformly from integers 1 through `min(n,m)`.
- Draw independent standard Gaussian `left[n,k]` and `right[k,m]`.
- Set every inner coefficient to `1/sqrt(n*k)`.
- Store the factors and their dense product on the node.

Each entry has population variance `1/n` under this law; entries are not generally
independent. Biases and base structural-noise scales retain the existing Gaussian
affine sampler's settings. All factor dimensions and their added storage are
admitted before factor arrays are allocated. The profile does not retry failures,
filter valid draws or alter previous profiles. It reaches the shared explorer
backend's generation, selected-row trace, save/reopen and replay paths. Actual
browser and Colab interactions remain pending.

## Source distinctions and remaining scope

The [pinned I2 matrix source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_matrix.py)
forms Gaussian factor products with inner positive weights and explicitly avoids
an orthogonality claim. This checkpoint delivers the factor-product construction
and interpretation relevant to `I2.matrix.03` and `I2.matrix.18`. Its configurable
authored coefficients and separate r0 exploration profile are not an identical
source sampler: the source uses inner dimension `min(n,m)`, sampled positive
weights, top-level jitter and row normalization. Those sampling details remain
required, separately from this construction.

The [positive-weight source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_weights.py)
also distinguishes uniform values from shuffled polynomial decay with log-Gaussian
perturbations. Those laws, importance-weighted Gaussian matrices, signed kernel
matrices and their distinct point variants, activation-transformed matrices,
nonrecursive children, jitter/normalization stages, masks/block sparsity and
shared weight-family preferences remain required. Source inspection did not copy
foreign implementation code. The full #28 goal stays active.

## Development and verification

The authored construction has RED `04f8e1c`, GREEN `559d2af`; saved sampled workflow
has RED `a14f620`, GREEN `fa26bcb`; discarded-provenance rejection has RED `dc84af8`.
The initial authored test contained an arithmetic typo in the unused second output
(50 instead of 52) and expected a list where inspection retains a tuple. Both were
corrected; the target's independently specified `17*x0 + 44*x1` stayed unchanged.
An initial commit hook also caught import ordering; no hook was bypassed.

The focused authored checks and all three new saved-workflow cases pass. Exact
frozen revisions, independent review, full checks and census results are appended
after execution; this paragraph does not claim those gates already passed.
