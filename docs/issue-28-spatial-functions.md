# Spatial causal functions

Inventory `P3.structure.02` and `SPEC.F01.06` require a named r0
reconstruction because the peer activation's exact form is undisclosed. The existing
`FrozenEMFunction` supplies a normalized radial basis without adding a second
implementation. This mapping is called **r0 normalized spatial basis**; it is not
claimed to reproduce TabPFN-3's undisclosed function.

For centers c_j, positive widths s_j and input x, the basis is

    a_j(x) = exp(-(||x-c_j||_p / s_j)^q) / s_j
    phi_j(x) = a_j(x) / sum_k a_k(x)
    y(x) = W phi(x)

The implementation evaluates the normalized weights through stable log-space
softmax. With p=q=2 these are Gaussian-shaped spatial responses. The existing
operator supports p in [1,4] and q in [1,2], unequal positive widths, arbitrary
finite centers and a signed output matrix. An identity output matrix exposes the
basis activations themselves; a rectangular matrix changes dimensionality. This
is neither a fitted Gaussian-mixture density nor a claim of Gaussian PDF
normalization in arbitrary dimension.

## Generate datasets

```sh
python3 scripts/run_check.py r0 generate --config configs/generation/spatial-causal-cpu.json --out /tmp/spatial-regression
python3 scripts/run_check.py r0 generate --config configs/generation/spatial-dual-target-cpu.json --out /tmp/spatial-classification
```

The examples use two observed Gaussian coordinates, three unequal-width spatial
centers, and a signed linear target readout. Both generate the same underlying
features and continuous targets. The classification view applies the explicit
fixed boundaries -0.5, 0.5 and 1.5. Change centers, widths, output matrices and
p/q in the saved recipe; use `FrozenEMFunction` directly inside other supported
causal graphs. The existing `JointFunctionPrior(functions="em")` also samples a
broader soft-cluster family with random centers, spreads and output mappings.
These examples are small usable configurations, not the limit of the family.

Saved-array verification evaluated the equation independently in NumPy for both
128-row datasets, checked exact paired feature tables, and checked classification
labels against independent threshold lookup. Raw outputs remain under ignored
`docs/evidence/issue-28-spatial/`. This establishes the named reconstruction's
behavior, not peer-source equivalence or whole-issue completion.
