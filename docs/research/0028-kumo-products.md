# Kumo same-input products

The [Kumo paper, Appendix A.1](https://arxiv.org/html/2611.23001v1#A1)
describes applying two sampled functions to the same input and multiplying their
outputs elementwise. Its children can be any of seven non-product families,
including MLP and clustering. This implements K1.structure.09/.10's product
composition boundary; it does not claim a complete reproduction of Kumo's
correlated Dirichlet sampling, clustering distance/scale distributions, or other
separately inventoried source options.

`JointFunctionPrior(functions="kumo-prod")` selects that broader product
catalog. It may be mixed with
non-product tokens, legacy `prod`, and `default`. Mixed products bind their
variant to distinct selected-slot identities; their child plans retain the
corresponding narrowed prior, validated against the parent selection. Pure
legacy and pure Kumo records preserve their existing slot identities.
Historical `prod`, its five cheap children, selector names, and archive versions
remain unchanged. New ordinary and child selector names isolate the broader
catalog in shared sampling contexts. The historical plan field `cheap` denotes
the product-child selector scope; it does not assert that a Kumo MLP fit is cheap.

`FrozenKumoProductFunction` and
`joint-kumo-product-function-realization-v1` distinguish the source contract.
Both children have identical input/output dimensions and are nonrecursive.
Their seven families are linear, quadratic, RFF, oblivious tree, discretization,
cluster/EM, and MLP. Separate fitted-tree variants and parent compositions are
not silently substituted for those children.

Fits retain the same original ancestor population for both children, including
MLP activations and cluster fitting, with raw and fitting streams retained by
the existing realization records. Aggregate admission includes both fitted
constructions and their duplicate frozen equations. Failed admission/fitting
rolls back streams; there is no production retry or normalization of failed
heavy-tail/product draws. Existing parameter, work-memory, width, calibration,
raw-draw and finite-moment limits apply before generation.

The new frozen type participates in decoding, certificates, parent membership,
explicit DAGs, geometry, numerical pipelines, categorical native states, roots,
generation, inspection, resplitting, pairs where supported, and saved groups.
Inspection shows both same-input child values, including categorical root base
reconstruction. Existing default UI source catalogs remain a separate transport
responsibility.

Author validation uses an independent NumPy oracle for a calibrated nonlinear
MLP multiplied by cluster assignments, fixed fixtures covering all seven child
families and both MLP/EM orders, and nine public graph construction paths.
Legacy product archive, generated recipe and mechanism bytes are compared with
base `52b4e10`. Raw logs and comparison scripts are retained only under ignored
`docs/evidence/issue-28-kumo-products/`. These are bounded software checks, not
scientific reproduction or browser evidence; root owns the combined full suite.

The mixed-catalog successor verifies `prod,kumo-prod` and `default,kumo-prod`
in local, global and meta modes, retaining both variants' child eligibility and
realization replay. Explicit, native and parent graph checks exercise mixed
catalog replay, generation and rejection without advancing any of four streams.
Swapping only a selected product variant is rejected against its child plans.
Evidence is ignored under `docs/evidence/issue-28-kumo-mixtures/`.
