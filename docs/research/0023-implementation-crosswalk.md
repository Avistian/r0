# #23 implementation crosswalk

The [versioned audit](0023-prior-audit.md) accounts for source claims. This crosswalk accounts for executable r0 coverage. Neither asserts exhaustive discovery of every available paper. All downstream usefulness is **NOT_MEASURED**; every full peer training-recipe reproduction remains **NOT_IMPLEMENTED**. Source availability is not equivalent to recipe or checkpoint provenance.

| Catalog | Executable #23 subset | Remaining implementation / owner |
| --- | --- | --- |
| C01–C02 graphs | Tiny chain/fork/collider with hidden variables | Layered, preferential/redirection, Cauchy, motif assembly, rewrites, disconnected graph controls: #28 |
| C03 standalone neural | 1–2 layers, width 4–8, tanh/ReLU, Gaussian weights | Broader architecture/weight/bias/sparsity hyperpriors: #28 |
| C04–C05 graph functions | Scalar tanh equations | Neural node/edge variants, masks, domain-safe nonlinearities, multi-parent composition: #28 |
| C06 smooth functions | Exact finite-support RBF joint draw | Matérn, kernel mixtures/hyperpriors, coherent arbitrary-coordinate extension, RFF: #27/#28 |
| C07 trees | Direct binary splits and leaf draws | Fitted-on-noise DT/XGBoost/ExtraTrees/GBDT/RF, symmetric trees, forests: #29 |
| C08 analytic functions | Sparse linear, product/parity/gate; tree piecewise control | Quadratic, prototypes, EM plateaus, periodic/spatial functions: #27–29 |
| C09 dependent rows | None | Convolution and VAR with explicit order/time semantics: #39 |
| C10 marginals/internal randomness | Normal roots, Gaussian SCM innovations | Uniform/mixed/prototype roots, hidden-dimensional alternatives: #28/#30 |
| C11 observation noise | Gaussian sd 0/1; uniform class replacement p 0/0.2 | Heteroscedastic/heavy tails, censoring, bounded targets: #30 |
| C12 target construction | Fixed independent thresholds, permuted class names, complete vocabulary | Broader independent adapter priors and categorical nodes: #30 |
| C13–C14 observations | No transforms or missingness | Categoricals, warping, quantization, MCAR/MAR/MNAR: #30 |
| C15 selection | Validity versus deliberate noise filter | Graph/solvability, bootstrap and descriptor-based selection: #30 |
| C16 mixtures | Balanced diagnostic allocation only | Across-table and within-graph mixtures, learned diversity proposals: #30/#36 |
| C17 adaptive proposals | None; exact recipes unresolved in audit | Adversarial variants compatible with synthetic-only contract: #30/#36 |
| C18 hyperpriors | Tiny versioned diagnostic ranges | Shape/noise/graph/sparsity distributions and curricula: #30/#33/#36 |
| C19 binary-to-multiclass | Three declared task types only | Architecture-dependent alternative: #35/#36 |
| C20–C21 environments | Interaction rules only, no shift/intervention evidence | Subgroups, spuriousness, OOD, interventions, temporal drift: #28/#30/#39 |
| C22 relational | None | Schema/history/joins: #39–42 |
| C23 measurement | Construction, duplicate identity, proposal accounting | Independent challenges #24; fixed-protocol learner evidence #36 |
| C24 real-data priors | Excluded | Incompatible with current synthetic-only training contract |
| C25 delivery | Serial local diagnostic records | Streaming/shards/cache/worker seeds: #33 |
| C26 learner interfaces | Existing restricted predictor view | Tokenization, heads and learner architecture: #31/#32/#35 |
| C27 newer specialized priors | None; algorithms unresolved | Extract/verify mechanisms before implementation: #28/#30/#36 |
| C28 guide recommendations | No production choice adopted | Treat as hypotheses; sources and experiments required: #27–30/#36 |

For local measured coverage use the retained walkthrough: 72 default proposals across six families/two noise regimes/three task types. The fixed default observes two of three SCM motifs; the fork is a separate labeled example. No count of catalog entries should be reported as a count of implemented generators.

Before calling the broader library complete, each source-backed option above needs its own explicit accepted/rejected/deferred disposition, resolved sampler parameters, diagnostic examples and appropriate comparison. New source discoveries extend the catalog; they do not silently change a production mixture. #23 establishes that workflow and its small reference implementations.
