# Generator prior audit, revision 1

2026-11-23 audit for [#23](https://github.com/Avistian/r0/issues/23), [spec](../specs/0023-generator-diagnostics.md), HTML guide SHA-256 `e8606a6e5d0bf7e2cc968910e9b256f445ac7cef429cc9feedd8a203c727a192`. Guide claims require primary evidence. No external code execution/copying, datasets/weights downloaded or training.

## Evidence contract

`PAPER`: exact linked publication revision. `SOURCE`: specified files/commit. Neither proves checkpoint provenance, which is `UNKNOWN` for every family here. `UNKNOWN` is an audit gap, not an absent capability. No full peer recipe reproduced. Coverage/usefulness: `NOT_MEASURED`; see artifacts for local results. Graphs, functions, observations, targets and sampling policies are separate axes. Component reconstruction is not checkpoint reproduction. No superiority claim.

## Versioned families

| Record | Primary evidence and material recipe distinction | Source/configuration limits; disposition |
| --- | --- | --- |
| P1: TabPFN v1 | [2207.01848v6, §§4.3–4.5, Appendix C][p1]: layered sparse neural SCM and standalone BNN, selected with equal probability; classification bins continuous targets using sampled bounds then permutes labels. | [MLP source][s1] confirms distinct causal/standalone readouts. Training invocation/checkpoint association unverified. Neural functions and SCMs → #28; adapters/mixtures → #30. |
| P2: TabPFN v2 | [Nature published 2025-01-08, Methods][p2]: growing-network-with-redirection graphs, possibly disjoint components; neural mappings, categorical prototype discretization, random tree rules and Gaussian edge noise. Normal/uniform/mixed root sampling; Kumaraswamy warping, quantization and MCAR masks; categorical-node classification versus continuous-node regression. | No implementation pin or complete configuration established. Graph/function choices → #28/#29; observations → #30. Four modules do not imply equal weights. |
| P25: TabPFN-2.5 | [2511.08667v1, §3][p25] reports broader synthetic distributions and larger task shapes. Real-TabPFN-2.5 is separately continued on real data. | New mechanisms/weights/code `UNKNOWN`; do not equate the revised recipe with P2. #28/#30 investigation; real-data path excluded by r0's synthetic-only contract. |
| P3: TabPFN-3 | [2605.13986v1, §2.5][p3] describes expanded graphs/combiners, categorical generation, high-frequency oscillators, spatial activations, many-class tasks, discrete-time dynamic SCMs and OOD tasks. | Algorithms/hyperpriors/proportions incomplete. Periodicity/spatial → #27/#28; categoricals/shift → #30; dynamic SCM → #39. Full generator release unverified. |
| I1: TabICL | [2502.05564v1, §4.1, Appendices B/D][i1]: 70% MLP-SCM / 30% tree-SCM; tree layers fit XGBoost against Gaussian targets; increasingly large-table curriculum. | [Pinned default config][si] exposes 0.7/0.3 and XGBoost, but also `is_causal=False`. Defaults are not historical training evidence. #28/#29/#30. |
| I2: TabICLv2 | [2602.11139v1, §5/Appendix E][i2]: new graph prior, eight function types (MLP, symmetric trees, discretization, GP, linear, quadratic, EM plateaus, products); multi-parent composition; hidden dimensions rather than node Gaussian noise; bootstrap ExtraTrees selection. | [Pinned README][sig] explicitly says graph sampling is not the complete dataset pipeline and documents compatibility options retaining unintended pretraining behavior. Detailed checkpoint linkage unverified. #27–30; do not replace I1 with I2 in a recipe comparison. |
| M1: MITRA | [2510.21204v1, §3, A.2][m1]: SCM plus tree priors; distinguishes estimators fitted on random data from directly sampled forests. Direct forests sample splits and leaf targets, aggregate by voting/mean. Selection considers performance, diversity and distinctiveness. | Paper inspected, generator repository revision/training invocation not established. #28/#29; mixture selection → #30/#36. Weights unverified. |
| M2: MITRA-v2 | [2609.04540v1, §§2.2–2.3][m2]: classification outer mixture is 0.35 base SCM, 0.35 hybrid SCM and 0.06 each DT, ExtraTrees, GBDT, RF, directly sampled RF. Hybrid DAG functions include MLP/tree/forest, convolution, RFF-GP and VAR. | Classification weights only; regression/checkpoint weights unverified. Distinguish outer table mixture from within-DAG mixing. #28–30; autoregression → #39. |
| L1: LimiX | [2509.03505v1, §4][l1]: hierarchical local causal structures; MLP, sample-axis convolution and directly constructed trees; average/weighted/neural aggregation; graph-aware and solvability-aware variable sampling; task adaptation. | [Pinned README][sl] is release/inference evidence, not generator code. Solvability/mixing parameters unverified. #28/#29/#30; ordered/sample-dependent convolution needs #39 semantics. |
| L2: LimiX-2 | [2609.17488v1, §4][l2]: recursive chain/confounding/collider motifs and acyclicity-preserving topology transformations; adds linear, kernel, piecewise, periodic and multiplicative functions; descriptor-based multiobjective task selection; stochastic observation transforms. | Selection details/hyperparameters/checkpoint trace `UNKNOWN`. #27–30/#39. LimiX-2M/1016 checkpoint names alone do not establish separate generator recipes. |
| F1: TabForestPFN | [2405.13396v1, §§4.1/5][f1] (this revision's title is “Why In-Context Learning Transformers are Tabular Data Classifiers”): forest data fit a decision tree on random data, generate fresh features and discretize predictions; TabForestPFN combines these with TabPFN data. | Paper revision deliberately fixed; later title in guide is not substituted silently. Generator code/config pin unverified. Distinct from direct split/leaf draws. #29/#30; fine-tuning is outside #23. |

## Inspected repository revisions

- `automl/tabpfn-v1-prior@778f0d0eb10f35ca85964fbc27ecb8baee9146a4`: read `tabpfn_prior/priors/mlp.py` and `tabpfn_prior/tabpfn_prior.py`. Causal versus standalone readouts differ; ordinary weight-mask scaling is not the usual variance-preserving formula. [Source][s1]
- `soda-inria/tabicl@9530bd65a7b806f6ad4f9f4335af745577e9d3cc`: read `_prior_config.py`, `_graph_scm.py`, `graph_lib/README.md` and tree listing. This is not exhaustive function-source inspection. [Config][si], [wrapper][siw], [README][sig]
- `limix-ldm-ai/LimiX@49a25fc190303041a3ce9dfd14a8c3234dd8b459`: read `README.md` only: release/inference evidence, not generator implementation. [README][sl]

Guide refs `7ccadda`/`2138b027a6`: training provenance unverified.

## Catalog / HTML accounting

`REP`: #23 representative; `LATER`: backlog; `UNKNOWN`: unverified; `EXCLUDE`: justified exclusion. Repetitions share IDs. Loci use visible chapters. Coverage/usefulness: `NOT_MEASURED`.

| ID / HTML locus | Option, evidence status and explicit disposition |
| --- | --- |
| C01 / 4.6, 4.17 | Layered sparse neural DAGs, arbitrary forward-ordered DAG proposal, chain/fork/collider motifs, hidden causes, disconnected nuisance components. P1/P2/L2 support related topology choices [p1], [p2], [l2]. REP small motifs; LATER #28. Forward-order Bernoulli DAG is the guide/r0 simplification, not an exact peer recipe. |
| C02 / 4.6, 4.17 | Preferential/redirection growth (P2), random Cauchy graphs (I2), hierarchical motif assembly and graph rewrites (L2): distinct alternatives, not synonyms for C01. PAPER [p2], [i2], [l2]; LATER #28, compare connectivity/depth/hidden-parent coverage. |
| C03 / 5.1–5.9 | Standalone random BNN/MLP, independently sampled weights, depth/width, biases, activation and sparsity; distinct from MLP inside SCM. PAPER/SOURCE [p1], [s1]. REP tiny standalone MLP; LATER #28 for broad hyperpriors and sparse/block networks. |
| C04 / 4.7, 5.5–5.6 | Node/edge neural functions: linear, tanh/ReLU, identity, sigmoid, abs, sine, log, square/power, smooth ReLU, rank, step and modulo options; weight-scale/fan-in conventions and block/individual masks change function distributions. PAPER/SOURCE [p2], [s1]. REP tanh/ReLU subset; LATER #28; validate domains/finiteness before adding log/power. |
| C05 / 4.7, 4.17 | Parent concatenation versus per-parent functions followed by sum/product/max/logsumexp (I2); average/weighted/neural aggregation (L1). PAPER [i2], [l1]. LATER #28; simple recorded equations only in REP SCM. |
| C06 / 5.4, 11.7, 12.1 | Joint GP draws; RBF and Matérn hyperpriors in the original PFN experiments [gp]; GP-derived activations in I1 [i1]; multivariate functions in I2 [i2]; RFF approximation in M2 [m2]. REP one-dimensional RBF; LATER #27. Exact GP, RFF and neural limits differ. |
| C07 / 4.7, 6 | Directly sampled splits, thresholds and leaf values; fitted-on-noise DT, XGBoost, RF, ExtraTrees, GBDT; symmetric/oblivious ensembles. PAPER [p2], [f1], [i1], [i2], [m1]. REP direct small tree only; LATER #29. Separate fit/sampling cost. |
| C08 / 4.7, 11.7 | Linear/sparse, multivariate quadratic, nearest-prototype discretization, EM-like plateaus, products, periodic/spatial functions. PAPER [i2], [p3]. REP sparse linear/products plus tree piecewise controls; LATER #27/#28/#29. Broader variants deferred. |
| C09 / 4.17, 9 | Convolution across samples and autoregressive/VAR mechanisms [l1], [m2]. LATER #39: require an explicit row-order/time contract, distinguish generated dependence from future leakage. Not an iid representative. |
| C10 / 4.8, 4.16 | Root normal/uniform/mixed distributions and prototype-induced row dependence [p2]; internal Gaussian noise [s1]; unobserved node dimensions inducing uncertainty [i2]. REP declared roots/internal SCM randomness; LATER #28/#30. Hidden causes need not confound every pair. |
| C11 / 4.8, 10.4 | Target noise, label flips, heteroscedastic/heavy-tail noise, censoring/bounded targets and high-noise stress [op]. REP #23's paired Gaussian sd 0/1 or uniform label replacement p 0/0.2; broader processes LATER #30. These are r0 demo settings, not common peer SNRs. |
| C12 / 4.9–4.10 | Random feature/target selection; continuous regression; thresholds/ranks, categorical target nodes, class-label permutations; incomplete observed vocabulary and imbalance [p1], [p2], [si]. REP declared independent thresholds and complete vocabulary; LATER #30. Guide whole-table quantiles violate r0’s adapter boundary. |
| C13 / 4.11 | Prototype categoricals/embeddings, quantization/cardinality, Kumaraswamy/monotone warping, skew/tails and correlated hyperpriors [p2], [i2], [l2]. LATER #30. Repeated guide entries duplicate this axis. `cat_prob=0.2` [si] is a config field, not a measured 20% categorical-column share. |
| C14 / 4.12 | MCAR [p2]; MAR driven by another column and self-value MNAR [op]; support-only imputation, mask metadata. LATER #30. O’Prior documents MAR/MNAR; support in other peers remains unverified. |
| C15 / 4.13, 7 | Hard nonfinite/schema/label failures versus valid weak/noisy tasks; graph independence and ExtraTrees-bootstrap selection [i2], graph-/solvability-aware selection [l1], descriptor selection [l2]. REP accounting + deliberately biased noise filter; LATER #30. No silent rejection/top-up. |
| C16 / 7.1–7.4 | Across-table prior mixtures versus within-graph mechanism mixtures; fixed/uniform/hand-weighted versus diversity/distinctiveness-driven proposals [p1], [i1], [m1], [m2]. REP balanced diagnostic allocations only; LATER #30/#36. Repeated mechanisms may be intentional. |
| C17 / 7.5 | Adaptive adversarial generator distribution: [APT 2502.04573v1][apt] and [RAD-TFM 2611.10230v1][rad] abstract-level support. LATER #30/#36; exact adversary implementation `UNKNOWN` here. Real-domain alignment is excluded from the synthetic training path pending a separately compatible design. |
| C18 / 8.1–8.5, 11.3 | Hyperpriors for rows, columns, class count, graph size, noise, sparsity; log-scale simplicity bias, stage-dependent table-size curricula [p1], [i1]. REP explicit tiny caps; LATER #30/#33/#36. Guide production caps not adopted. |
| C19 / 4.10, 11.5 | Binary-only training with context-defined multiclass output: [Two Is Enough 2610.27569v1][two], abstract claim. LATER #35/#36; architecture-dependent; retain r0 multiclass coverage. |
| C20 / 11.5 | OOD/support-query mismatch, spurious features, confounding, subgroups/interactions, marginal shift, temporal/seasonal drift [op]; dynamic SCM/spatial/high-frequency extensions [p3]. LATER #30/#39. Match causal/temporal controls; keep shared mechanisms and shifted observations distinguishable. |
| C21 / 11.5 | Intervention-created environments, hard/soft interventions and label/covariate/conditional shifts: [TAFFY 2610.07559v1, Appendix A][ta]. LATER #28/#30/#39; observational generation is not intervention evidence. |
| C22 / 8.4, 9 | Relational schemas, table joins, relational curriculum and history-conditioned tasks: [Curriculum Matters 2607.29120v1][rel]. LATER #39, #40–42 evaluate. Outside #23; flat SCMs do not demonstrate joins. |
| C23 / 10.1–10.5 | Descriptor coverage/density [coverage]; fixed-architecture/fixed-protocol prior comparisons [play]. REP generator accounting only; independent challenges #24 and downstream ablations #36. Structural similarity is not downstream utility; probes cannot silently select tasks. |
| C24 / 11.2 | Real-data pretraining [TabDPT 2410.18164v1][dpt], continued real pretraining [Real-TabPFN 2507.03971v1][real]. EXCLUDE from r0 training under AGENTS.md/ADR 0002. Contamination allegations in linked tracker issues are not established by this generator audit. |
| C25 / 11.4, 12.4 | Streaming, pre-generated shards, hybrid caching, worker seeds and replay are engineering options; LATER #33. Guide speed/cost is `NOT_MEASURED` for r0. [Local issue #33](https://github.com/Avistian/r0/issues/33) |
| C26 / 11.6, 2.4 | Cell/row-token design, regression bar heads, masking, query isolation: adjacent learner constraints; LATER #31/#32/#35, duplicate of learner backlog rather than a new generator family. [Local roadmap](https://github.com/Avistian/r0/issues/19) |
| C27 / 9, 11 | TabAbax [abax] and Kumo Tabular [kumo] abstracts establish changed/specialized priors full algorithms not extracted. `UNKNOWN` follow-up #28/#30/#36; new mechanisms unresolved. Kumo's row encoding is also C26. |
| C28 / 1–3, 5.4–5.6, 13 | Bayesian derivations, GP limits, glossary and teaching simulations: background, not extra recipes. Existing-component duplicates C03/C06; guide plot values and toy speed measurements are not r0 measurements. Wide limits do not identify finite sparse priors. |

## Component reconstruction and extension

**Published component, not checkpoint reproduction.** [PFN 2112.10510v1, experimental details][gp] specifies zero-mean RBF, length scale 0.6, output scale 1, covariance diagonal increment 1e-4 (visual examples change length scale). Independent r0 construction: `K_ij=variance*exp(-(x_i-x_j)^2/(2*length_scale^2))`; sample standard normal `z`, compute `f=Lz`, `LLᵀ=K+diagonal*I`. Retain coordinates, K, L, z and parameters separately from added target noise. Check covariance/factorization and a hand-derived two-point calculation independently. Planned r0 boundary: eight mechanism-sampled coordinates uniform[-2,2], retained joint function values, observations resampling that finite domain; same kernel parameters with an explicit modeled 1e-4 diagonal component. This finite-domain distribution is a deliberate restriction, not the published full experiment. Kernel identity alone is not function identity. New arbitrary coordinates require coherent conditional extension, deferred to #27; #22 later checks predictive posteriors.

**r0 hypothesis: gated sparse interaction.** #23 uses `f(x)=product(x[1:degree]) if x0>0 else -1`, degree 2–3. A richer signed linear-plus-gated rule with sampled threshold remains proposed for #28/#29. Neither is a peer claim. Verify gate states, exact threshold and nuisance invariance. Trace C08/C20 to configuration/card/coverage; #36 tests utility.

## Falsifiable backlog

Ranked by evidence, relevance, complementarity and cost; benefits remain hypotheses. Initial comparisons: ≤72 tiny CPU proposals/arm, no learners/cloud. Later #36 requires separate authorization, frozen evaluation and matched compute.

| Rank / owner | Comparison, risk and promotion/rejection criterion |
| --- | --- |
| 1 / #27/#22 | RBF joint draws versus independent point noise: establish smooth-function coverage. Risk: conflating jitter/noise or kernel/function identity. Promote after independent covariance/factorization and replay checks; reject failed accounting. |
| 2 / #28/#29/#36 | Gated sparse interaction versus matched linear/product/tree controls: hypothesis of added localized interactions. Risk: rare gates or duplicated coverage. Require both gate states, degree coverage, hand-derived labels and nuisance invariance; reject the coverage hypothesis if controls already span the declared cells. Utility awaits held-out comparison. |
| 3 / #28/#30 | Motif/hidden-cause versus layered graphs at fixed node/function budgets: hypothesis of broader dependencies. Risk: disconnected targets or excess uncertainty. Require verified ancestry/observability cells and retained selection losses. Promote only demonstrated new coverage. |
| 4 / #29/#33 | Direct versus fitted-on-noise trees and symmetric ensembles [m1], [i1], [i2]: hypothesis of complementary boundaries/cost. Risk: costly fits or redundant axis alignment. Record paths, elapsed time and failures at fixed depth/shape; promote distinct coverage within the cost cap. |
| 5 / #30/#36 | Unfiltered versus noise/solvability-filtered identical proposals, then observation transforms [i2], [op]. Risk: losing valid sparse/high-noise cells. Require complete before/after counts/reasons with no top-up; reject unexplained accounting. Learner usefulness is separate. |
| 6 / #30/#39 | Missingness, shifts, dynamic SCM and relational curriculum [op], [p3], [rel]. Risk: future leakage and ambiguous time semantics. Defer until #39 invariants; one mechanism/config at a time with paired history controls. Reject future-row use. |

Retain unselected C01–C28 options. #36 must isolate shape/noise/compute, filtering, architecture and evaluation-overlap explanations.

## Sources

[p1]: https://arxiv.org/html/2207.01848v6
[p2]: https://www.nature.com/articles/s41586-024-08328-6
[p25]: https://arxiv.org/html/2511.08667v1
[p3]: https://arxiv.org/html/2605.13986v1
[i1]: https://arxiv.org/html/2502.05564v1
[i2]: https://arxiv.org/html/2602.11139v1
[m1]: https://arxiv.org/html/2510.21204v1
[m2]: https://arxiv.org/html/2609.04540v1
[l1]: https://arxiv.org/html/2509.03505v1
[l2]: https://arxiv.org/html/2609.17488v1
[f1]: https://arxiv.org/html/2405.13396v1
[gp]: https://arxiv.org/html/2112.10510v1
[op]: https://arxiv.org/html/2605.18971v1
[ta]: https://arxiv.org/html/2610.07559v1
[rel]: https://arxiv.org/html/2607.29120v1
[apt]: https://arxiv.org/abs/2502.04573v1
[rad]: https://arxiv.org/abs/2611.10230v1
[two]: https://arxiv.org/abs/2610.27569v1
[coverage]: https://arxiv.org/abs/2609.06912v1
[play]: https://arxiv.org/abs/2606.29241v1
[dpt]: https://arxiv.org/abs/2410.18164v1
[real]: https://arxiv.org/abs/2507.03971v1
[abax]: https://arxiv.org/abs/2611.00562v1
[kumo]: https://arxiv.org/abs/2611.23001v1
[s1]: https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py
[si]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_prior_config.py
[sig]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/README.md
[siw]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_graph_scm.py
[sl]: https://github.com/limix-ldm-ai/LimiX/blob/49a25fc190303041a3ce9dfd14a8c3234dd8b459/README.md
