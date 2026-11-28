# Issue #28: initial paper-option coverage audit

Prepared 2026-11-26. Read-only investigation; no implementation or experiments.
This is a completion checklist, not a claim that every source has been exhaustively
verified. The accepted requirement is to cover every documented option relevant to
#28 in the agreed paper set. An unimplemented relevant option remains outstanding;
an author cannot silently mark it deferred. Neighbor-ticket ownership must identify
the interface and explain why it is outside #28. Undisclosed recipes remain unknown.

### Required coverage record

Each option needs: stable ID; paper version and section; source revision/file where
available; precise construction; implementation entry point and versioned recipe;
reference versus r0 adaptation; supported shapes/types/compositions; independent
worked example; replay evidence; diagnostics; local/Colab inspection evidence;
CPU/GPU/resource evidence; and explicit remaining gaps. All options below start
as implementation UNVERIFIED in this audit. Paper availability is not evidence
of checkpoint-linked reproduction or downstream benefit.

Disposition vocabulary: REQUIRED #28; SHARED WITH #27/#29; INTERFACE #30/#39;
UNDISCLOSED; SOURCE DISCREPANCY. 'Deferred' is not an author-selected substitute
for REQUIRED #28. Incompatibilities must be explicit and reviewed rather than
resolved by silently shrinking the accepted inventory.

### Source register

Existing catalog rows, preserved separately rather than collapsing model versions:

| ID | Primary source | Audit status |
|---|---|---|
| P1 | [TabPFN-v1, 2207.01848v6](https://arxiv.org/html/2207.01848v6), §§4.3–4.5 and Appendix C | Existing #23 audit; pinned MLP source below |
| P2 | [TabPFN-v2, Nature](https://www.nature.com/articles/s41586-024-08328-6), Methods | Existing #23 audit; direct page fetch unavailable in this pass |
| P25 | [TabPFN-2.5, 2511.08667v1](https://arxiv.org/html/2511.08667v1), §3 | Broadening reported; exact new mechanisms unresolved |
| P3 | [TabPFN-3, 2605.13986v1](https://arxiv.org/html/2605.13986v1), §2.5 | Existing audit; version deliberately retained |
| I1 | [TabICL, 2502.05564v1](https://arxiv.org/html/2502.05564v1), §4.1 and Appendices B/D | Existing #23 audit |
| I2 | [TabICLv2, 2602.11139v1](https://arxiv.org/html/2602.11139v1), §5 and Appendix E | Detailed paper and selected pinned source inspected |
| M1 | [MITRA, 2510.21204v1](https://arxiv.org/html/2510.21204v1), §3 and A.2 | Existing audit; full paper available |
| M2 | [MITRA-v2, 2609.04540v1](https://arxiv.org/html/2609.04540v1), §2.3 | Detailed mechanism section inspected |
| L1 | [LimiX, 2509.03505v1](https://arxiv.org/html/2509.03505v1), §4 | Existing #23 audit |
| L2 | [LimiX-2, 2609.17488v1](https://arxiv.org/html/2609.17488v1), §4 | Detailed generation section inspected |
| F1 | [TabForestPFN, 2405.13396v1](https://arxiv.org/html/2405.13396v1), §§4.1/5 | Original revision/title retained |
| A1 | [TabAbax, 2611.00562v1](https://arxiv.org/html/2611.00562v1), §3.3 | Full prior section inspected |
| K1 | [Kumo Tabular, 2611.23001v1](https://arxiv.org/html/2611.23001v1), §3 and Appendix A | Concrete changes now available |
| G1 | [TabFM, 2609.37959v1](https://arxiv.org/html/2609.37959v1), §3.2 | Broad SCM description; no new exact mechanism inventory there |
| P35 | [TabPFN-3.5, 2609.17895v1](https://arxiv.org/html/2609.17895v1), §3.3 | Broad diversity/high-cardinality/grouped extensions; algorithms undisclosed |

Supporting sources already in the audit: [PFN, 2112.10510v1](https://arxiv.org/html/2112.10510v1),
[O-prior, 2605.18971v1](https://arxiv.org/html/2605.18971v1),
[TAFFY, 2610.07559v1](https://arxiv.org/html/2610.07559v1).
Their exact relevant algorithms still need an option-level extraction check.

Pinned source anchors:

- [P1 MLP sampler](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py).
- [I2 graph](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_graph.py).
- [I2 activations](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py).
- [I2 function settings](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py).
- [I2 pipeline limits](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/README.md).

### Option checklist

Each semicolon-delimited alternative below must become a separate traceable child
record before implementation is called complete. Rows group related choices for
readability, not to hide missing implementations behind one family label.

| ID | Required options or distinctions | Source / ownership |
|---|---|---|
| G01 | Layered sparse neural DAGs; observed/hidden variable selections; sparse connectivity | P1; REQUIRED |
| G02 | Growing network with redirection; potentially disjoint components | P2; REQUIRED. Ordinary preferential attachment is not automatically the same algorithm |
| G03 | Cauchy global/per-node connectivity propensities with forward DAG ordering | I2 pinned graph; REQUIRED. Cauchy here is topology, not observation noise |
| G04 | Chains; common-cause/fork motifs; colliders; recursive/hierarchical motif assembly | L1/L2, P1/P2; REQUIRED |
| G05 | Acyclic edge redirection; local-path replacement; node structural transformations | L2; REQUIRED. Exact rewrite distributions are not disclosed |
| G06 | Disconnected nuisance components; target disconnected from observed inputs; hidden ancestors | P2/K1; REQUIRED. Retain valid draws and diagnose |
| G07 | Chain-plus-skip topology; vector nodes; nonlinear path plus direct parent projections | M2; REQUIRED |
| G08 | Arbitrary forward Bernoulli DAGs; broad graph/network families accepted in interview | r0 extensions; REQUIRED, not falsely attributed to a peer |
| N01 | Standalone BNN/MLP distinct from neural SCM; architecture, weight, bias, sparsity hyperpriors | P1; REQUIRED |
| N02 | Dense; individual sparse; block sparse; skip; residual; irregular feedforward; gated subnetworks | P1 masks/M2 residual ideas plus accepted r0 extensions; REQUIRED |
| N03 | Optional activations before/after neural mappings; layer-specific choices; scalar/vector outputs | I2 and accepted broad scope; REQUIRED |
| W01 | Gaussian weights; feature-importance weighting; random-factor matrices; signed kernel matrices; activation-transformed matrices | I2; REQUIRED |
| W02 | Weight scales; fan-in conventions; bias distributions; per-edge projections; masks and blocks | P1/I2/M2; REQUIRED |
| W03 | Independent choices versus shared task preferences; correlated numerical hyperparameters; positive weight vectors | I2/K1; REQUIRED. Preserve their distinct constructions |
| A01 | tanh; leaky ReLU; ELU; identity; SELU; SiLU; ReLU; softplus; ReLU6; hardtanh | I2 activation source; REQUIRED |
| A02 | sign; Heaviside; Gaussian bump; exponential; bounded interval indicator; sine; square; absolute value | I2 activation source; REQUIRED |
| A03 | stabilized log-absolute; sigmoid; log-sigmoid; rounding; modulo | I2 activation source; REQUIRED |
| A04 | softmax across channels; maximum indicator; rank; argsort | I2 activation source; REQUIRED. These are vector operations, not scalar activations |
| A05 | positive powers of ReLU; signed absolute powers; reciprocal absolute powers; integer powers | I2 activation source; REQUIRED. Domains and sampler placement explicit |
| A06 | GP-derived activation/function option; numerical domain safeguards; activation shifts/scales | I1/I2; SHARED #27 with #28 composition/inspection |
| C01 | Concatenate parents then transform; transform each parent separately | I2; REQUIRED |
| C02 | Sum; product; maximum; logsumexp; average; weighted aggregation; neural aggregation | I2/L1/L2; REQUIRED |
| C03 | Softmax-weighted aggregation; nonlinear mechanism plus linearly projected parents | M2; REQUIRED |
| F01 | Linear/sparse; quadratic; GP/kernel; RFF; periodic; spatial; piecewise; multiplicative functions | I2/P3/L2/M2; SHARED #27/#28. Composition and trace support required |
| F02 | Nearest-prototype discretization; clustering/EM-like plateau assignments; distance/width/power variations | I2/K1; REQUIRED |
| F03 | Products over eligible child functions, including neural and cluster children | I2/K1; REQUIRED. Preserve source-specific restrictions and broader r0 recipes |
| T01 | Direct random split/threshold/leaf trees; symmetric/oblivious tree components | P2/I2/M1; REQUIRED reusable SCM components; standalone portfolio #29 |
| T02 | Trees/forests fitted on random data: DT, XGBoost, RF, ExtraTrees, GBDT | I1/F1/M1/M2; REQUIRED #28 structural components under the final all-paper-options requirement. Standalone portfolio #29. Distinct from direct random trees |
| R01 | Gaussian; uniform; mixed/prototype roots; bounded-ball/correlated roots; transformed roots | P2/I2; REQUIRED where structural, with existing #27 inputs reused |
| R02 | Internal innovations; hidden channels; broad accepted root/noise laws; added target noise separately recorded | P1/P2/I2 and accepted extensions; REQUIRED |
| O01 | Scalar/vector nodes; partially observed channels; features/target sharing nodes; hidden channels | I2/M2/A1; REQUIRED |
| O02 | Ancestor/descendant/common-cause/collider/disconnected observation profiles | Accepted role profiles, supported by graph families; REQUIRED |
| D01 | Mixed continuous/discrete endogenous variables; native categorical targets; shared sigmoid/softmax adapters | P2/I2/A1 and accepted scope; REQUIRED typed schema throughout |
| D02 | Neighbor/prototype-based categories; stochastic softmax categories; node readout versus state transformation | I2; REQUIRED endogenous mechanisms. Calibration uses accepted independent fixed sample |
| D03 | Categorical-state representations: unchanged latent vector; repeated index; selected center; transformed center; sampled category embedding | I2; REQUIRED distinction for effects on children |
| I01 | Hard interventions; soft interventions; matched-noise and fresh-population comparisons | TAFFY plus accepted scope; REQUIRED within declared non-temporal SCM contract |
| I02 | Graph edits; mechanism edits; channel edits; preserved/new identity and fixed calibration semantics | Accepted workbench scope; REQUIRED replayable versions |

### Source changes that must not disappear inside broad family names

Kumo uses shared per-table Dirichlet preferences, accepts disconnected target graphs
before a separate learnability stage, varies cluster-width spread with L1/L2 geometry,
and permits neural/cluster children in products. Its noisy measurement converters,
shared-noise column blocks and missingness belong to the #30 interface. [K1]

TabAbax retains the Cauchy/vector-node base, changes per-node function sampling,
checks categorical cardinality/one-hot agreement, uses varied class-frequency
distributions, and removes post-hoc target warping. Its exact expanded function list
is not disclosed in §3.3: do not invent it. [A1]

LimiX-2's graph rewrite distribution and detailed function hyperparameters remain
undisclosed even though the operations are documented. TabPFN-2.5/3.5 and TabFM
need explicit source-gap rows rather than invented new exact recipes. [L2/P25/P35/G1]

### Material paper/source discrepancies

Pinned I2 activation code uses an interval indicator for abs(x)<=1; the paper lists
an indicator on [0,1]. Code uses an is-maximum mask, allowing multiple maxima on
ties; the paper says one-hot argmax. Rank/argsort act along channels. Plain activation
sampling includes negative and integer powers, but the standard neural activation
sampler only selects fixed, ReLU-power and signed-power classes. Preserve named
paper/source variants; do not silently merge their semantics. [I2 activation source]

I2's random-factor matrix is not a draw with exactly prescribed singular values:
its factors are Gaussian rather than orthogonal. Its feature/activation calibration
uses data-dependent statistics. M2 also standardizes node outputs and uses target
quantiles. The accepted r0 independent fixed calibration is an adaptation that must
remain visible. Source graph pruning/learnability rejection must not overwrite the
user's accepted retention of valid weak/saturated/disconnected tasks.

### Explicit boundaries, not omissions

| Boundary | Required treatment |
|---|---|
| #27 smooth functions | Reuse public mechanisms; #28 must support their composition as structural functions and inspection. Existing implementation is not proof every paper variant is covered |
| #29 tree portfolio | Reusable random tree components in SCM are accepted #28 scope. Full standalone tree/ensemble portfolio remains #29. Fitted-on-synthetic-random-data structural components are also REQUIRED #28 under the final completeness condition; they are not equivalent to random splits |
| #30 observation/selection | Measurement warps, missingness, censoring, broad external observation effects, production filtering/mixtures remain there. Structural discrete states and native categorical targets are already #28 |
| #39 dependence/time | Row-axis convolution, VAR, dynamic SCMs and temporal/grouped shifts need row-order/time semantics. Feature/channel-axis transformations must not be mislabeled as row dependence |
| Learner architecture | Cell/row embeddings, attention, output heads and binary-only learner proposals are outside generator scope |
| Real-data priors | Excluded by synthetic-only training policy; retain explanation rather than count as missing synthetic mechanism |

### Completion and outstanding evidence

- Expand group rows into option records; verify every relevant section/appendix
  of every registered paper has a coverage row or justified boundary row.
- Finish primary extraction for P1/P2/I1/M1/L1/F1 and supporting O-prior/TAFFY;
  this bounded pass does not establish exhaustive option extraction from them.
- Check version changes explicitly; newer papers do not overwrite earlier recipes.
- T02 is included as structural components under the final all-paper-options
  condition; validate its shared interface with #29 without duplicating the standalone portfolio.
- Preserve unknown sampler details. Choose independent explicit reconstruction
  settings under the accepted author/scientific-review delegation, label them, and
  verify them; do not manufacture source fidelity.
- For every required option, retain accepted and numerically failed cases where
  applicable, diagnostics, construction checks, calibration/query isolation,
  replay, intervention invariants, typed output and complete export evidence.
- Inspect all options through the shared local/Colab workbench; record actual
  target-environment CPU/GPU/resource proof separately from local tests.
- Broad ranges do not imply their Cartesian product. Unsupported combinations
  remain explicit without silent substitutions; material exclusions need review.
- Do not call #28 complete until all relevant documented options are mapped and
  implemented or covered by an explicitly accepted existing owner/interface.

Repository audit references: [catalog](https://github.com/Avistian/r0/blob/main/docs/research/0023-prior-audit.md),
[crosswalk](https://github.com/Avistian/r0/blob/main/docs/research/0023-implementation-crosswalk.md).
