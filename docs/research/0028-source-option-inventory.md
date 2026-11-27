# Issue #28 finite source-option inventory

Prepared 2026-11-26 against the [accepted specification](../specs/0028-causal-network-generators.md),
[initial coverage audit](0028-paper-coverage-audit.md), [#23 audit](0023-prior-audit.md)
and [#23 crosswalk](0023-implementation-crosswalk.md). This document is the index
for the finite implementation inventory, not an implementation acceptance report.

## Pre-implementation gate

**READY for implementation of the enumerated documented options.** The earlier
P1/P2/I1/M1/L1/F1/PFN/O-prior/TAFFY generator sections and relevant appendices have
been extracted. The [newer-source check](0028-newer-source-check.md) covers
I2/M2/L2/A1/K1/G1/P35; P25 §3 and P3 §2.5 were checked separately. Paper/source
contradictions have named records. No known documented mechanism in these checked
generator sections is intentionally omitted. This is bounded corpus extraction,
not a claim of an exhaustive search over all literature or upstream repositories.

Unspecified numeric ranges, probabilities, calibration constants and recipe caps
are delegated construction decisions: record them as independently reconstructed
r0 recipes and review them before acceptance. They do not block implementation of
the disclosed algorithm. Unnamed algorithms remain **UNDISCLOSED**; a generic
replacement cannot establish their coverage. Consequently **full peer-prior or
checkpoint reproduction remains UNKNOWN**, and an unqualified full-paper-coverage
claim still requires resolution or the specification's explicit disposition of
those disclosure gaps. This gate does not declare those gaps resolved.

There are **583 source/extension/disposition records**, plus **140 specification
anchors**. Counts include the same operation's occurrence in different sources,
source variants, sibling-ticket interfaces and unresolved disclosures; they are
**not counts of unique algorithms or implemented generators**. Specification
anchors preserve the previous grouped checklist; atomic records supply the
implementation units. The finite scope is their union, with aliases allowed only
when the semantics and verification truly coincide.

The initial gate contained 581 source records. Two additional source-code variants
were recorded during implementation: `I2.converter.02.graph` and
`I2.converter.03.graph` distinguish actual historical graph alias behavior from
reconstructed tuple intent. The [source audit](0028-wrapper-source-laws.md) and
[sampling follow-up](0028-node-pipeline-sampling-laws.md) retain the distinction.
This addition expands traceability; it does not mark either complete.

## Machine-readable inventory

| File | Records | Contents begin/end |
|---|---:|---|
| [01](0028-source-options-01.json) | 93 | P1 through I1 |
| [02](0028-source-options-02.json) | 104 | I1 through O-prior |
| [03](0028-source-options-03.json) | 95 | O-prior through activation catalog |
| [04](0028-source-options-04.json) | 91 | Activation catalog and specification anchors |
| [05](0028-source-options-05.json) | 71 | Specification anchors and accepted r0 extensions |
| [06](0028-source-options-06.json) | 103 | Pinned I2 mechanisms, matrices, roots, node processing |
| [07](0028-source-options-07.json) | 110 | I2 converters/activations through A1 |
| [08](0028-source-options-08.json) | 56 | K1/G1/P35/P25/P3 and adjacent corpus boundaries |

Each record has a stable `id`, `option`, `source`, `disposition`, and `notes`.
Source keys resolve through the register below and the newer-source check.
`REQUIRED_28` includes source variants requiring named safe adaptations;
`SHARED_27_28` requires #28 composition/replay using the shared smooth-function
interface. `INTERFACE_*` identifies the actual sibling owner, not completed work.
`SPEC_REQUIREMENT_ANCHOR` is traceability only and inherits its listed ownership.
`OUTSIDE_28` retains excluded/adjacent corpus entries with boundaries below.

All records inherit this initial evidence state:

- Implementation entry point / recipe: **NOT_IMPLEMENTED** in this research gate.
- Independent worked calculation, replay, intervention and typed-shape proof:
  **PENDING**.
- Saved valid/flagged/failed examples and accounting: **PENDING**.
- Local browser, Colab, CPU/GPU and resource measurements: **NOT_RUN**.
- Downstream usefulness: **NOT_MEASURED**; checkpoint association: **UNKNOWN**.

An implementation coverage ledger must link every required ID to the delivered
entry point, versioned recipe, supported shapes/types/compositions, adaptation,
worked example, replay artifact, diagnostics, inspection exports and environment
evidence. Broad source-family labels cannot close atomic records. Numeric bounds
and valid combinations are per-recipe obligations, not an implied Cartesian product.

## Source register

Version dates refer to the linked revision as shown by the publication, not the
research date. Repository source pins do not establish historical training use.

| Key | Publication / checked generator locus | Revision date |
|---|---|---|
| P1 | [TabPFN-v1 2207.01848v6](https://arxiv.org/html/2207.01848v6), §§4.3–4.5, C.1–C.2, E.4/Table5 | 2023-09-16 |
| P2 | [TabPFN-v2 Nature](https://www.nature.com/articles/s41586-024-08328-6), Synthetic data and Methods graph/mapping/root/postprocessing sections | 2025-01-08 |
| I1 | [TabICL 2502.05564v1](https://arxiv.org/html/2502.05564v1), §4, B.1–B.2, D.1 | 2025-02-08 |
| M1 | [MITRA 2510.21204v1](https://arxiv.org/html/2510.21204v1), §3, A.2–A.3, Algorithms2–5 | 2025-10-24 |
| L1 | [LimiX 2509.03505v1](https://arxiv.org/html/2509.03505v1), §4.1–4.3 | 2025-09-03 |
| F1 | [TabForestPFN 2405.13396v1](https://arxiv.org/html/2405.13396v1), §4.1/Table1, §5.5, A.9 | 2024-05-22 |
| PFN | [PFN 2112.10510v1](https://arxiv.org/html/2112.10510v1), §§5–6, E/F/Table5 | 2021-12-20 |
| OP | [O-prior 2605.18971v1](https://arxiv.org/html/2605.18971v1), §§2.1–2.4 | 2026-05-18 |
| TA | [TAFFY 2610.07559v1](https://arxiv.org/html/2610.07559v1), §4.2, A.1–A.4/Table3 | 2026-10-06 |
| P25 | [TabPFN-2.5 2511.08667v1](https://arxiv.org/html/2511.08667v1), §3 Data | 2025-11-11 |
| P3 | [TabPFN-3 2605.13986v1](https://arxiv.org/html/2605.13986v1), §2.5 | 2026-05-13 |
| I2 | [TabICLv2 2602.11139v1](https://arxiv.org/html/2602.11139v1), §5/Appendix E | 2026-02-11 |
| M2 | [MITRA-v2 2609.04540v1](https://arxiv.org/html/2609.04540v1), §§2.2–2.3 | 2026-09-03 |
| L2 | [LimiX-2 2609.17488v1](https://arxiv.org/html/2609.17488v1), §§4.1–4.5 | 2026-09-15 |
| A1 | [TabAbax 2611.00562v1](https://arxiv.org/html/2611.00562v1), §3.3 | 2026-10-31 |
| K1 | [Kumo Tabular 2611.23001v1](https://arxiv.org/html/2611.23001v1), Appendix A | 2026-11-17 |
| G1 | [TabFM 2609.37959v1](https://arxiv.org/html/2609.37959v1), §3.2 | 2026-09-29 |
| P35 | [TabPFN-3.5 2609.17895v1](https://arxiv.org/html/2609.17895v1), §3.3 | 2026-09-15 |

Pinned source keys:

- **P1C**: [`automl/tabpfn-v1-prior@778f0d0eb10f35ca85964fbc27ecb8baee9146a4`, `tabpfn_prior/priors/mlp.py`](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py).
- **I1A/I1M/I1T/I1C**: `soda-inria/tabicl@9530bd65a7b806f6ad4f9f4335af745577e9d3cc`, respectively
  [`_activations.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_activations.py),
  [`_mlp_scm.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_mlp_scm.py),
  [`_tree_scm.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py),
  [`_prior_config.py`](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_prior_config.py).
- **I2S**: same TabICL commit, [`src/tabicl/prior/graph_lib/`](https://github.com/soda-inria/tabicl/tree/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib), per-record filename. Additional exact links and inspections are in the newer-source check.
- **F1C**: [`FelixdenBreejen/TabForestPFN@53114795d3c96f87348a7ccbb675665e9d3e5243`, `tabularbench/data/synthetic_generator_forest.py`](https://github.com/FelixdenBreejen/TabForestPFN/blob/53114795d3c96f87348a7ccbb675665e9d3e5243/tabularbench/data/synthetic_generator_forest.py).

No new generator source pin was established for P2/M1/L1/P25/P3/M2/L2/A1/K1/G1/P35/OP/TA.
P2 explicitly states its synthetic pretraining generator was not released with the
models. L1's previously inspected inference README is not generator evidence.
The above source files were read, never executed or copied into r0 implementation.

## Required distinctions and reconstruction decisions

**P1:** C.1 places noise inside the activation; pinned code uses activation, linear
mapping, then noise. Its individual-mask scaling uses `1-p**e`, whereas block
scaling uses `keep**e`. Preserve source and conventional variance-preserving
variants. Paper “Multivariate Distribution” is not evidence for the source's
multinomial categorical branch. [P1 paper](https://arxiv.org/html/2207.01848v6#A3),
[P1 source](https://github.com/automl/tabpfn-v1-prior/blob/778f0d0eb10f35ca85964fbc27ecb8baee9146a4/tabpfn_prior/priors/mlp.py).

**P2:** a Gamma law is not bounded to valid redirection probabilities. Its
`Multinomial(beta)` prototype-weight notation lacks complete count/probability
semantics. Declare a bounded probability law and normalized prototype-weight
construction as r0 reconstruction, retaining these disclosure gaps. Normalizing
or clipping silently would falsely assert fidelity. The redirection family is
not interchangeable with arbitrary preferential attachment. [P2 Methods](https://www.nature.com/articles/s41586-024-08328-6).

**I1:** preserve paper versus code Fourier exponent laws. The tree source refits
on every input batch, while r0 requires a mechanism frozen using independent
calibration. Its noncausal default, extra DT/ExtraTrees/RF variants and corrected
variance scaling are source options, not proof of the paper's historical recipe.
[I1 Appendix B](https://arxiv.org/html/2502.05564v1#A2),
[I1 source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/_tree_scm.py).

**M1:** prose describes a fixed sampled forest; Algorithm5 resamples inside the
row loop. Prose randomizes the first class leaf; Algorithm4 uses a zero-initialized
counter and ignores that draw. Both receive named records. Per-row forests need
explicit stochastic-mechanism/exogenous-draw semantics, never a false fixed-function
claim. [M1 Appendix A](https://arxiv.org/html/2510.21204v1#A1).

**F1:** source uses `max_features='sqrt'`, exclusive integer upper bounds and a
base-size default1000; the paper reference is1024. The `n_estimators` signature
argument does not create a forest. [F1 source](https://github.com/FelixdenBreejen/TabForestPFN/blob/53114795d3c96f87348a7ccbb675665e9d3e5243/tabularbench/data/synthetic_generator_forest.py).

**I2/K1:** retain both kernel-point laws, both EM-width descriptions, paper/code
interval and argmax semantics, historical hidden-state versus corrected readout
warps, and restricted versus broadened product children. Exact findings and source
links are in the [newer-source check](0028-newer-source-check.md#discrepancies-requiring-named-variants).

**Calibration:** generated-batch standardization, prototypes, tree thresholds,
class quantiles and TAFFY base-table bounds must become frozen independent
calibration artifacts for training-ready r0. Batch-dependent references are
separately named and cannot claim query-isolated semantics. Valid disconnected,
saturated or weak tasks remain retained despite peer pruning/filtering rules.
This paragraph is the accepted r0 adaptation, not an attribution to a paper.

## Boundaries and remaining disclosures

- **#27:** exact GPs, spectral functions and kernels require structural composition
  and trace/replay in #28. I1's Fourier activation and I2's two spectral choices
  are not automatically covered by an existing RBF implementation.
- **#29:** standalone DT/XGBoost/RF/ExtraTrees/GBDT/direct-forest portfolios remain
  there. Their reusable structural functions are required now in #28, including
  fitted-on-random-data classifier/regressor distinctions.
- **#30:** observation transforms, missingness, measurement noise, outer mixtures,
  learned selection, curricula and production filtering. #28 supplies typed
  state/readouts, descriptors, retained mechanisms and explicit interface hooks.
  Historical converters that alter hidden structural state are #28 variants.
- **#39:** sample-axis convolution, VAR, dynamic SCM and time/group shifts. O-prior
  explicitly describes feature-index convolution, which is required in #28;
  L1 explicitly describes sample-axis convolution, which belongs to #39.
  Unspecified CNN axes remain labeled; one axis cannot stand in for the other.
- **Adjacent corpus:** APT/RAD-TFM adaptive selection belongs to #30/#36; Two Is
  Enough is learner architecture; relational curriculum belongs to #39–42;
  coverage/fixed-protocol studies guide #24/#36 evaluation; streaming belongs to
  #33. Their #23 source links remain in the audit. Real-data pretraining and
  TabPFN-derived generator tuning/distillation are excluded by r0 policy.

The `UNDISCLOSED` rows retain P25/P35 unnamed expansions, P3 unnamed graph/combiner/
categorical changes, A1 unnamed richer families/frequency laws, and L2 unnamed
kernel/root catalogs and CNN axis. G1's disclosed dual-target behavior is required
via a named r0 reconstruction; exact peer internals are unknown. Named periodic,
spatial, kernel and piecewise families are required even when peer parameters are
undisclosed. Neither a generic implementation nor a diagram closes a missing
undisclosed algorithm.

No experiments, model training, cloud calls, dataset/weight downloads or source
implementation changes were performed for this inventory. File-size/JSON checks
only validate the inventory artifact, not scientific correctness or coverage of
implemented code. Independent review remains required.
