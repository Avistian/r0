# Generator capability comparison for #27

Source review: **2026-11-25**. Scope: [accepted #27 specification](../specs/0027-training-ready-generators.md).
r0 implementation inspected at `cd57d10390338c7db261023aca6c86d4e38393a0`;
paper revisions and the external source commit below are fixed for this comparison.
The [#23 prior audit](0023-prior-audit.md) is historical context; the four papers
and the selected TabICLv2 source files were read again for this note.

`PAPER` describes disclosed mechanisms; `SOURCE` establishes the inspected code;
`ADAPTED` means r0 preserves selected settings with different implementations.
None establishes reproduction of a complete competing prior or its trained
checkpoint. `UNKNOWN` means not established by this review, not absent.
Predictive benefit and relative model performance are **NOT_MEASURED** here.

## Published mechanisms and corresponding r0 capabilities

| Model and fixed primary source | Disclosed mechanisms relevant to this comparison | r0 correspondence and boundary |
| --- | --- | --- |
| **TabICLv2 — PAPER**, [2602.11139v1, §5 and Appendix E][tabicl] | A DAG propagates random functions: MLPs, symmetric tree ensembles, nearest-center discretization, GP functions approximated with random Fourier features (RFF), linear, quadratic, EM-inspired plateaus and products. Multi-parent aggregation includes sum, product, maximum and logsumexp. Hidden dimensions introduce uncertainty; categorical converters and filtering are separate steps. | Linear, quadratic, Fourier and analytic compositions overlap at the function level. Selected source settings are adapted below. r0 radial functions do not reproduce EM assignment; independent inputs and Gaussian observation noise do not reproduce this causal observation pipeline. [Public API][api], [profiles][profiles]. |
| **TabPFN-3 — PAPER**, [2605.13986v1, §2.5][tabpfn] | An SCM with expanded graph samplers and combiners, revised categorical generation, high-frequency sinusoidal activations, spatial activations, many-class tasks, discrete-time dynamic SCMs and distribution-shift tasks. The section describes capability categories rather than an executable sampler with complete parameter distributions or mixture weights. | Periodic/sinusoidal functions, radial functions and spatially varying kernels provide related mathematical building blocks. They do not reproduce the disclosed spatial, temporal or shift prior. No equivalent numerical composition or frequency limits are established by this section. [Profiles][profiles], [kernels][kernels]. |
| **LimiX-2 — PAPER**, [2609.17488v1, §4][limix] | Hierarchical causal motifs and graph transformations; MLP, CNN and tree edge functions plus linear, kernel, piecewise, periodic and multiplicative functions. Functions compose; parent aggregation includes averaging, weighting and neural aggregation. Task selection uses structural/statistical descriptors. Observation transforms compose, and classification discretizes continuous targets. | Linear, kernel, periodic and multiplicative r0 functions correspond to categories, not the complete SCM. r0 uses explicit sigmoid/softmax target adapters, not this discretization procedure. Exact function hyperpriors, composition caps, selection thresholds and mixture probabilities are not established by §4. [Profiles][profiles], [compositions][compositions], [targets][targets]. |
| **MITRA-v2 — PAPER**, [2609.04540v1, §§2.2–2.4][mitra] | The classification outer mixture assigns 0.35 each to base and Hybrid SCMs, and 0.06 each to five tree generators. Within Hybrid SCM, MLPs, trees/forests, convolutions, RFF-GPs and vector autoregression populate a 3–7-node DAG with residual aggregations. Classification configuration: 160–5,120 support rows, 1,280 query rows, 1–50 features and up to 10 classes; categorical transformations and quantile labels are included. | r0 RFF and compositions overlap with components, but are not Hybrid SCM reproduction. Outer table selection differs from composing functions within one task. The disclosed classification weights are not adopted as r0 production weights, nor assumed to establish regression-checkpoint weights. [Profiles][profiles], [compositions][compositions]. |

These are scoped summaries, not exhaustive inventories. The absence of a mechanism
from a paper's description is not evidence that its generator cannot express it.
Published shape settings describe those configurations, not universal model or
generator ceilings. Complete checkpoint-linked sampler configurations, realized
training mixtures and training-task provenance remain **UNKNOWN** in this review;
the explicitly reported MITRA classification weights above remain known paper facts.

## Independently implemented TabICLv2 adaptations

The external **SOURCE** pin is
`soda-inria/tabicl@9530bd65a7b806f6ad4f9f4335af745577e9d3cc`.
r0's [`tabiclv2-adapted-v1`][profiles] and [selection implementation][sampler]
preserve only the settings identified below. Inputs are standard Gaussian;
calibration, coefficients, targets and random streams follow r0's definitions.

| Setting in pinned source | r0 adaptation and limits |
| --- | --- |
| Integer-power activation samples exponents 2–5. [Activation source][activation] | `power` raises one sampled normalized linear projection to the selected power. Projection and calibration are adaptations. This is not a ceiling on the degree of a composed function. |
| Quadratic function selects at most 20 inputs and includes affine terms. [Function source][function] | `quadratic` rejects more than 20 active inputs and uses r0's rank-20 factor construction. **Twenty input dimensions in the source is not a rank-20 source claim**; random-matrix distributions are not reproduced. |
| GP approximation defaults to 256 frequencies with a generalized random spectrum. [Function source][function] | `rff` preserves 256 frequencies but uses r0's Gaussian spectrum and sampled length scale. It is an approximation with a different kernel distribution, not an exact GP or the source GP sampler. |
| Product has two children; its restricted child pool excludes products, neural networks and EM. [Function source][function] | `product` has two non-nested analytic children selected from r0 linear, adapted quadratic/RFF, radial and sinusoidal recipes. The child pool is adapted; the profile restricts active inputs to 20 so every choice is valid. |

## r0 exploration beyond the adopted reference settings

These are **r0 design extensions**, not claims of capabilities missing from any
competitor. Exact ranges and exclusions are versioned in [sampling profiles][profiles];
the [public API][api] exposes the recipes through `sample_recipe` and `generate_task`.

| Axis | Implemented r0 controls at the inspected revision |
| --- | --- |
| Linear and polynomial structure | Dense/sparse linear, grouped sparsity, polynomial degree 1–16, quadratic rank 1–256 and explicit interaction strength. Active inputs can reach 256 subject to family constraints. |
| Smooth and Fourier functions | Additive and ridge radial/sinusoidal functions, sampled widths and weighted projections; periodic functions; RFF exploration choices 64–4,096. Periodic broad-profile active count is at most 64. |
| Analytic compositions | 1–64 distinct components and at most 64 leaf occurrences, nested sums/products up to depth 8, selected feature subsets, signed strengths and recorded interaction order. Exact GP draws are excluded as children. [Composition contract][compositions]. |
| Exact GP references | RBF, Matérn 1/2, 3/2 and 5/2, rational-quadratic, spectral and Gibbs kernels; 1–64 covariance components with sums/products, feature-specific scales and spatially varying smoothness/amplitude. These are covariance combinations, distinct from multiplying sampled functions. [Kernel contract][kernels]. |
| Inputs and targets | Independent Gaussian, uniform, Student-t and two-Gaussian-mixture inputs where supported; explicit moment checks/bounds. Plain linear and Fourier recipes require standard Gaussian inputs. Regression uses independent Gaussian observation noise; binary/multiclass labels use sigmoid/softmax probabilities with separation and offsets. [Profiles][profiles], [target implementation][targets]. |

The shared [task shape validator][shapes] accepts 16–20,000 total rows,
1–256 features and binary or 3–100-class multiclass tasks. Exact GP references
have a smaller limit: 1,024 task rows plus at most 512 calibration rows.
These are **API limits**, not measured throughput guarantees or approval of every
Cartesian combination. Finite-moment checks do not establish accurate calibration
from every finite sample. Invalid numerical tasks remain visible failures.

Broad **resource qualification is PENDING**: this source review performs no runs
and establishes no bulk-training eligibility. The profile walkthrough's retained
example checks are narrower than a full diversity/resource matrix; see its
[evidence description][profiles]. Exact methods are not silently replaced by RFF.

## Explicit scope exclusions and remaining evidence

Causal graphs and standalone neural generators remain **#28**; tree generators
remain **#29**; correlated-feature transformations, categorical integration,
missingness, broader observation noise, filtering and production mixture selection
remain **#30**, as separated by the [accepted specification][spec]. Temporal and
distribution-shift prior reproduction is not delivered by the #27 functions above.
An existing diagnostic representative does not establish a training-ready family.

For acceptance, retain generated examples, truth calculations, replay and graph
checks, controlled pairs, full coverage and measured CPU/GPU resource evidence at
identified source revisions. The local explorer, local saving and Colab generation/
saving remain delivery requirements of [#27][spec]. This comparison alone does not
close that issue, validate hosted execution or establish model performance.

[tabicl]: https://arxiv.org/html/2602.11139v1
[tabpfn]: https://arxiv.org/html/2605.13986v1
[limix]: https://arxiv.org/html/2609.17488v1
[mitra]: https://arxiv.org/html/2609.04540v1
[activation]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L168-L173
[function]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py
[api]: ../../src/r0/modules/priors/__init__.py
[sampler]: ../../src/r0/modules/priors/lib/recipe_sampling.py
[shapes]: ../../src/r0/modules/priors/lib/linear.py
[targets]: ../../src/r0/modules/priors/lib/targets.py
[profiles]: ../sampling-profiles.md
[compositions]: ../function-compositions.md
[kernels]: ../gp-kernels.md
[spec]: ../specs/0027-training-ready-generators.md
