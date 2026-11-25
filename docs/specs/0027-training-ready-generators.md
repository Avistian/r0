# Issue #27: training-ready linear, sparse and smooth generators

Status: accepted for implementation on 2026-11-25.
Ticket: https://github.com/Avistian/r0/issues/27

## Accepted scope

- Deliver configurable, replayable generator families that sample mechanisms,
  feature distributions and noise levels, with inspectable examples and measured
  throughput. These must be training-ready, not merely diagnostic demonstrations.
- Keep mixture selection and training-stream integration in #30. Expose a repeatable
  sampler producing valid training episodes with explicit seeds, configurable sizes
  and bounded memory. Demonstrate actual bounded batch consumption without
  generator-specific caller glue. Full model training and distributed delivery
  remain separate work except the resumable parallel generation explicitly included below.
- Variety is a primary requirement: do not limit delivery to the proposed minimal
  dense-linear, sparse-linear and RBF set. Design for training at up to 100 million
  independently sampled source tasks, each with its own mechanism and observations.
  Additional episodes/splits from one source task do not count as independent tasks.
  Resource limits and the detailed scale validation protocol remain open. This
  target does not authorize a full run.
- Require an explicit diversity coverage matrix across mechanisms, feature
  distributions, dimensionalities, sparsity, signal strengths, noise and
  classification difficulty. Record supported combinations and exclusions; distinct
  seeds alone are not evidence of variety.
- Breadth is an explicit ambition: extend coverage beyond the generator families
  documented in frontier-model papers, rather than copying a minimal published
  recipe. The accepted inventory and sources below define the current breadth;
  this ambition is not evidence of performance superiority.
- Include dense, sparse and grouped-sparse linear mechanisms; quadratic functions
  with controllable interactions; additive smooth polynomial, sinusoidal and
  radial-basis components; periodic functions with sampled frequencies/phases;
  Gaussian processes with RBF and Matérn kernels and kernel combinations; and
  explicitly identified random Fourier feature approximations. Each family needs
  configurable parameter distributions, replay and visual explanations. This is
  an accepted baseline inventory, not a ceiling on breadth. Causal generators and
  standalone random networks remain in #28.
- Maintain a source-linked comparison of published generator capabilities,
  reproduced capabilities and our extensions. Mark undocumented details unknown.
  Demonstrate extensions with generated examples and diagnostics; predictive
  benefits require separate evidence.
- Support structured compositions, including sums and interactions of mechanisms,
  with explicit component strengths, interaction order and complexity limits.
  Preserve replay and expandable visual explanations. Report standalone and
  composed mechanisms separately in coverage evidence.
- Distinguish bulk-training generators from expensive reference generators.
  Exact GPs support bounded workloads and validation; scalable approximations have
  distinct names. Measure resource use per family and shape. Bulk recipes explicitly
  select eligible generators; never silently replace exact methods with approximations.
- Complete the literature comparison and agree on an expanded finite inventory
  before implementation. Every included family requires a training-consumable
  sampler, replay, checkable truth, diversity diagnostics, visual explanations and
  resource measurements. Record later discoveries as explicit follow-up candidates.
- Support independently sampled Gaussian, uniform, heavy-tailed and multimodal
  continuous features with configurable location and scale. Correlated-feature
  transformations, categorical integration and missingness remain in #30.
- Target eventual trained-model performance at least matching LimiX-2, TabPFN-3
  and TabICLv2. Generator correctness or coverage alone does not establish this;
  training and model comparisons require separate measured evidence.
- Support 16–20,000 total rows, 1–256 features and 2–100 classes, with configurable
  context/query sizes. Expensive reference generators have smaller explicit limits.
  Reject unsupported sizes clearly. Measure representative small, medium and large
  workloads before setting throughput targets.
- Retain numerically valid tasks with weak signal, uncertain labels or classes
  absent from context; expose these properties in diagnostics and the explorer.
  Reject invalid numerical outputs. Filtering valid tasks remains #30 policy.
- Give each family explicit, versioned parameter distributions. Record every
  sampled value. Separate broad exploration recipes from fixed throughput workloads;
  production mixture weights remain in #30.
- Express regression noise as a noise-to-signal ratio, using independent Gaussian
  observation noise initially. Broader observation-noise models remain in #30.
  Normalize signal using the generating distribution or an independent reproducible
  calibration sample, never query labels. Distinguish heavy-tailed input features
  from heavy-tailed observation noise.
- Invalid numerical output or a declared resource-limit breach produces a visible
  failure retaining task identity, recipe and reason. Do not silently retry, resize,
  change mechanisms or substitute families. Preserve failures in task counts and
  throughput reports.
- Classification exposes score scale for class separation and class offsets for
  prevalence. Show expected probabilities and observed class counts. Do not force
  every class to appear or resample labels to match exact counts.
- Completion includes a reproducible walkthrough of every family, all three task
  types, controlled pairs and largest supported shapes. Retain worked target
  calculations, replay checks, graphs, resource measurements and failures. Inspect
  small cases in detail and measure larger cases. Label extrapolated time for
  100 million tasks separately from measured execution; full training is separate.
- Use published reference settings and the accepted broader ranges below,
  preserving unknowns instead of inventing paper-backed settings.
- Support both CPU and GPU generation. The user has Google Colab compute available
  and wants to use it for generation and measurement. CPU-only delivery with GPU
  deferred is not accepted. The proposed one-core/4-GiB/60-second/30-minute CPU
  measurement budget is not accepted. Runs declare their own validated limits.
  This is not authorization to launch an unattended cloud run.
- Provide a Colab notebook installing a pinned repository revision and supporting
  CPU/GPU generation, a small smoke run, configurable larger runs, throughput
  measurement and exports. Display actual hardware, elapsed time, peak memory and
  failures. Use the same generator code as training and the explorer.
- Support generation runs in both Colab and Modal, including larger workloads.
  Specify task count, time limit and output-storage limit for each run. Show measured
  rate and estimated completion time before the larger run. Resume by task identity
  without regenerating completed tasks. Implement bounded parallel execution,
  persistent output groups, progress and restart recovery in this delivery;
  do not defer the working scaling path to #33. Use the existing guarded workflow
  for Modal. This does not authorize a paid launch.
- Save bounded groups of tasks to persistent storage, with identities and checksums
  in manifests. Mark groups complete only after saving and verifying their files.
  Skip verified groups on restart and recover incomplete groups. Avoid one tiny file
  per task at 100-million-task scale. Support Drive paths in Colab and Modal Volumes.
- Keep failed tasks visible. Retry only with an explicit option, preserve every
  attempt, and reuse the same task identity and recipe. Report attempted, completed
  and failed counts separately; never replace a failed task with another task.
- The same task identity on CPU/GPU preserves sampled mechanism and random inputs.
  Compare calculated values with explicit numerical tolerances; bitwise equality
  across devices is not required. Record device and software environment for exact
  replay. Unsupported operations and failures remain visible.

## Published reference recipes and broader exploration

Use TabICLv2 commit `9530bd65a7b806f6ad4f9f4335af745577e9d3cc` as the
source of reference settings, implementing them independently. The user explicitly
requests broader ranges wherever feasible: these settings are reference recipes,
not library-wide ceilings. Preserve them for comparison while adding separately
named exploration recipes varying polynomial degree, active inputs, component
count, composition depth and approximation size. Expose exact ranges in recipes;
validate numerical behavior and resource use before marking them bulk-training
eligible. Accepted exploration ranges are polynomial degree 1–16, 1–64 combined
components, composition depth 1–8, up to 256 active features, and 64–4,096 Fourier
features. These are exploration ranges to validate, not guarantees that their
Cartesian product is supported or eligible for bulk training.

Validate mathematical compatibility between inputs and mechanisms. In particular,
do not claim finite expected signal variance for combinations without finite
required moments. Offer explicitly bounded input distributions where appropriate,
record the bounds, and explain rejected combinations. Never silently clip inputs
or treat a finite empirical variance as proof of finite theoretical variance.

- Integer-power components sample exponents 2, 3, 4 or 5. This is not a bound on
  the degree of an entire composed function.
- Its quadratic recipe uses degree 2 over at most 20 selected inputs.
- Its random Fourier feature recipe uses 256 frequencies.
- Its product recipe multiplies two functions and excludes nested products.
  Adapt eligible child families to #27's accepted inventory and label that adaptation.
  This restriction belongs to the reference recipe, not all broader compositions.
- Do not adopt the earlier universal eight-component/two-level cap. TabICLv2's
  causal graph recipe has 2–32 nodes, not a universal two-level composition rule;
  causal graph generation remains #28. Limits for our additional compositions
  still need explicit recipes and resource validation.

Pinned sources:
[integer powers](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_activation.py#L168-L173),
[function settings](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py#L206-L313),
[graph configuration](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py#L25-L31).

LimiX-2 section 4 and TabPFN-3 section 2.5 describe composition and broader
mechanisms but do not disclose equivalent numerical limits. Do not attribute
invented caps to them. These adopted settings do not reproduce either full prior.

## Generation, replay and controlled pairs

- Include smooth functions of weighted feature combinations; smooth components
  acting on chosen feature groups; multiple frequency bands; feature-specific
  rates of variation; and spatially varying amplitude or smoothness.
- Generate tasks on demand by versioned recipe, root seed and task identifier,
  independent of generation order. Memory depends on the current batch rather than
  total task count. Retain compact provenance and representative inspection
  artifacts. This delivery includes output groups and parallel execution; #33 can
  build comparative training-delivery experiments on that working path.
- Support regression, binary classification and multiclass classification through
  shared underlying signal-generating mechanisms and explicit target adapters.
  Retain noiseless regression targets and classification scores/probabilities.
- Binary classification uses a sigmoid of latent scores; multiclass uses softmax
  over class scores. Sample labels from these probabilities, with an explicit
  ambiguity parameter. Scores and probabilities are diagnostic truth excluded
  from model inputs.
- Controlled pairs preserve the shared mechanism, active inputs and context/query
  split when adding irrelevant columns or changing observation noise. Use separate
  nuisance and noise random streams. Sparsity interventions hold expected signal
  variance fixed by default; the precise normalization remains to be specified.
- Distinguish mathematical properties from model performance. Verify controlled
  interventions directly and measure model performance across paired draws, without
  requiring every model or individual draw to worsen monotonically. Sparsity alone
  does not guarantee greater difficulty.

## Accepted interactive explorer

- Ship a local browser application started with one command, using the same public
  generator API as training, without cloud services or TabPFN.
- Support family selection, parameter controls, new tasks, exact replay by identity,
  and export of recipes and inspectable datasets. Display identity and recipe.
- Show table previews, feature and target distributions, active/irrelevant feature
  roles, and context/query membership. Include regression signal/noise plots and
  classification probability views. Label high-dimensional slices/projections.
- Show controlled pairs side by side for noise, sparsity and irrelevant-feature
  interventions, identifying fixed properties and displaying measured changes.
- Make the generation process itself chartable, including generation graphs,
  sampled parameters and how observations and targets were produced. Diagrams
  represent actual generation steps, with selectable nodes exposing parameters,
  formulas where applicable, and charts of inputs/outputs. Expand family-specific
  components and kernel combinations.
- Reconstruct detailed intermediate values on demand from compact provenance;
  do not persist full traces for every training task. Label sampled or truncated
  views. Verify inspection does not change generated data.
- For exact GP views, distinguish original sampled values, interpolation and
  additionally sampled values, recording how additional values were obtained.
  Analytic and finite-feature mechanisms expose their actual response curves.
- Generate the full requested task and compute full-task summaries, with bounded,
  reproducible row samples for expensive charts. Display shown and total row counts,
  allow individual-row inspection, and export the complete dataset and recipe.
  Reconstruct detailed traces only for selected rows or components.
- Export replay bundles containing JSON recipes, identities and sampled parameters,
  lossless NumPy arrays, and inspection CSV tables. Export charts as SVG/PNG.
  Keep hidden targets and mechanism details separate from model-input exports.
- Expose the explorer inside the Colab notebook, including controls, generation
  graphs, plots and paired comparisons. Reuse visualization code; local application
  access must not be required to inspect Colab-generated tasks.

## Pending work and owners

| Kind | Work | Owner | Blocks |
| --- | --- | --- | --- |
| Decision | Parameter distributions, normalization, numerical tolerances and exact-GP extension rules | Implementation author, checked by scientific reviewer | Relevant generator PR completion |
| Decision | Explorer implementation and chart limits using existing dependencies where possible | Implementation author | Explorer PR completion |
| Decision | Measurement repetitions, timing boundaries and per-run resource-cap validation | Implementation author | Runner and evidence PR completion |
| Verification | CPU/GPU agreement, maximum supported shapes, exact-GP row limit and throughput | Verifier on recorded hardware | Bulk-training eligibility claims |
| Authorization | Hosted run configuration, money caps and guarded Modal launch | User | Paid/cloud execution |
| Verification | Colab/Modal execution, durable restart and notebook interaction | Verifier in the target environments | Hosted-runtime acceptance |
| Decision | Any additional dependency required by implementation | User, through ticket scope | Adding that dependency |

These details must be explicit in versioned recipes and evidence. They do not
authorize silently reducing the accepted families, platforms or visualization scope.

## Research basis for accepted extensions

These support the accepted extensions. They are not evidence that competing models
lack these capabilities. Exact constructions and parameter ranges remain to be set.

- Smooth functions of weighted feature combinations:
  [projection pursuit](https://doi.org/10.1080/01621459.1981.10477729).
- Additive GP components over chosen feature subsets:
  [additive GPs](https://proceedings.neurips.cc/paper_files/paper/2011/hash/4c5bde74a8f110656874902f07378009-Abstract.html).
- Multiple frequency bands:
  [spectral mixture kernels](https://proceedings.mlr.press/v28/wilson13.html).
- Feature-specific length scales, position-dependent covariance, smooth amplitude
  envelopes and rational-quadratic multiscale covariance:
  [GPML chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).

Published comparison sources include
[TabICLv2](https://arxiv.org/html/2602.11139v1),
[TabPFN-3](https://arxiv.org/html/2605.13986v1),
[MITRA-v2](https://arxiv.org/html/2609.04540v1) and
[LimiX-2](https://arxiv.org/html/2609.17488v1).
Their disclosed function categories do not establish complete checkpoint-linked
sampler configurations. The coverage comparison must preserve that limitation.

## Design and language requirements

- Keep abstractions as simple as possible. Breadth of supported mechanisms must
  not become unnecessary framework complexity.
- Use one public generation interface and one result format: recipe, seed and task
  identifier produce an episode with separate diagnostic information. Implement
  families as ordinary functions, sharing sampling, classification, replay and
  inspection code. Introduce abstractions only for concrete implementation needs.
- Use concrete, direct, understandable language in APIs, documentation and the
  explorer. Define technical terms when needed; avoid vague capability claims.

## Handoff

Deliver small, reviewable PRs with explicit dependencies and per-slice evidence:
specification; generation/replay slices; resumable bounded execution; Colab/Modal
integration; explorer and final coverage/resource evidence. A partial PR must not
claim to close #27. Use the public generator, run/resume and explorer interfaces as
test seams; check mathematical results independently rather than testing internals.

This document is the authoritative accepted scope. Link its published commit-pinned
revision from #27, preserving the rest of the issue body. Full completion requires
all accepted behavior, with unavailable hardware or hosted execution stated as
NOT_RUN rather than inferred from local tests.
