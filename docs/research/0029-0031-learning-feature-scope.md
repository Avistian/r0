# Issues #29–#31: detailed learning and implementation scope

Prepared 2026-11-27 from published head
`5c48672e13e3f7606584d2e9873e427273f93295`. This is research and issue planning,
not implementation acceptance, training authorization or a report of model quality.
The GitHub issue bodies are authoritative planning artifacts:
[#29](https://github.com/Avistian/r0/issues/29),
[#30](https://github.com/Avistian/r0/issues/30),
[#31](https://github.com/Avistian/r0/issues/31).

The supplied **Synthetic priors deep dive** educational HTML was parsed as text,
without executing scripts or changing the file. Its tree, mixture, hyperprior,
missingness and learner chapters motivate the walkthroughs below. Technical
attribution follows primary papers, not the guide. Seventeen versioned primary
paper HTML documents were fetched; relevant sections were inspected. This is a
bounded literature pass, not exhaustive source-code or checkpoint reproduction.

## Decisions preserved and ownership

The roadmap requires broad variety **and intentional extensions beyond papers**.
That breadth must be observable in recipes, compositions, realized parameter
coverage, rare valid cases, saved examples and actual consumed tasks. It is not
established by counting broad family names or widening one parameter interval.

| Ticket | Owns | Does not absorb |
| --- | --- | --- |
| #29 | Standalone direct, oblivious and fitted trees/ensembles; broad new tree variants; independently checkable functions and boundaries | #28's required direct/fitted structural-tree integration, composition, calibration and native types |
| #30 | Observation transforms, missingness, task adaptation, production mixture policies, filtering and accepted/consumed coverage | #27/#28/#29 latent mechanisms; #33 delivery benchmarks; #36 downstream scientific comparison; #39 temporal/relational semantics |
| #31 | One small CPU context learner for regression/binary/multiclass, with broad compatibility and visible forward passes | #32 learned context use/training/recovery; #35 architecture comparisons; every published architecture |

Existing blockers remain #23 for #29; #27/#28/#29 for #30; #21/#22 for #31.
Original roadmap identifiers T17 and T6,T18 remain. No earlier #28 requirement is
removed or reclassified to make that issue appear complete. Source options and
current implementation evidence remain separate; sibling planning does not assert
that current #28 work has delivered every source family or scientific result.

## #29: standalone tree variety

[TabForestPFN §4.1](https://arxiv.org/html/2405.13396v1) constructs a data generator
by fitting a tree on synthetic random features/targets and predicting fresh rows.
[Mitra A.2](https://arxiv.org/html/2510.21204v1) distinguishes fitted estimators from
direct split/leaf construction. These cannot be merged into one “random tree”
implementation. [TabICLv2 Appendix E](https://arxiv.org/html/2602.11139v1) and its
[pinned function implementation](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py)
provide another distinct symmetric-tree mechanism. Exact fitted-source differences
are recorded in [the existing tree-law audit](0028-fitted-tree-source-laws.md).

The detailed issue requires individually traceable options:

- Direct stumps/trees: balanced/unbalanced shapes, depth/stopping laws, leaf/node
  budgets, sparse/repeated relevant features, ancestor-aware thresholds and equality
  routing. Topology is a draw; row evaluation uses the frozen draw.
- Oblivious trees: shared splits at a depth, variable counts, feature preferences,
  scalar/vector leaves and exact reductions; compare with asymmetric trees.
- Fitted DT/ExtraTrees/RF/GBDT/XGBoost recipes: synthetic fit-population identity,
  explicit estimator/version settings, classifier/regressor distinction, frozen
  evaluation and exported-backend parity. Different fitting populations/lifecycles
  are named contracts, not silently interchangeable implementations.
- Target/ensemble semantics: regression values, class labels/probabilities,
  discretized continuous outputs, target noise, mean/weighted reduction, votes,
  probability averages and boosting scores/links. Preserve ties/class ordering.
- **Beyond-paper r0 extensions:** oblique splits, categorical subset splits, soft
  gates, linear/smooth leaf models, correlated forests and tree/smooth/neural
  hybrids. Source labels do not imply these are original paper algorithms.
- Independent/correlated/clustered/heavy-tail feature populations; nuisance and
  duplicate columns; fixed and distributed hyperparameters with explicit joint
  laws; rare branches, empty leaves, near-boundary values and extreme finite scores.

A source reference preset and an exploratory r0 preset must coexist. Unsupported
compositions receive explicit records. A valid constant or very imbalanced task is
not a numerical failure. Rejected/failed draws remain inspectable.

**Teaching artifact:** a stump routing `x1 <= 0.25` to -2 and larger values to 3,
with probes 0.24/0.25/0.26 and an irrelevant-feature change. Add a depth-two
interaction, an empty leaf, a forest tie, a hard-vote/probability-average difference
and a boosted score decomposed by stage. Match feature rows across direct,
oblivious and fitted families; visualize boundaries, path counts and response
surfaces. New oblique/soft/model-leaf recipes need matched simple controls.

**Acceptance evidence:** exact independent path/label calculations, fitted backend
versus exported prediction parity, serialization/replay, local/Colab create-save-
reload examples, printed trees, path/leaf occupancy and class counts. Across seeds
report realized depth/leaf count, split-feature usage, boundary crossings, ensemble
disagreement/correlation, duplicate signatures, parameter-bin coverage and separate
fit/generation/evaluation cost. These establish coverage/correctness, not transfer.

## #30: observations and the distribution that reaches training

[O'Prior v2 §2.2](https://arxiv.org/html/2605.18971v2) explicitly separates latent
structural signal from observation realism and uses support-only preprocessing.
[Kumo Appendix A](https://arxiv.org/html/2611.23001v1) discloses measurement,
postprocessing, missingness and filtering components. In contrast,
[TabICLv2 §8](https://arxiv.org/html/2602.11139v1) says missingness pretraining is
unexplored. The guide must not imply one common missingness recipe for all models.

Required issue inventory:

- Typed continuous/categorical/ordinal/binary/count-like observations, cardinality
  and frequency laws, rare/unseen levels and explicit encoding maps.
- Scaling, warping, heavy/bounded tails, clipping, rounding, censoring, discretizing,
  measurement noise, heteroscedasticity, outliers, noisy duplicate/shared-noise
  columns and engineered features, with explicit transform order.
- MCAR; observed-covariate MAR; informative self/latent/target-dependent masks;
  row/block/categorical masks, tail censoring, near-empty families and latent drivers.
  State the variables conditioned on before calling a process MAR or MNAR.
- Regression units/tails and classification frequency/cardinality adaptations;
  tiny/wide table profiles, weak/constant signals and rare valid regimes.
- **Beyond-paper combinations:** rare categories × informative missingness ×
  imbalance; tails × noise × clipping; redundancy × block missingness; weak latent
  functions × sparse context. Include pairwise coverage plus selected higher-order
  combinations rather than claim the entire Cartesian product is valid.
- Fixed, stratified/quota and hierarchical outer mixtures, shared preferences,
  constrained/random weights, coverage floors and cost-aware schedules. Compare
  static versus curriculum/adaptive candidates while retaining policy decisions.
- Proposed/transformed/accepted/rejected/queued/consumed events, including retry,
  reuse and skipped work, with task/row/query-target/compute denominators.

[Mitra-v2 §2.2](https://arxiv.org/html/2609.04540v1) distinguishes choosing a family
per table from combining functions inside a hybrid SCM. Its reported mixture is
manual, not a learned optimal policy. [Mitra §3.2](https://arxiv.org/html/2510.21204v1)
motivates distinctiveness within a portfolio. These motivate comparisons, not
claims that maximizing nominal variety always helps. Downstream opportunity cost
and complementarity must be measured with #36 under matched budgets.

Mechanism calibration retains its declared frozen contract; learner transforms use
context statistics only. Target-dependent *simulation* can use privileged latent
or target values to create a declared observation process. This is not permission
to expose query labels to learner preprocessing or policy selection. Maintain the
separation in APIs, serialized views and audit examples. Never use protected
milestones or TabPFN outputs as mixture adaptation/filtering signals.

**Teaching artifacts:** transform four rows step by step; compare equal-rate MCAR,
MAR and informative masks; show a fixed latent tree/SCM before and after noise,
coarsening and missingness. An arithmetic fixture proposing A=60/B=40 and accepting
A=30/B=40 yields accepted fractions 30/70 and 40/70. It is an educational example,
not measured coverage; consumed fractions can differ again through skipping/reuse.
Reconstruct one rejected task and one adaptive decision from saved records.

**Acceptance evidence:** local/Colab saveable pipelines, raw/observed pairs,
per-step parameters/masks, independent arithmetic, unchanged hidden/visible
boundaries, event-count reconciliation and requested-versus-realized marginal/joint
coverage. Report duplicate rate, rare/subgroup coverage, correlations, severity,
difficulty, concentration and cost before/after every filter and at consumption.
Preserve valid weak tasks even when a named production policy excludes them.
A downstream gain remains `NOT_MEASURED` without the authorized comparison.

## #31: broad compatibility without expanding the tiny learner into every model

[PFN §4 and Appendix D](https://arxiv.org/html/2112.10510v1) motivates conditional
prediction and distributional outputs. [TabForestPFN §3.2](https://arxiv.org/html/2405.13396v1)
provides an explicit query-isolating attention description. [TabICL §3](https://arxiv.org/html/2502.05564v1)
and [TabICLv2 §3/Appendix A](https://arxiv.org/html/2602.11139v1) illustrate richer
encoders/many-class approaches. These support a decision table, not an obligation
to implement every architecture in a CPU reference ticket.

The issue requires one settled small design with:

- #21 episode/#22 fixture inputs; context labels only; typed features, missingness,
  row/feature/padding masks, query features and declared task/class schema.
- Context-only normalization/category maps; missing versus observed zero versus
  padding; unseen categories, constant/all-missing columns and stable arithmetic.
- A visible feature/label embedding → context conditioning → query readout → head
  → original-unit/class decoding path, with tensor dimensions and CPU caps.
- Regression/binary/multiclass outputs, class ordering/valid-class masks, absent
  context classes and explicit class/shape limits. Shared/separate heads and
  point/bar-distribution regression remain design choices.
- Context permutation and query permutation/chunking contracts across the entire
  pipeline. Query-independent attention alone does not prevent leakage through
  all-rows preprocessing or an earlier encoder. Feature/class symmetries must be
  declared separately rather than assumed.
- A broad compatibility matrix of types, masks, shapes, class imbalance, rare
  categories, weak/constant targets, extreme values, correlated/nuisance features
  and beyond-paper generator combinations. Unavailable generator cases remain
  `PENDING_GENERATOR`, not fabricated passing fixtures.
- Architecture/config/weight identity and saved inputs/intermediates/outputs with
  numerical tolerances, timing/memory and actually executed environment records.

**Teaching artifact:** one whole-model diagram and three tiny saved forward passes,
one per task type. For a transparent attention calculation, scalar query q=0 and
keys 0/1 give equal softmax weights; values 2/4 give readout 3. This is a teaching
fixture, not a trained-model result. Inspect the head's logits/probabilities or
regression units; mutate inaccessible query labels, reorder context, duplicate
queries and change chunks. Compare missing/zero/padded cells through the diagram.
Local and Colab examples must call the same public learner and save their outputs.

**Beyond-paper scope:** study named r0 combinations of encoder/head/missingness
choices and their compatibility with rich generators. Implement one agreed tiny
reference, leaving clean interfaces for #35 comparisons. The architecture decision
record compares row versus cell/dual-axis encoding, numeric embeddings, shared
versus separate heads, point versus distributional regression and fixed versus
expandable class representations. Read curie-llm for reuse before new model code,
as required by `docs/requirements.md`.

**Acceptance evidence:** saved actual traces, shape/mask inspection, independent
selected head/attention arithmetic and perturbation counterexamples that would
reveal query leakage. No learned use of context, calibration or accuracy claim
follows from an untrained forward pass; #32/#35/#36 retain those responsibilities.

## Source coverage, versions and disclosure gaps

The [existing inventory](0028-source-option-inventory.md) retains all relevant
paper options, boundaries and contradictions. Its JSON records are not a count of
unique algorithms. Each option gets a source/version, construction, explicit
hyperprior, compatible compositions, owner, example and evidence status.

| Versioned primary source | Relevant disposition |
| --- | --- |
| [PFN](https://arxiv.org/html/2112.10510v1), [TabPFN v1](https://arxiv.org/html/2207.01848v6) | Conditional learner/head/mask concepts; source-specific prior/observation contracts |
| [TabPFN v2](https://www.nature.com/articles/s41586-024-08328-6) | Existing inventory Methods coverage; this pass does not claim a new Nature full-text audit |
| [TabPFN-2.5](https://arxiv.org/html/2511.08667v1), [3](https://arxiv.org/html/2605.13986v1), [3.5](https://arxiv.org/html/2609.17895v1) | Preserve version distinctions; broad improved-diversity claims do not reveal every sampling law; temporal/grouped extensions retain #39 boundary |
| [TabICL](https://arxiv.org/html/2502.05564v1), [v2](https://arxiv.org/html/2602.11139v1) | Fitted versus symmetric trees, preprocessing/filtering, encoder/class alternatives; v2 missingness caveat |
| [TabForestPFN original revision](https://arxiv.org/html/2405.13396v1) | Fitted random-data tree and query-mask construction; preserve original revision/title |
| [Mitra](https://arxiv.org/html/2510.21204v1), [v2](https://arxiv.org/html/2609.04540v1) | Direct/fitted tree distinction, portfolio distinctiveness, outer versus hybrid mixture; missing numeric fitting details stay unknown |
| [LimiX](https://arxiv.org/html/2509.03505v1), [LimiX-2](https://arxiv.org/html/2609.17488v1) | Observation/target adaptation and dual-axis learner alternatives; structural graph operations remain #28 |
| [O'Prior v2](https://arxiv.org/html/2605.18971v2) | Support-only realism layer; preserve version change relative to earlier v1 inventory rather than overwrite it |
| [Kumo](https://arxiv.org/html/2611.23001v1) | Disclosed measurement/missingness/filtering and shared preferences; encoder alternatives |
| [TabAbax](https://arxiv.org/html/2611.00562v1), [TabFM](https://arxiv.org/html/2609.37959v1) | Typed/category and architecture distinctions; broad expanded-prior claims are not exact undisclosed algorithms |
| [TAFFY](https://arxiv.org/html/2610.07559v1) | Causal intervention/structural interface remains #28; does not justify an invented #29/#30 sampler |

Guide references on GP/BNN theory remain #27/#28 background. Real-data-pretraining
examples are excluded by synthetic-only training policy; guide comparison or
research provenance does not authorize copying code, data or weights. Exact
library defaults/checkpoint-linked settings remain unknown where unpublished.
XGBoost dependency availability is an implementation decision, not satisfied by
renaming another backend. Fixed source ratios/ranges are reference presets;
exploratory extensions must carry their own explicit definitions.

## Publication and review evidence

Issue bodies retain every original body verbatim beneath the detailed additions;
titles, state, labels, assignees and relationships are not edited. Generated backup,
comment/relationship snapshots, source extracts, drafts, PATCH requests and
readback verification live under ignored
`docs/evidence/issue-29-31-learning-scope/`. Only this authored note is tracked.
No implementation, dependency installation, dataset download, training/cloud run,
protected evaluation edit, HTML edit or new PR is part of this work.

The issue update is complete only after exact body readback and comparison of
untouched metadata, comments, dependency/sub-issue ordering and original-body
preservation. The detailed bodies preserve open decisions; publishing this planning
detail is not evidence those implementation choices have been settled.
