# Parent composition and fitted trees in the explorer

The local explorer and shared notebook page expose parent-composition and fitted
structural-tree construction in **Construct a causal graph**. These controls use
the existing timeout-protected generation worker and retained attempt receipts.
The construction seed creates a frozen mechanism; the dataset seed/task identifier
creates rows from that mechanism. Hosted notebook execution remains separately
unverified.

## Parent construction

The explicit presets expose `tabiclv2`, `mitra2` and `extension` composition
sources. Geometry presets include identity readouts and each numerical pipeline
variant. Native-domain composition retains typed observed variables and direct
native targets on an explicit DAG.

`composition` is an optional JSON field on explicit, geometry (including attachment
and motif geometry), and sampled-native controls. It contains the versioned
`ParentCompositionPrior`: source, mode and aggregation. Geometry can also contain
`pipeline_prior`. Native controls with composition use request version
`native-parent-graph-request-v1`. The existing Cauchy-with-fixed-widths transport
does not accept composition; use its geometry topology or an explicit DAG instead.

TabICLv2 distinguishes concatenation from independent functions of each parent,
then sum/product/max/logsumexp. Mitra2 reduces the joint nonlinear output **together
with** linear parent projections for multiple parents; mode remains `sample`,
and the reduction can be mean/weighted/MLP/product/max. Weighted and neural
parameter laws are named r0 adaptations. The extension catalog is separate.
Shared named preferences, actual choices and raw draws remain in the original
archive. See [parent integration](research/0028-parent-integration.md).

The active dataset displays its composition source and lifecycle. The mechanism
editor exposes the complete frozen parent function, including children, joint
member and reduction. Generation inspection shows numerical parent members for
functional graphs and the native equations for typed graphs. Pipeline and domain
editors stay available on their corresponding recipes. Editing produces a new
linked authored graph; it does not change the original sampled construction.

## Fitted structural-tree construction

Four presets expose DT, ExtraTrees, RF and GBDT. The editable controls contain:

- `topology: "fitted"`;
- `skeleton`: an authored, uncalibrated `FunctionalCausalRecipe`;
- `fits`: selected nonroot node indices and complete `FittedTreePlan` records.

The skeleton owns the dataset row/context/split settings. Each selected plan owns
its independent fitting row count, family, regression/classification task, input
and output widths, depth, estimator count, class count, DT splitter, classifier
combiner and boosting learning rate. Widths must match the selected equation.
Use `task: "classification"` in a fit plan for classifier components; the enclosing
functional task still has its declared adapter and does **not** become a native
categorical target. Probability averaging and majority voting remain distinct.

Construction draws a fresh synthetic ancestor population in topological order,
fits independent standard-normal regression targets or uniform class codes, and
freezes split tables. Selected nodes cannot reuse pre-fitted pipelines; a wider
pipeline/native fitting surface remains required work. This is an r0 adaptation,
not any complete published prior. XGBoost, support-only and batch-refit source
lifecycles, categorical synthetic fit populations and other open source options
remain required and pending in [the fitted-tree delivery note](issue-28-fitted-structural-trees.md).

The dataset summary identifies fit nodes, families, rows, target laws/adaptation,
backend version/seeds and output combiners. Complete synthetic populations,
raw input/target/seed draws and split tables remain in Recipe JSON and exported
replay data. Backend-internal random draws are represented by seeds/version,
not claimed to be captured raw draws.

**Next task** reuses the exact original archive text and supplies only a new
seed/task identity. It does not send the immutable numerical records through a
JavaScript JSON reserialization or rerun a fit. Changed task identities produce
new saved task records. The original construction and prefix digests are retained.
Manual frozen-equation changes belong in the mechanism editor: it creates an
authored copy with the fitted tables, leaving original fit custody in the original
archive. Changing a table or upstream equation is an explicit mechanism edit,
not a refit or causal intervention. Edits support save, reopen and export.

## Verification and limits

Public lifecycle checks cover every supplied parent preset and all four fitted
families in both regression and classification, including original-row inspection,
typed/native behavior, equation edits, exports, reopen, frozen task replay and
failed-fit receipts that preserve active state. Tests also prohibit fitting/task
decoding in the UI process. Existing numerical/native/pipeline transport checks
remain regression coverage.

An installed-browser walkthrough is reproducible as
`scripts/parent_fitted_explorer_browser_smoke.mjs CDP_PORT EXPLORER_URL OUTPUT_DIR`.
It exercises parent joint-member inspection, parent pipelines/native domains,
fitted next-task table identity, linked edits, downloads, unsupported XGBoost
failure without state loss, and mobile layout. The local server and Chromium
ports must belong to that run. Generated receipts, logs, screenshots and downloads
are ignored under `docs/evidence/`; full-suite integration, hosted Colab, exhaustive
catalog combinations and downstream scientific comparisons are separate evidence.
