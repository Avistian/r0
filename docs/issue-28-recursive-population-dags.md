# Row-owned recursive forests inside DAGs

This library slice extends the explicit terminal population lifecycle to mechanisms inside an unchanged authored DAG. `RecursivePopulationGraphRequest(base, replacements)` selects ordered, unique node indices with `RecursiveNodePopulationPlan(node, forest)`. Each forest plan must declare `fresh-forest-per-row`, the complete task row count, and the replaced function's input/output geometry. `realize_recursive_population_graph(request, seed=..., task_id=..., rng=...)` retains every node's complete sampled forests, raw draws, construction snapshots and actual original input rows. It never inserts row-varying objects into the frozen-function registry.

All four authored graph families are supported: functional numeric, scalar native, heterogeneous native vector, and typed converter graphs. Parents remain one joint input, with native categories encoded into their declared descendant state. Regression forests can emit vectors jointly. A direct classification forest requires a single native categorical variable with deterministic argmax emission: its exact majority-label indicator is decoded to the actual label, not represented as sampled class probabilities. Ordinary downstream softmax/threshold domains and typed converters remain available, including stochastic emissions driven by their original keyed streams.

The source distinction from [the terminal population slice](issue-28-recursive-populations.md) remains: MITRA Algorithm 5 motivates fresh row ownership; each tree retains its separately named prose/algorithm/extension laws. Placement into arbitrary authored graph families, ordered node-major stream allocation, vector leaves and these composition APIs are explicit r0 extensions, not a claim of full MITRA prior reproduction. The original graph is retained as a counterfactual and valid geometry contract; its selected functions are overridden during population execution.

## Replay, intervention and inspection

Generation replays the retained population only for its original seed/task identity. A different population requires an explicit realization. The optional caller NumPy generator is advanced atomically only after complete validation; all five supported engines retain their identity. Construction uses one ordered stream across replaced nodes, independently of task root, structural innovation, emission and adapter streams.

`intervene_recursive_population_graph(base, change)` accepts a `CausalInterventionRecipe` whose base exactly matches the retained original DAG. Matched mode retains every forest; population mode derives a separate engine-preserving stream from the original identity and explicit population index, then retains every new receipt. Full hard cuts bypass the counterfactual runtime mechanism/root/innovation; partial logical-variable cuts evaluate the joint mechanism before applying the explicit span clamp. Baseline archive custody is validated separately before this runtime. Explicit soft function replacements disable population replacement at that node. Root, noise, domain, converter and pipeline changes retain population ownership where compatible. A classifier's direct-label semantics cannot silently become a stochastic logit mechanism.

`generate_population_graph_pair` returns the unchanged baseline and explicit counterfactual, with receipt/input/split equality metadata and raw variance semantics. It performs no query calibration or variance matching. Standard recipe decoding, task ZIP persistence and `explain_generation` support both records. Traces replay the full original population before selecting repeated/out-of-order rows; show row-specific paths and leaves, downstream encoded dependencies, structural draws/scales, effective cuts, typed probabilities/uniforms, and stochastic target adapters. Strict task inspection binds tensor dtype/layout/shape/values, complete semantic metadata and implementation identity while allowing validated descriptive historical environment fields.

Whole-graph geometry, node counts, aggregate forest raw/parameter budgets, retained capture matrices and nested replay workspace are reserved before construction or nested archive RNG restoration. These conservative array estimates are not measured process RSS. Metadata budgets include the complete realized archive and both sides of intervention custody. Caller limits remain authoritative.

## Verification and remaining scope

Public focused checks use literal row-specific path traversal and vote/mean arithmetic, all four graph families, all five RNG engines, stochastic emission probabilities, hard/soft cuts, complete archives, large integer IDs, ZIP replay, pre-restore sentinels and caller rollback. Author logs and earlier failures remain in ignored `docs/evidence/issue-28-recursive-population-dags/`; independent review is a separate gate.

Dynamic target calibration for row-owned mechanisms remains required and explicitly unsupported in this slice; a fixed-row population cannot silently stand in for an independent calibration population. Sampled topology/catalog construction and workbench controls for internal population placement remain required follow-ons. Full-paper hyperprior reproduction, large-workload measurements, hosted/GPU behavior and a combined full suite are not claimed here.


Shared `compare_causal_pair`, `save_causal_pair` and `load_causal_pair` now validate population-DAG comparisons against the complete retained original and counterfactual records. Both matched and fresh-population pairs preserve their full forests, original row inputs and task semantics through paired ZIP groups. Existing guarded open/inspect/export actions operate on these saved comparisons; workbench construction/catalog integration remains a separate required follow-on. Caller limits pre-admit the complete two-population record before nested construction snapshots are restored.

## Outer intervention admission repair

Independent whole-DAG review of `4487ae523e1d2b0ac3f7511396b5bfadd98b28c0` stopped with REJECT: changing only the outer comparison mode to an invalid value restored six nested construction snapshots before rejecting. Its historical evidence remains unchanged in the frozen pair worktree. Earlier narrow paired-archive acceptance does not supersede this defect.

The decoder now validates the complete outer comparison schema, versions, identity, intervention semantics, receipt bindings and retained input geometry before baseline replay. A temporary authored view of retained sampled roots and functions passes through the existing intervention constructors and graph compatibility validators. This view is discarded: the original sampled construction records are decoded and replayed afterward, retaining their paths, distributions and custody. Parameter and workspace admission still accounts for the original complete archive before constructing the semantic view.

Repair evidence is retained under ignored `docs/evidence/issue-28-population-dag-preflight/`. The original public invalid-mode probe now requires zero snapshot restores, with a valid control. The initial 21-case public RED run exposed 18 failures; the repaired checks also cover additional malformed receipt, identity, nonfinite-input and unknown-field cases. A sampled prototype root with sampled recursive source and soft replacement verifies preservation of the original records. Separate valid-artifact comparisons cover four graph families, five RNG engines and matched/fresh populations, including JSON replay, tensors and repeated-row traces.

These are bounded repair checks. Independent repair acceptance and continuation of the stopped broad DAG review remain separate gates; no whole-issue acceptance or full-suite result is claimed.

### Nested archive reservation repair

Independent review of `b6e39b1` found a generated16-row, two-output/two-estimator
population pair that could not decode under its original256MiB limits. The outer
pair decoder had already reserved the complete archive and two separate runtime
workspaces; passing a reduced runtime allowance to the standalone baseline
decoder charged baseline archive metadata again. The repair keeps the outer
aggregate admission and restores the baseline under the original caller ceiling.
Original source/plan limits and archive bytes remain unchanged.

The worked case reserves11,038,130 bytes of outer archive storage (with the
existing8x factor), while each forest requires79,268,864 estimated working bytes.
Its base alone and the repaired pair decode at256MiB. Public regression checks
also verify1/64/128/224MiB callers reject before any snapshot restoration. The
49-case focused pair/preflight run passes. Broader numerical continuation and
independent repair acceptance are recorded separately; earlier rejections remain.
