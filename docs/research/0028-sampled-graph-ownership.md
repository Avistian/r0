# Issue 28: sampled functional graph ownership

Research 2026-11-27 at `ae815a6f44205b13ea39654ee36df5ce3099a14d`. Preflight confirmed
`/tmp/r0-28-graph-sampling`, this HEAD and only untracked `.venv`; instructions read.
Only this note authored. Citations remain pinned despite later parent edits. Source/
checksum reads executed; checks **NOT_RUN**, resources **NOT_MEASURED**. Independent
authored-graph review and full #28 remain unfinished. [History:34–54; Spec:122–167]

## Records

Use builder-owned staging; registry/runtime validation precede caller RNG commit.
[Graph:16–61,82–105; Eval:50–105]

Proposed interfaces (not implemented):

```text
JointGraphRequest(task, topology_prior, width_prior, root_prior, function_prior,
                  pipeline_prior, noise_prior, readout_prior, calibration_policy,
                  normalization_policy, limits, source_variant)
JointGraphPlan(operation, topology_plan, nodes: tuple[JointGraphNodePlan],
               observed, target, policies, source_variant, version)
JointGraphNodePlan(index, parents, input_width, output_width,
                   mechanism: JointRootPlan | JointFunctionPlan,
                   pipeline: JointPipelinePlan | None, noise: JointNoisePlan)
JointGraphConstruction(plan, registry, owners, raw_manifest, realizations,
                       calibration_receipts, bindings, admission, version)
SampledFunctionalRecipe(runtime: FunctionalCausalRecipe,
                        construction: JointGraphConstruction, version)
sample_joint_graph(request, *, named_rng, raw_rng, calibration_rng, fit_rng)
    -> SampledFunctionalRecipe
```

Use closed tagged records and sampled card/decoder; current cards attest authored
provenance. Preserve runtime validation, parent/readout order, state-width sums and
plan/realization/runtime width bindings. Constants are not draws.
[Graph:52–61,82–105; Eval:204–238]

Source `JointRootPlan` is d→d, matching its realization and `RandomPoints(n,n)`.
Authored roots allow different output widths. Sampled d→o requires versioned
`JointGeneralRootPlan(base,function)`/realization/admission, explicitly r0 extension;
reject that unsupported sampled request without relaxing v1. Preserve authored
support and heterogeneous nonroots. [RootPlan:86–98; Root:79–118; Graph:60–61; UpPoints:7–16; Spec:68–71]

`JointPipelinePlan` owns ordered activations, channel importance, L2 fit, converters
and node importance. `JointNoisePlan` owns output-width scales, exogenous law,
response and sampled/constant origins. Converter fitting already requires the actual
stage input and delegates ancestor provenance, moments and aggregate admission.
A full pipeline sampling composition is proposed here. [Pipe:166–214; FitPlan:53–67; Graph:33–50]

## Ownership and raw manifest

Registry checks preorder/containment/sibling order/exact spans, not semantic types.
Decision-only checks miss foreign empty scopes; uniform weights need ownership with
zero named draws. [Registry:47–106; FPlan:209–244; RootPlan:100–123; Weights:178–182]

Propose a closed `OperationOwner` union for Graph/Topology/Widths/Readouts/Node,
Root/Function/Matrix/Weight/Point/Activation/Converter/Pipeline/Noise plans,
Realization, CalibrationPopulation and Fit. Each has path, typed reference and
ordered children. Derive expected traversal from plans plus construction schedule;
compare exact registry path sequence/spans/types. Reject extra/missing/mistyped/
reordered owners, including zero-event ones. Each leaf event/raw destination has
one owner; parent containment is intentional. [Registry:64–106; FPlan:167–186,209–244]

Suggested layout:

```text
graph/plan/{topology,widths,node:0/...,node:1/...,readouts}
graph/realization/node:i/{function-fit-population,materialize,pipeline-fit/...}
```

Preserve existing nested plan names (`function`, `matrix`, `weights`, `linear:i`,
`activation:i`, `left`, `right`). Construction scopes reference plan owners; late
plan decisions violate exact-subtree validation. Open actual transactions; never
fabricate spans afterward. Shared materialization owners still require typed raw
field roles. [FPlan:76–102,151–165,243–244; RootPlan:120–123; Context:90–154]

Propose:

```text
RawDraw(owner, role, stream_role, sequence, distribution, parameters,
        shape, dtype, raw_ref, digest)
Derivation(owner, algorithm_version, input_refs, output_ref, output_digest)
```

References resolve to retained realization/receipt payloads or a canonical raw store;
digests alone are insufficient. Bind plan→realization→runtime, dimensions, fit
samples, parents and variant by equality/digest. Compare expected raw inventory;
reject orphans, duplicate destinations and wrong roles. Calibration digest alone
does not prove ancestral/RNG origin. [FitSample:15–56; Root:67–122; MLP:79–121]

Record at owning draw sites with real staged generators (exact-type checks reject
proxy RNGs); roll back recorder state with helper failure. Preserve draw order.
Post-hoc traversal loses discarded evidence: kernel matrices retain `sign(normal)`,
not raw normals; prefix ball noise also needs raw directions/radii. Add recording
before transformations, not Gaussian labels on derived values.
[Matrix:209–240,244–248; Exo:63–88; RootBase:119–126]

| Owner | Required inventory and owning code |
| --- | --- |
| Linear | Matrix realization and frozen equation; no function fit. [Function:246–263; Matrix:211–275] |
| Quadratic | Matrix batch, full permutation if input width >20, selected indices, bias-augmented geometry. [Function:248–260; FPlan:121–127] |
| RFF (`gp`) | Frequency uniforms, radial directions/factor/weights where applicable, phase uniforms, Gaussian coefficients, derived equation. [RFF:127–148] |
| Oblivious tree | Calibration, feature probabilities, selected features/threshold-row indices, Gaussian leaves, derived thresholds. [Tree:120–159] |
| Discretization | Calibration, full fit-row permutation (empty for one row), centers, matrix and targets. [Disc:103–144] |
| EM assignment | Calibration, anchor-row indices, Gaussian jitter/spreads, output matrix, derived assignments/equation. [EM:145–186] |
| Product | Two cheap children share the SAME input population, with distinct child raw/fit choices; no recursive full selector. [Function:267–326; FPlan:85–105] |
| MLP | Ordered matrices, per-activation stage inputs, shift rows, fitted constants/outputs, start/end flags and heterogeneous layer widths. [MLP:129–202; FPlan:138–186] |
| Root | Four bases; covariance factor/weights; base raw arrays/ball uniforms; function and own-fit outputs. [Root:150–178; RootBase:119–126] |
| Nested matrix/weight/point | Gaussian/importance/factorized/kernel/transformed branches, normalization jitter, polynomial normals/permutations or uniform weights, point assignments/centers/factors/innovations. [Matrix:211–275; Weights:198–214; Points:57–78] |

RNG replay needs algorithm, trusted pre-state/seed, parameters and order. Raw values
alone prove no distribution law; never unpickle archive state. [Streams:26–38]

## Fresh populations and evaluation

EACH receiving function fit gets fresh draws through frozen ancestor root laws,
functions, pipelines and innovations, concatenating parent **states in order**.
Ancestor own-fit outputs are in-sample; new IDs do not make them independent.
Helpers leave independence to the caller. [Root:101–118,163–178; FitSample:38–56;
Function:184–187; Eval:64–101]

Schedule: admit plans; materialize roots on their own independent base fit draws;
fit root pipelines on NEW populations through frozen functions. For each nonroot
fit, draw fresh frozen-prefix inputs, materialize function, then fit its outer
pipeline on ANOTHER fresh population through that frozen function. Freeze the node
before descendants. Tree/discretization/EM/MLP and products containing fitted cheap
children need samples; linear/quadratic/RFF need no fictional fit receipts. MLP and
pipeline internal stages consume preceding stage outputs within their respective
receiving population. Emission/target-fit populations stay separate; no query labels.

Independent calibration is an r0 adaptation: source fits on first call, sequential
stages see preceding outputs and roots pass base samples directly to functions.
[UpBase:172–197,221–232; UpPoints:7–16; MLP:154–178; Function:194–245,277–285; Spec:76–80]

Use distinct named/raw/calibration/fit streams. Fresh populations consume staged
calibration draws with ordered IDs/state bindings and deterministic subroles.
Derived streams need versioned seed law/accounting; IDs do not prove independence.
Avoid one RNG/node (default cap 64). [Streams:11–16,41–64; Context:244–252]

Complete-recipe validation requires task readouts; extract a prefix seam, not dummy
feature/target nodes or the affine fitter. [Graph:89–105; Eval:50–65; OldFit:53–108]

```text
admit_functional_prefix(nodes, *, rows, limits, retained_cost)
    -> AdmittedFunctionalPrefix
evaluate_prefix(prefix, *, draws: FunctionalDrawProvider, device, trace=None)
    -> PrefixEvaluation(states, readouts)
evaluate_node_equation(node, inputs, raw_innovation, *, trace=None)
    -> NodeEvaluation(function_output, noise_scale, state, readout)
```

Prefix validates DAG/widths/certificates/row bounds; internal immutable admission
token binds nodes/rows/limits. Shared kernel handles f, scale from f, before-noise
pipeline, innovation, finish, finiteness and diagnostics. Runtime validates task
then delegates; builder/reconstruction share kernel and typed retained draw provider.
Preserve emitted keys/original-row replay. [Eval:50–105,244–256; Graph:82–105; Pipe:166–214]

Receipt binds receiver/purpose/rows, ancestor IDs, prefix digest (roots/functions/
noise/pipelines), ordered parents/widths, raw/stream references, actual input matrix/
digest and fit/stage constants. Replay inputs then fits; upstream edits invalidate
descendants. Count retained intermediates. [FitSample:15–56; MLP:91–121; Root:108–118]

## Normalization and decision 13

Source MLP has ordered linears and start/end wrappers:
standardize → random shift/rescale → primitive → standardize. Std uses correction=0,
epsilon 1e-4 outside sqrt; shifts select one row/channel. Preserve formulas, placement,
stage inputs/constants. [UpFunction:76–111; UpActivation:15–28,176–202;
Calibration:128–143,283–347; MLP:91–121]

Keep certified-second-moment gates at EVERY normalization, including MLP/pipeline/
future target stages. Finite fits are not certificates. Improve justified analytic/
support bounds; otherwise UNKNOWN stays UNKNOWN. Reject incompatible compositions
with reasons, without retries, clipping, plain-activation substitution or family
blacklists. Preserve all eight families and complete failure accounting. Earlier
finite-fit policy proposal is NOT accepted fallback.
[Analysis:62–90,124–135; Spec:19–23,81–87; Earlier:160–184]

Named source `log_clip_abs`/Kumaraswamy clamps are explicit laws, not generic graph
clipping authority. Heavy-tailed/multimodal roots remain beyond the four-base scope.
[UpActivation:57–58,205–217; RootPlan:15–18; Spec:81–87]

## Admission and atomic commit

Runtime and standalone admission do not cover the sampled envelope; child reports
duplicate ledger costs. Traverse actual envelope with `function_record_cost`.
[Graph:107–169; Admission:14–48,54–80,218–260; RootAdmission:19–78]

Two phases: bounded plan arithmetic BEFORE raw arrays/tensor conversions/reconstructing
constructors; exact retained admission before final validation. Reserve ALL samples,
realizations, ledger/registry, owners/manifest/bindings, snapshots and transaction
copies. Valid children can fail aggregate caps. Caller limits dominate payload; bound
containers/depth/products before allocation. [Root:108–112,148–151; Admission:18–48; RegistryDecode:29–68]

Peak = retained records + largest live prefix/concat + fitter + validation workspace
+ snapshot/serialization copies. Count EVERY receiving input, root fit output, MLP
stage input/output and serialized raw/derived duplicate; only canonical references
remove duplicate storage. Include tree t·2^h·o, quadratic o·(min(i,20)+1)^2, RFF, EM
rows×components, covariance d², certificate sets and trace clones. No fictional
affine i×o matrix for other families; estimates are not RSS.
[Admission:118–138,147–195,264–298; RootAdmission:54–77; Graph:131–158; Pipe:162–164]

**Hazard (source deduction, NOT_RUN):** transaction exit commits RNG/cache/ledger.
Registry freeze rejects live children/busy outer contexts. Thus final validation
after `with caller.transaction("graph")` can fail AFTER caller mutation; trying
to freeze registry inside the child fails. [Context:78–82,117–154,235–242]

| Design | Contract |
| --- | --- |
| Builder-owned staging (recommended) | Fresh context uses clones of all four RNGs. Exit commits only clones; freeze idle registry; validate full graph; publish caller states last. Discard private caches/ledger on failure. No change to registry API. [Context:40–76,100–154,235–242] |
| Extend context sealing | Prepare/seal or validation callback closes provisional spans and validates candidate registry/result BEFORE commit. Needed for preexisting-context cache continuity. Test nesting, span closure, busy handles and callback reentrancy. Simply permitting child snapshots leaves ancestor spans unfinished. [Context:97–154; Registry:73–106] |

API takes RNGs/policies, not a context; cross-graph shared caches require sealing.
[Context:59–76,105–115,141–149]

Commit: validate distinct/bounded caller states; save/clone; build using staged
streams; exit; freeze registry; validate ownership/raw/receipts; construct runtime
(including moments); admit envelope; prepare return/state copies; verify callers
unchanged; publish all four states; return. Nothing fallible in graph validation
follows publication. Require exclusive caller ownership, prevalidate same-type
state assignments, restore saved states if a publication setter fails. Reject
concurrent caller mutation without rewinding another actor. Test partial publication;
never reuse closed context handles. [Streams:41–64; Context:123–154,244–252; Root:180–184]

Failure receipts consume no graph RNG: retain phase/owner/reason and available
digests. Same-state retry reproduces intrinsic failure, never redraw-until-pass.
[Spec:19–23,125–129; Eval:102–103]

## Sources

Citation keys use pinned lines under `src/r0/modules/priors/lib/` unless specified:

```text
Graph functional_causal.py             Eval functional_causal_evaluation.py
Context joint_sampling.py              Registry joint_operation_registry.py
RegistryDecode joint_registry_decoding.py  Streams joint_streams.py
FPlan joint_function_plans.py          Function joint_function_realization.py
RootPlan joint_root_plans.py           Root joint_root_realization.py
RootBase joint_root_base.py            FitSample joint_function_calibration.py
MLP joint_mlp_realization.py           Matrix joint_matrix_realization.py
RFF joint_rff_realization.py           Tree joint_tree_realization.py
Disc joint_discretization_realization.py  EM joint_em_realization.py
Weights joint_weight_plans.py          Points joint_point_realization.py
Admission joint_function_admission.py RootAdmission joint_root_admission.py
Calibration calibrated_activations.py Analysis functional_analysis.py
Pipe node_pipeline.py                 FitPlan joint_plan_calibration.py
Exo exogenous.py                      OldFit source_causal_calibration.py
Spec docs/specs/0028-causal-network-generators.md
Earlier docs/research/0028-joint-graph-integration.md
History docs/issue-28-functional-graph-integration.md
UpFunction /tmp/r0-28-source-read/I2func.py
UpPoints /tmp/node-pipeline-sampling-research/_points.py
UpActivation /tmp/node-pipeline-sampling-research/_activation.py
UpBase /tmp/node-pipeline-sampling-research/_base.py
```

Cached function/points attribution to TabICL commit
`9530bd65a7b806f6ad4f9f4335af745577e9d3cc` is inherited, not independently reverified.
No upstream retrieval this turn. [Earlier:275–284] Executed commands include
`pwd`, `git rev-parse HEAD`, `git status --short`, `nl -ba` source reads and:

```sh
sha256sum /tmp/r0-28-source-read/I2func.py /tmp/node-pipeline-sampling-research/_points.py /tmp/node-pipeline-sampling-research/_activation.py /tmp/node-pipeline-sampling-research/_base.py
```

Checksum stdout:

```text
ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c  /tmp/r0-28-source-read/I2func.py
04dcbd7664004f6acc8453ca866caba0879d87d4ecbcdad08dd79f9ca2ff7a04  /tmp/node-pipeline-sampling-research/_points.py
11b8bca1e34d11725bbe2a19a229059840fe14ca0f604cbe864c07987bb9f51a  /tmp/node-pipeline-sampling-research/_activation.py
065ed93d700fb125222ee60b68656eb419a39c630645d1b8f44fe9d6baf718d7  /tmp/node-pipeline-sampling-research/_base.py
```

## Implementation priorities and oracles

Checks **NOT_RUN**; retain HEAD, commands, states, outputs/failures.

1. Extract shared prefix admission/kernel. Literal roots `(2,3)` through matrix rows
   `(1,0),(0,1),(1,-1)` give `(2,3,-1)`; child input width 3. Root square at 2
   then ×3 gives 12, not 6. Preserve authored generation/inspection.
2. Implement plans/owners/staging. Inject failures in planning, after root/descendant
   fit, AFTER transaction exit during registry validation, final moments/admission,
   and partial RNG publication. Four caller states remain byte-identical. Remove
   only injected fault: retry equals clean same-state build (raw/constants/registry).
   Intrinsic invalid draws reproduce the same failure.
3. Add raw recorder/fresh-fit receipts. Tree threshold 0, leaves 0/10: own-fit roots
   `(-1,1)` give `(0,10)`; receiver fresh roots `(2,3)` must give `(10,10)`. Each
   receiver gets new draws. Reject stale prefix/order/noise/pipeline, old fit under
   new ID, swapped roles, forged digest, extra/missing empty scopes and changed
   kernel Gaussian with unchanged sign (binding/RNG replay).
4. Add pipelines/moment gates. f(2)=4; scale=3, offset=1, strength=2, power=1,
   innovation=2 →58. Signal=2, channel importance=3, converter +1, innovation=5,
   node importance=2: historical-tuple state/readout=24/6; historical-graph=24/7;
   readout-only=22/7. Child uses state. Cover every MLP start/end placement and
   finite-fit/uncertified-moment compositions without policy downgrade.
5. Aggregate admission before arrays: many individually valid fits exceed total cap.
   Observe rejection allocations for huge geometry/depth, duplicate fits, forged
   looser payload caps and covariance roots; no caller RNG movement.
6. Shared generation/archive/inspection: 4×8 source root/function coverage, all eight
   heterogeneous nonroots, both cheap-product positions, covariance bases, later
   pipelines/noise, ties/numerical failures. Reopen/replay repeated and reordered
   rows. Emission/query counts cannot change fit constants. Independent exact-head,
   measured resource, CPU/GPU, browser/Colab evidence remain separate. Full graph/
   network/typed-variable/intervention inventory remains active. [Spec:28–64,105–167]

Oracle equations: [Function:248–260; Eval:78–101; Pipe:166–214]. Fresh-fit trap is
authored, unexecuted.
