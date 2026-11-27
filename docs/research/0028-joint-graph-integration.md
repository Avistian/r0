# Issue 28: joint functions inside complete causal tasks

Design research, 2026-11-27, HEAD `de1bead86d51721fd84d84598b7bea2800ec8885`,
`/tmp/r0-28-graph-integration`, branch `feat/issue-28-graph-integration`. Only this
note authored; existing `.venv` untouched. Checks **NOT_RUN**, resources
**NOT_MEASURED**. Facts are inspected code; recommendations are proposed work.

## Recommendation and current boundary

Implement typed functional nodes, shared evaluation/analysis and a sampled graph
recipe owning complete construction. Deliver all eight functions/four bases through
generation/save/reopen/inspection, without dummy weights or the Gaussian fitter.
Accepted scope requires API/local/Colab delivery and domain safeguards. [S:14–23,36–42,54–80]

Facts at HEAD:

| Contract | Boundary |
| --- | --- |
| Nodes/recipe | Bias/weights define widths; five exact types accepted. Root has no function field. [N:24–67,150–208; C:66–150] |
| Helpers | Eight evaluators, dimensioned plans, two cheap same-input product children. Four root bases; realization binds fit inputs/function/outputs, emission never refits. [FE:22–42,148–206; FP:27–55,85–105; RB:18–84; RR:58–184; RE:92–126] |
| Transaction | Named/auxiliary RNGs stage together; event-free registry scopes/spans exist, raw distribution/destination binding does not. [JS:81–154; OR:47–106] |
| Runtime | Affine response drives noise. Pipeline captures readouts before noise/node importance; pipeline recipe requires authored activated nodes. [E:39–94; ST:23–79; P:166–214; CP:23–40] |
| Fitter | Gaussian affine ancestors; plain mode skips ancestor evaluation. Non-Matrix nodes become ActivatedCausalNode. Validator replays this. [SC:53–108; SR:154–172] |
| Consumers | Ancestry/moments/cost assume affine projection; inspection describes affine equations. Root admission is not graph admission. [C:225–281; A:24–79; M:99–190; I:28–158; L:29–102; RA:19–78] |

Inference: attaching root functions after source fitting leaves descendant constants
stale. An evaluator-only change leaves other contracts wrong; standalone functions
do not establish graph integration. [SC:53–108; C:71–84]

## Minimal interface, dimensions and provenance

Recommendation: closed tagged node union, preserving legacy records/seeds; no dummy
affine fields. Centralize `input_width`, `output_width`, `evaluate_signal`, noise,
dependency/support/moment/cost analysis and bounded decoding; replace affine
assumptions throughout graph consumers. [N:24–67; C:71–84; L:41–52]

Proposed records (names illustrative):

```text
FunctionalCausalNode(parents, function: FrozenFunction, innovation: InnovationSpec)
RootFunctionalCausalNode(base: JointRootBase, function: FrozenFunction, innovation)
JointGraphPlan(topology, ordered_node_plans, pipelines, readouts, stream_roles,
               calibration_schedule, policies, source_identity)
JointGraphConstruction(plan, registry, realizations, raw_manifest,
                       calibration_receipts, aggregate_admission)
JointCausalRecipe(task, nodes, pipelines, observed, target, construction_record)
```

Nonroot input width equals concatenated parent **state** widths; output width is
independent. Authored roots can support different base/output widths when their
function agrees; source-sampled `JointRootPlan` keeps its existing d→d restriction.
Root and innovation widths differ when the function changes dimension. Reuse
`FrozenFunction`, not an activation disguised as an MLP/tree. [FP:29–32; RR:79–118]

Authored equations make no source-draw claim. Sampled equations bind complete
realizations: decisions/raw inputs/fits/function. Check runtime references/digests
against actual owner contents. Canonicalize storage or count duplicates; never
invent draws for authored values. [RR:67–122; FD:119–131]

Use a new recipe version/construction tag. Generalize `CausalRecipe`'s seam; do not
inherit source fitter behavior or authored-only pipeline restrictions. New metadata
should say `joint_sampling`/`graph_construction`, not `activation_sampling` (the
current card field). Preserve v1/v2 readers and behavior. [C:90–120,358–361; CP:23–40]

Per-parent mechanisms need typed parent slices/functions and explicit aggregation
(sum/product/average/max/logsumexp/neural), distinct from same-input cheap product.
These, categorical states/converters and other graph families remain required.
[S:28–47,105–121; FP:85–105]

## One equation for fitting, generation and reconstruction

Recommendation: factor a pure `evaluate_node(node, parent_state, root_input,
innovation, pipeline, trace)` shared by builder and runtime. Define the new law:

```text
u = f(concat(parent_states))             # nonroot
u = f(frozen_base(raw_root_draws))        # root
scale = noise_scale * (offset + strength * abs(u)**power)
v, captured = pipeline.before_noise(u)
state = node_importance * (v + scale * raw_innovation)
readout[j] = captured[j] if captured else state[j]
```

Version response to actual function output `u`; extract stable scaling from
`stochastic_value`. Specify any bias/activation placement. Preserve legacy noise
orders; new positions require shape/fit/analysis rules, never unequal-width
broadcasting. [ST:23–79; P:166–214]

Root outputs are its own fit evaluations. Freeze roots before pipeline/descendant
fit; use fresh prefix populations downstream. [RR:101–118,163–178]

Recommendation: version these stream roles and include every draw in graph ownership:

| Role | Use |
| --- | --- |
| named-choice | Shared context numerical/categorical names and catalogs. |
| construction-raw | Factors/matrices/leaves/phases/coefficients/jitter. |
| root-function-fit-input/root | Base sample realizing that root function. |
| function-fit/node | Fresh frozen ancestor-prefix evaluation; parent states become function fit inputs. |
| pipeline-fit/node | Another fresh prefix population evaluated through the now-frozen local function, then outer pipeline fitted sequentially. |
| fit-choice | Row choices/shifts, with distinct owner paths. |
| fit-innovation/purpose/node | Prefix innovations, separate from roots/emission. |
| target-fit | Separate frozen-graph population for optional target calibration. |
| emitted-root / emitted-innovation | Dataset rows; independent of all fits, keyed for original-row/intervention replay. |

Use bounded role streams/derived identities, not one live RNG per node: the
64-stream default must not cap graph nodes. Account for derived state in the outer
transaction. Different IDs alone do not prove independence.
[JS:81–154,244–252; JStreams:11–16,41–64; FC:38–56]

Inference: reusing ancestor fit samples creates adaptive in-sample values (notably
thresholds/centers). Fresh receiving-fit samples follow the frozen prefix law;
name this adaptation. Inside standalone function fits preserve source order:
MLP stages see preceding outputs; product children share inputs. Outer pipeline
stages likewise fit sequentially. [MLP:144–178; FP:94–102]

Receipts bind role/rows/digest, prefix fingerprint, parent order, upstream
pipelines/converters/noise, function owner and stage input/output/constants.
Upstream changes invalidate descendant fits. Emission never refits; validation
may reconstruct retained data. Admit fresh-prefix costs; query counts/labels
never choose fit rows/streams. [FC:15–56; SR:154–172; S:68–80]

## Potential dependence and honest population claims

Recommendation: one dependency engine returns input sets and exact constant/zero
proofs, composed through state/readout capture. Sample zero variance/derivatives
cannot prune paths. Overcoverage means potential dependence. [A:15–79; C:225–305]

Mathematical deductions from the frozen equations, conditional on finite frozen
parameters; these are proposed transfer rules, not executed certificates: [FE:148–206]

| Family | Dependence and moments |
| --- | --- |
| Linear | Nonzero entries select input paths; minimum participating input moment bound. Zero row proves constant. |
| Quadratic | Inspect selected nonzero polynomial terms, including cross/constant terms; conservative degree-2 bound q/2. Symbolic cancellation may improve it; sample cancellation cannot. |
| RFF | Nonzero frequency/coefficient paths; output bounded by sum of absolute coefficients, hence all moments on its valid finite input domain. |
| Tree | Conservatively retain tested features; finite leaf range bounds output. Identical reachable leaves may prove constant. |
| Discretization | Distance coordinates potentially affect selection; finite center-target range bounds output. |
| EM | Distance coordinates affect assignments; convex combinations of finite output-map columns bound output. Numerical logits can still fail and must reject that evaluation. |
| Product | Union children. Shared inputs are dependent: Hölder gives 1/q=1/q_left+1/q_right; bounded children consume no finite moment budget. Exactly zero child proves zero. |
| MLP | Compose linears and true calibrated stages; check every internal normalization site, not only final output. |

Root Gaussian/finite-factor Gaussian has all polynomial moments; uniform, ball
and finite-factor uniform are bounded. Singular/zero factors are valid. This is
conditional on frozen recipe parameters, not a claim about the mixture over
random recipes. [RB:28–84]

Existing bounds certify orders strictly below their bound; exp/log zero means
UNKNOWN, not universally invalid. Recommendation: add support/constant/tail facts.
Bounded exp is bounded; affine-Gaussian exp has all polynomial moments, Gaussian
quadratic exp may fail. Log/negative powers require support separation or justified
integrability. Clipped operators are distinct laws; finite sample extrema do not
prove support. [M:43–57,99–110; S:36–42]

Literal deduction: for Z~N(0,1), Y=exp(aZ²), E[Y²] is finite iff a<1/4, by
integrating exp(-(1/2-2a)z²). Finite samples at a=1/4 cannot certify variance.
For Z∈[-1,1], the same expression is bounded for every finite a. Fitted affine
standardization cannot repair an infinite second moment.

Recommendation: make normalization claims explicit in new versioned profiles:

* `certified-population-second-v1`: keep current strict gate at every required
  input/output site, including MLP internals and actual target readout. UNKNOWN
  cannot become certified through finite fitting.
* `frozen-finite-fit-affine-v1`: preserve source fitted affine transformations as
  deterministic finite-calibration operators; report population moments UNKNOWN
  where necessary and make no population-standardization claim. Select this
  named adaptation explicitly in the new source-oriented profile so every
  function selector remains reachable. Finite arithmetic failures stay recorded,
  without retries.

Explicit type boundary: `CertifiedNormalization` requires analytic/support-backed
second moments at each normalization site; `FrozenCalibrationAffine` guarantees
only retained finite constants, independent fit inputs, fixed replay and finite
successful evaluations. It does not certify moments. These are proposed new types,
not existing behavior. Never downgrade failed certified draws, clip silently,
blacklist accepted families or substitute linear. Preserve legacy profiles.

Spec tension: decision 13 requires compatibility rules including finite moments
for normalization. Treating empirical affine transforms as distinct operators is
an interpretation, not established spec acceptance. Implement certified mode
unchanged; explicitly label/independently review this new source-profile contract
against decision 13 before claiming compliant normalization. UNKNOWN stays UNKNOWN,
including when every evaluated row is finite. [M:73–85; C:158–170,335–345; S:81–87]

## Typed graph ownership, raw manifest and aggregate bounds

Recommendation: one outer operation stages plan/admission/realization/fits/receipts/
validation/freeze. Helpers use `stream(name)`. Exceptions discard graph and preserve
caller streams/caches/registry; failure receipts are separate. Current transaction
machinery is not proof of future builder atomicity. [JS:81–154,244–252]

Traverse typed plans to derive the expected operation tree, event ownership,
child order and shapes; compare with registry, including event-free operations.
Registry alone cannot establish raw array law or ownership. [OR:47–106; FP:85–105]

Manifest entries need `owner_path, role, stream_identity, distribution+parameters,
shape, destination, raw_digest, derivation_version, input_references, derived_digest`.
Bind selectors→plans, raw draws→realization fields, and realized functions→nodes.
Derived entries identify transpose/normalization/clipping/constants, without
inventing draws. Reject orphan/duplicate owners or destinations, aliases and
wrong-role arrays. Parent containment is not duplicate leaf ownership. Uniform
weights and raw factors still require proof beyond event-free registry presence.
[RB:43–60; RR:88–118; OR:85–106]

Separate graph geometry from affine parameter cost; no imaginary i×o matrix.
Aggregate actual serialized construction/raw/fit/ledger/registry/receipt copies,
calibration and stream snapshots, live prefixes/concats/workspace/traces and
transaction copies before allocation. Avoid double-counting helper ledgers.
[L:41–95; RA:19–78; RE:58–89]

Count tree t·2^h·o, quadratic o·k², RFF i·K+K·o/R·K work, distance R·c·i,
MLP layers, simultaneous product outputs, covariance d² and fit populations.
Bound products/depth/nodes/containers/bytes before conversion. Caller caps override
payload; valid children can fail aggregate admission. Preserve bounded decoders;
estimates are not RSS. [FE:46–75; FD:119–131; L:84–102]

## Implementation chunks that reach delivery

Recommendations in dependency order:

1. **Authored graph vertical slice.** Tagged interfaces, all eight functions/four
   bases, unequal widths, evaluator/analysis/admission/decode/inspection. Exit:
   actual heterogeneous `generate_task`, selected-row JSON replay, unchanged legacy
   records; no sampled-source claim. [D:78–143,204–207; C:174–224,360–361]
2. **Joint sampled graph.** Typed ownership/manifest/transaction/fresh-prefix fits,
   all 32 root combinations, heterogeneous nonroots, policies/profiles. Exit:
   `sample_recipe → generate_task`, failure accounting, adversarial decode/rollback.
   [RR:125–184; Recipes:300–355; Requests:65–103]
3. **Archive/workbench.** Register profile/examples/API/decoder; persist complete
   construction. Drill down base/function stages/noise/pipeline/state/readout;
   distinguish fit data/emitted rows and population UNKNOWN/sample diagnostics.
   Exit: guarded generate/save/load/replay/export, actual local-browser and shared
   Colab session. Unexecuted checks remain PENDING. [Saved:30–69; Load:26–72;
   Session:129–155; Notebook:13–43; S:122–131]
4. **Continue full scope.** Per-parent aggregation, categorical states/converters,
   interventions/editor, wider graphs/networks remain. Match unaffected random
   rows for interventions, retain baseline, assign changed mechanism new identity
   and fit receipts. First functional task is not #28 closure. [S:28–121,140–160]

## Literal oracles and regression cases

Proposed checks **NOT_RUN**, specified independently of implementation output:

| Oracle | Expected result |
| --- | --- |
| Dimension change | Parent (2,3), linear rows (1,0),(0,1),(1,-1) → (2,3,-1). Child receives width 3. |
| Nonlinear ancestor | Root base x=2, quadratic f=x² →4; child f=3y →12. Raw-root/affine shortcut gives wrong 6. |
| Fresh-fit trap | Tree threshold 0, leaves 0/10: fit inputs (-1,1) give (0,10); fresh roots (2,3) give (10,10). Descendant receipt must bind fresh values. |
| Converter order | Function 2, channel importance 3→6, converter 6→7, noise 5, node importance 2: historical-tuple state/readout=24/6; historical-graph=24/7; readout-only=22/7. Child consumes state. |
| Signal-dependent noise | f(2)=4, offset=1,strength=2,power=1,scale=3 →27; raw innovation 2→54; output 58. Affine surrogate f=2 gives wrong 32. |
| Dependent product | X~Student-t(3), children identity: X² has moment bound 3/2, not 3. Cannot certify second moment. |
| Quadratic/tree | z=(2,1), Q=((1,3),(5,7)) →27. Tree dims (0,1), thresholds (1,2), leaves 10/20/30/40, input (1,3) →30 (strict split, low bit first). |
| Root factors | A=((1,2),(3,4)), w=(1/2,1/4), B=(2,-4): B(A diag(w))ᵀ=(-1,-1). Ball d=2, direction (3,4), U=1/4 →(0.3,0.4). |
| Sample-zero path | x² fitted on (-1,1) has constant sample output yet potential dependence. Exact zero linear map can prune it. |
| Moment support | exp(Z²/4) has infinite second moment for Gaussian Z despite finite fit; bounded-domain case can be certified. |

Equation sources: [FE:148–206; ST:37–79; P:166–214; RB:63–84].
Require 4×8 root coverage, each function as dimension-changing nonroot, both cheap
product positions, MLP placement cases, covariance bases, converter/noise roles,
constant/singular/weak valid draws, distance ties and numerical failures.
[S:19–23,68–80; FP:55–56,85–105; RB:28–60]

Adversarial records: wrong width/parent order, stale ancestor digest, forged fit
identity, mismatched root function, changed raw factor with unchanged derived
function, orphan event-free operation, duplicate raw owner, global overflow despite
valid children, invalid version/type/depth. Inject failure after root realization,
descendant fit and final validation: verify every RNG/context rolls back and failure
reason survives. Check arbitrary selected-row order, save/reopen, fixed-fit
emission-count/query independence, matched interventions, CPU/GPU tolerances,
actual browser/Colab and measured resources. [RR:67–122; JS:100–154; I:28–53; S:122–138]

## Provenance

Cached primary code, local SHA256 checked, no fresh upstream verification:
`/tmp/r0-28-source-read/I2func.py`:
`ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c`;
`/tmp/node-pipeline-sampling-research/_points.py`:
`04dcbd7664004f6acc8453ca866caba0879d87d4ecbcdad08dd79f9ca2ff7a04`.
Earlier extraction attributes these to TabICL commit
`9530bd65a7b806f6ad4f9f4335af745577e9d3cc`; [_function.py:12–74][UpF] and
[_points.py:7–44][UpP] establish selectors and base→RandomFunction structure.
[Earlier design](0028-joint-root-function-plans.md) read, not edited.

Citation key: ranges after `:` are source lines. Unless qualified below, filenames
are under `src/r0/modules/priors/lib/`.

```text
N causal_nodes.py           C causal.py
FE frozen_function_evaluation.py   FP joint_function_plans.py
RB joint_root_base.py       RR joint_root_realization.py
RE joint_root_emission.py   FC joint_function_calibration.py
JS joint_sampling.py       OR joint_operation_registry.py
E causal_evaluation.py     ST causal_stochastic.py
P node_pipeline.py         CP causal_pipeline.py
SC source_causal_calibration.py    SR source_causal.py
A pipeline_ancestry.py     M causal_moments.py
I causal_inspection.py     L causal_limits.py
RA joint_root_admission.py FD frozen_function_decoding.py
D generate.py             MLP joint_mlp_realization.py
JStreams joint_streams.py  Recipes recipe_sampling.py
S docs/specs/0028-causal-network-generators.md
Requests src/r0/modules/generation/lib/requests.py
Load src/r0/modules/generation/lib/loading.py
Saved src/r0/modules/explorer/lib/saved.py
Session src/r0/modules/explorer/lib/session.py
Notebook src/r0/modules/explorer/lib/notebook.py
```

[UpF]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_function.py#L12-L74
[UpP]: https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_points.py#L7-L44
