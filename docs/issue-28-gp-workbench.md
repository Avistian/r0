# Finite GP population workbench

The workbench exposes 18 construction presets: linear, quadratic and neural
frozen targets, regression/binary/multiclass adapters, and shared/per-column RBF
kernels. All population geometry, location families, kernels and target settings
remain editable in the construction JSON. These use the same public GP generator
and retain its complete request, locations, factors, innovations and independent
calibration in exported tasks. These are finite joint feature populations, not a
function that can independently evaluate arbitrary new input rows.

Construction, archive replay and graph inspection run in the bounded worker.
The generic construction seed is unused for this request-only adapter; episode
seed and task identity determine the population. Next is labelled **Next GP
population + target**: it retains the request, explicitly realizes a new joint
population and target, and leaves the original archive replayable. Ordinary replay
uses the exact original identity and realized population. Changing task geometry
requires explicit construction controls rather than silently resizing an archive.

The shared generation-run admission now recognizes a request that produces its
strictly replayable realized GP envelope. It checks exact transport identities,
complete task/request equality, and original realized custody when supplied.
This also enables the local guarded generation runner for GP request recipes.
Other recipe-family admission remains unchanged.

Author evidence is ignored under `docs/evidence/issue-28-gp-workbench/`:
initial missing preset RED, then a real producer/request realization mismatch,
then parent-process covariance replay RED. The corresponding end-to-end tests
now pass, including exact large task IDs, repeated-row joint-axis inspection,
parent Cholesky sentinel, export, independent Next and original replay. The
combined GP/workbench run passes47 tests. Invalid geometry, timeout and caller
memory limits preserve the accepted active task. Static checks and final commit
status are recorded separately; no full-suite claim applies yet.

Actual installed Chromium at `a036dde` passed construction, all seven selectable
trace stages, large exact IDs, export, Next and original replay with zero
exceptions. Its downloaded16-row ZIP passed independent RBF Cholesky, quadratic
target and calibration arithmetic. A subsequent display-only repair labels
features as joint GP columns rather than incorrectly calling them irrelevant;
its public test was RED before the repair and GREEN afterward. Browser results
remain tied to the preceding source; independent review is pending.

Still required beyond the target-edit delivery below: hybrid DAG roots; broader population/location catalog coverage; Colab/shared
callback coverage; resource/diversity qualification. This checkpoint is not
complete issue #28 or complete workbench acceptance. The source-related core
contract is in [GP column populations](issue-28-gp-column-populations.md).

Independent review of `111284c` rejected producer admission ordering: mismatched
request target family or task noise triggered covariance replay before rejection.
The follow-up repair compares complete task/request/sampling metadata, exact
outer and nested identities and any originally realized source before replay.
The original public probe now observes zero Cholesky calls for wrong producers;
a valid control still completes. Five public cases retain three genuine baseline
failures (two already rejected early). The combined52-case GP/workbench/admission
check passes. Historical rejection remains retained; independent repair review
and broader UI review remain pending.

## Frozen target reference

`evaluate_gp_column_reference` evaluates a one-column grid with all other inputs
fixed to a retained original row. The raw-record counterpart validates complete
source metadata, controls and aggregate source-replay/query budgets before any
covariance replay or grid allocation. The original population, target parameters,
independent calibration, seed and task ID stay unchanged. Query points never fit
or update normalization. This CPU reference is a target-equation counterfactual,
not a conditional GP feature posterior, fresh feature population or general DAG
do-operator. Combinations can be outside joint feature support even within the
observed marginal range; explicit bounds may extrapolate.

The returned raw target is `f(x)`, and the normalized score uses the original
`(f(x) - centers) / scales`. Regression reports this score as the expected target
under independent zero-mean Gaussian observation noise, with the original
`noise_ratio` as its standard deviation. Classification reports the shared task
adapter's sigmoid/softmax probabilities including separation and class offsets;
no labels or observation noise are drawn. A single binary score is the positive
class score. All three target families and task adapters use the same semantics.

The explorer exposes raw, normalized and expected-target/class-probability views,
original-row anchors, grid resolution, optional explicit bounds and JSON export.
Replay and evaluation stay in the bounded worker. Failed queries and timeouts
preserve the original active task and saved archive. Original population identity,
frozen target and calibration constants remain visible in the reference artifact.
Generic task-pair, mechanism-editor and DAG-policy controls are marked unavailable
for this finite-population family. Dedicated comparison-environment curves/further linked edits, hybrid graphs,
full covariance interactive views and hosted Colab
verification remain REQUIRED follow-ons, not excluded scope.


## Target equation comparison environments

`generate_gp_column_pair` exposes every authored linear, full quadratic and tanh
neural parameter through `GPColumnTarget`. Its two `GPColumnTargetRecipe` sides
retain the complete original `GPColumnPopulation`, replacement target, comparison
mode/fresh seed, selected feature draw, independent calibration and before/after
scores. This versioned environment is separate from an original population: it
never forges population identity or the original sampling request to disguise an edit.

Matched mode reproduces original feature and adapter streams exactly on the
before side. Fresh mode retains original locations/kernels but draws fresh feature
innovations under its separate exact seed. Both sides share those inputs and the
same freshly keyed observation-noise, class-uniform and split streams. Fresh before
is therefore the original equation on the fresh environment, not the active baseline.
Each target uses the original independent marginal calibration inputs; replacement
centers/scales are fitted once there. No query fitting occurs, and a bias-only change
may cancel after this normalization. This is a target-equation comparison, not a
conditional feature posterior or a general DAG do-operator.

The explorer offers all three target families, editable complete parameter JSON,
matched/fresh controls and explicitly authored deterministic r0 starting drafts.
Its guarded save action preserves the original active archive and any previous
comparison on malformed controls, resource failure or timeout. Both actual comparison
sides appear in the shared panels; repeated-original-row traces retain full covariance
row dependencies. The existing generic pair container (`save_causal_pair` and
`load_causal_pair`, a historical API name) binds both environment ZIPs with a distinct
`gp-target-edit` card. Reopen, exact-identity replay, task export and paired ZIP export
use the same guarded local/notebook callback. Actual hosted Colab remains NOT_RUN.

Metadata validates outer task geometry, exact identities, mode/fresh seed, complete
source/target shapes and combined admission before source replay. Admission reserves
the existing comparison/replay budget plus four task-buffer sets (two retained sides
and two validation copies), including inputs, scores, probabilities, targets and split
indices. Parameter and working-byte caps use the same formula in typed construction
and strict decoding. Supplied derived arrays are still checked against numerical replay;
metadata admission is not a substitute for provenance validation.

Comparison envelopes are fixed-identity environments. Their Next control is disabled
and named accordingly; original population Next retains its new-population meaning.
Dedicated reference curves and further linked edits for an already reopened comparison
remain REQUIRED follow-ons, explicitly separate from the available original-population
reference and original-to-replacement editor. Hybrid roots, full covariance interactive
views, broad population catalogs and resource/diversity qualification remain required.
