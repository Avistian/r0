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

Still required: actual browser and exported numeric verification; explicit target
edit/comparison controls and paired persistence; response/reference semantics;
hybrid DAG roots; broader population/location catalog coverage; Colab/shared
callback coverage; resource/diversity qualification. This checkpoint is not
complete issue #28 or complete workbench acceptance. The source-related core
contract is in [GP column populations](issue-28-gp-column-populations.md).
