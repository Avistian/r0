# Issue 28: sampled functional graph assembly

Status: implementation candidate, not independently accepted. Full #28 remains
unfinished. This extends the root/function and prefix capture work published at
`1b547ec0433d9fa7a6e623668dbd7a49bd22d328`.

## Public boundary

`r0.modules.priors.JointGraphRequest` supplies an ordered DAG, channel widths,
task readouts, root/function priors and resource limits.
`sample_joint_graph(request, named_rng=..., raw_rng=..., calibration_rng=...,
fit_rng=...)` requires four distinct injected NumPy generators. It returns a
`SampledFunctionalGraph` containing the request, plan, completed operation
registry, typed plan owners, realizations, calibration receipts, raw transcript,
initial/final generator snapshots, admission estimate and frozen runtime.

The current version samples mechanisms. Topology, widths, readouts, structural
noise descriptors and optional `authored_pipelines` are explicit inputs. They
are not represented as sampled decisions. Named parameter contracts remain
shared across the graph: root and nonroot function catalogs using the same
parameter name must agree. Conflicting catalogs fail without resetting caches.

All eight function families are supported: linear, quadratic, random Fourier,
oblivious tree, discretization, EM, MLP and cheap-function products. Root bases
are Gaussian, uniform, ball and covariance. Roots retain the existing source
width-preserving contract; heterogeneous nonroot widths concatenate parent
states in their declared order.

Every fitted nonroot receives a new calibration population through the frozen
prefix. This includes prior root laws, functions, structural innovations and
pipelines. It does not reuse ancestor own-fit outputs. Version one evaluates
all earlier nodes, including earlier nonancestors. Receipts identify the actual
receiving fit, ordered parents, raw calibration span and retained input digest.
The realization owns the numerical calibration matrix. Pipeline fits supplied
by the caller remain authored inputs; sampling fresh pipeline fits is pending.

## Replay and publication

All construction happens on private clones of all four caller streams. The
named registry closes before validation. The complete runtime and retained
record envelope are admitted and their bindings checked before returning from
the outer transaction and publishing caller states.

`graph.validate()` performs complete replay. `graph.replay()` restores bounded,
canonical JSON states and regenerates the graph, including named decisions,
raw values/roles/order, fits, runtime and final states. Equality with the
retained construction is required. Replay never consumes caller generators.
No archive state is unpickled. NumPy version and generator identity are bound;
PCG64, PCG64DXSM, Philox, SFC64 and MT19937 are supported, including noninitial
buffered states. Structural validation alone is not a provenance proof.

Admission conservatively sums existing family-specific construction reports,
multiple retained/replay copies, prefix buffers, certificate storage and
snapshot space. The raw ledger receives only the remaining budget. The final
whole-envelope pass counts duplicate serialized metadata and equations. These
bounds are estimates, not measured RSS or OS-enforced limits. Independent
review must scrutinize tight aggregate budgets and failure atomicity, including
publication setter failures; current tests establish late recording-budget
rollback, not every possible publication fault.

## Verification and reproduction

The focused suite tests eight function families, four roots across all five
engines, noninitial generator states, altered raw roles/values, receipts,
snapshots, runtime equations and typed plan owners. A scalar hand equation
checks fresh root draws, reversed parent order and structural noise/pipeline
state propagation rather than captured readouts. Changing emission row count
must preserve fitted constants and construction draws.

Use the existing environment; no installation or downloads are required:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -ra \
  tests/priors/test_joint_graph_plans.py \
  tests/priors/test_sampled_functional_graph.py

UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

Author evidence currently retained locally:

- Initial runtime RED: `/tmp/r0-28-sampled-graph-runtime-red.log`, nine failures
  because the public assembly boundary did not exist.
- Focused graph plans/runtime: `/tmp/r0-28-sampled-graph-runtime-binding.log`,
  205 passed in 15.05 seconds. This precedes a type-check-required None guard;
  the full suite is checking that final source.
- Full check: `/tmp/r0-28-sampled-graph-check-first.log`, running at the time
  this note was authored. Static gates passed; no final pytest claim yet.
- Initial expanded test failure is retained in
  `/tmp/r0-28-sampled-graph-runtime-expanded.log`: nine tests requested
  conflicting named catalogs. Corrected fixtures use a common `lin,tree`
  catalog and fixed named seed 13, producing two linear roots and a tree child.
  Fixture selection is not production retry or rejection sampling.

These local paths are not a published portable evidence archive. Independent
review, exact committed-head verification and evidence packaging are pending.
Training/evaluation smoke evidence is unavailable until the train/eval CLI
lands. No scientific, browser, Colab, GPU or measured resource claim is made.

## Remaining integration and full issue scope

This candidate does not complete sampled card/archive decoding or generation
and inspection integration. Typed construction owners, source topology/width/
noise/pipeline/readout priors, independent receiving pipeline fits, generalized
root dimensions and additional paper options remain work. Native categorical
variables, interventions, the local/Colab workbench and required runtime and
scientific evidence also remain in #28. The authoritative scope remains
`docs/specs/0028-causal-network-generators.md`; this checkpoint does not replace
or narrow it.
