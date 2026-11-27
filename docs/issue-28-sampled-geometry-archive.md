# Complete sampled geometry archive candidate

`GeometryGraphRequest`, `sample_geometry_graph` and `SampledGeometryGraph` connect
sampled feature-node assignments and latent widths to topology and mechanism
construction. This closes the provenance gap between the geometry components and
the shared generator for numerical identity readouts. It is not completion of
all geometry, converter, categorical, intervention or workbench options in #28.

## Construction and scope

The request specifies a `CauchyGraphPrior`, a `LinearRecipe` task adapter,
`subsample`, `JointWidthPrior`, calibration rows, root/function priors, a broadcast
Gaussian structural `noise_scale`, named sampling policies, and causal/raw limits.
Features form ordered group `x`. Group `y` has one score channel for regression
or binary classification and one per class for the multiclass adapter. These
classification adapters do not implement native categorical graph variables.

Four caller-owned generators are staged atomically. Construction draws Cauchy
topology, named eligible-node counts and raw assignments, contiguous identity
converter spans, named latent widths, and mechanism plans. Every named decision
uses one `JointSamplingContext`, including the graph mechanisms. A single complete
operation registry retains the observation, width and graph owners. No scope is
reset or sliced, and foreign or missing owners are rejected.

Mechanisms use the realized ordered parents and full node widths, including
hidden channels. The existing materialization loop is reused: each fitted receiver
gets fresh independent ancestor calibration through all prior fixed mechanisms.
Identity numerical converters capture their declared readouts; node state and
structural innovation remain distinct. Converter sampling/calibration, native
categorical converters, and other topology families still need their own integration.

## Archive and validation

The new envelope is `sampled-geometry-graph-v1`; its request is
`geometry-graph-request-v1`. Existing sampled-functional and sampled-topology v1
schemas are unchanged. The new envelope retains topology, observation raw records,
channel layout, latent widths, the complete shared registry and mechanism/runtime
records, plus four initial/final snapshots. It contains a distinct
`materialized-joint-graph-v1` record, whose scope is owned by the outer sampler.
It cannot be exported as standalone explicit-DAG v1 provenance when geometry
scopes are present.

Structural validation checks all geometry bindings, exact operation ownership,
mechanism/runtime bindings, and raw/calibration/fit stream chains across topology,
observations and interleaved mechanism records through the final snapshots. Named
decisions are not raw-ledger draws: their preference provenance and final generator
state are checked by complete replay. Full replay reconstructs from all four initial
states and compares the entire envelope, including discarded topology/assignment
draws, fitted calibration records and final snapshots.

`recipe_from_dict`, `generate_task`, `explain_generation`, `paired_recipe`,
`generate_pair` and `compare_pair` support the new envelope. Saved tasks retain all
construction records. Larger exploration caps require explicit caller limits when
reopening, inspecting or comparing saved tasks. Only `context_rows` and `split`
are excluded from the three nested task locations used for source identity; they
remain in the archive and change episode identity. Actual task/mechanism changes
remain source changes. Verified paired controls are regression observation-noise ratio and classification
separation. Changing feature count resamples geometry and does not satisfy the
unchanged-mechanism pair contract. These task-level pairs are not causal interventions.

## Resource accounting

The request reserves the combined topology/observation raw-call count, all raw
values (including full permutations and square adjacency proposals), and retained
record/snapshot storage before mechanism sampling. The mechanism ledger receives
only the remainder. Conservative complete-geometry metadata and snapshot allowances
are deducted from causal limits before planning/materializing mechanisms. Whole
archive admission precedes reopening arrays and generator states. These are bounded
admission estimates, not measured peak resident memory or throughput.

## Author evidence and remaining gates

Evidence lives under ignored `docs/evidence/issue-28-geometry-archive/` in the
implementation worktree. Reproduce the focused suite with the existing environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_sampled_graph_archive.py \
  tests/priors/test_topology_graph_assembly.py tests/priors/test_sampled_geometry_graph.py
```

`materialization-extraction.log` retains 49 passing existing graph/topology cases.
`equivalence.py`, run both at base `60f13bafee12d03f6b26001e96c9db2b489b046c`
and the candidate, emits full serialized v1 graph hashes, generated source identities
and four next-stream-tail hashes for all eight function families. The complete
`equivalence-base.log` and `equivalence-new.log` match byte for byte; this checks
the materialization extraction and source-identity helper against the old behavior.

`geometry-first.log` contains eight initial successful integration cases. The first
advanced-engine run (`geometry-engines.log`) retained a PCG64DXSM/global MLP failure:
a sampled calibrated exponential lacked a finite-second-moment certificate. That
valid rejection is now an explicit four-stream rollback case, with no retry/filter
added to generation. `geometry-checked.log` contains 13 passing geometry cases,
including all task adapters, both observation modes, advanced engine states,
fitted families, shared-scope reconstruction, archive tampering and budget rollback.
The later `archive-compatibility.log` also covers explicit raw-chain tampering,
final-state mismatch and larger-limit paired generation. `static-final.log` records
format/lint/type/boundary/module/size checks.

Root-owned full-suite checking and independent review of the complete candidate
remain required. Browser, Colab, GPU agreement, measured resource behavior, native
categorical delivery and broad paper-option completion are not claimed here.
