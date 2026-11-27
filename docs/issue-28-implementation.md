# Issue #28 implementation and evidence ledger

The [accepted specification](specs/0028-causal-network-generators.md) is the
authoritative scope. This ledger tracks execution, not a smaller replacement goal.

Latest: [frozen source functions](issue-28-frozen-functions.md) at `cbbaff2`
pass 868 tests (one skipped). All eight numerical function records evaluate and
round-trip; sampling/fitting/graph integration remain required.
Earlier [joint matrices](issue-28-joint-matrices.md) connect all five matrix
families to shared decisions. Full #28 acceptance remains pending.

## Checkout and scope preservation

- Working branch: `feat/issue-28-generators`, worktree `/tmp/r0-28`.
- Fetched base: `13f49167460058f847d8a651af489e578783a6c0`.
- The older home checkout and its unrelated edits remain intact.
- The accepted design was copied before edits; its fingerprint is retained in
  local `docs/evidence/issue-28/preflight.txt`. Only issue #28 glossary additions
  were appended to current main's vocabulary.
- `make preflight tracked-files` passed using the existing home-repo virtualenv
  with `scripts/run_check.py`; it resolved `r0` to this worktree's `src`.
- The oversized interview document was separated into the authoritative scope and
  its supporting research appendix to comply with the repository's file-size rule.

## Baseline verification

The public generation/composition, saved task-group and explorer-session baseline
passed **19 tests** before implementation. Command:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_compositions.py \
  tests/generation/test_task_groups.py tests/explorer/test_session.py
```

This verifies the selected inherited behavior and environment, not the new #28
requirements or full repository acceptance.

## Public verification seams

These are the public interfaces covered by the user's accepted generator/replay,
intervention, inspection and delivery requirements (decisions 2, 8, 10, 14, 18):

| Seam | What it proves | What it does not prove |
| --- | --- | --- |
| `priors.generate_task` and recipe decoding | Recipe behavior, typed episodes, randomness separation and reproducibility | Browser behavior or predictive superiority |
| Public recorded-mechanism inspection and intervention operations | Worked structural arithmetic, node/channel roles and intervention invariants | Inferring causal models from observed data |
| Generation run/group save/load | Identity, failures, bounded output, verified restart and durable replay | Hosted execution without a target-environment run |
| Explorer session and actual browser/Colab transports | User controls, comparisons, diagrams, exports and reopen behavior | UI proof from API tests alone |

Use one failing public-behavior test followed by its implementation per slice.
Expected results come from authored calculations or independent invariants, not
the implementation under test. Full completion still requires all specification
requirements and option records to have appropriate evidence.

## Execution stages

| Stage | State | Completion evidence |
| --- | --- | --- |
| Primary-source option extraction and finite inventory | READY | Source/option record for every documented relevant choice, including variants and scope boundaries |
| Replayable explicit causal and neural mechanisms | IN_PROGRESS | Independent worked calculations through shared public interfaces |
| Topology, function, activation, weights, roots/noise and hyperprior sampling | IN_PROGRESS | Every inventory option exercised with traceable recipe and evidence |
| Mixed typed variables and native/adapted targets | NOT_STARTED | Schema, probabilities, held-out-label separation and saved roundtrips |
| Matched/population interventions and graph/mechanism edits | NOT_STARTED | Independent invariants, versioned identities and saved comparisons |
| Diagnostics and coverage/filtering evidence | IN_PROGRESS | Original/retained denominators, failure reasons and structural/functional diversity |
| Shared explorer/Colab workbench and exports | NOT_STARTED | Actual interactions and rendered artifacts in each environment |
| Generation/storage/parallel/restart integration | NOT_STARTED | Retained execution, failure/restart and resource measurements |
| Independent review, checks and publication | IN_PROGRESS | Fixed-revision review, required repository checks, PR and approved-specification link |

## Important semantic constraints

- An arbitrary observed subset of a causal model does not define a deterministic
  function from its observed columns to its target. Mechanism replay/intervention
  must not masquerade as conditional prediction or feed observed descendants back
  into an invented forward function. The explorer must label the requested operation.
- Constant, saturated and disconnected but numerically valid draws remain valid.
  Calibration must represent zero dispersion explicitly rather than silently
  filtering these draws or dividing by zero.
- Mechanisms and fixed calibration are sampled independently of query rows and
  labels. Inspection must not change the task. New edits have new identities.
- Original generation failures remain visible; no silent retries, substitutions,
  clipping, implicit CPU fallback or claims that a narrow representative covers a
  complete published family.
- Source-specific neural samplers and trees fitted to synthetic random data are
  distinct constructions. Standalone tree breadth remains #29, but all documented
  structural component variants remain required #28 scope.

## Outstanding authorization and evidence

This implementation goal does not itself authorize a paid/full-scale launch,
merge, deployment, or real-data download. Local code and synthetic verification
can proceed. Hosted measurements remain pending until executed under the existing
authorized path. Draft [PR #125](https://github.com/Avistian/r0/pull/125) publishes
the specification. Issue #28 links the verified scope at `66e91e2`; fresh readback
and the saved before/after bodies confirm the original issue body is unchanged.
Reproduce using `python3 scripts/verify_issue28_spec_link.py` (read-only `gh api`
access). Its [receipt](research/0028-spec-link-receipt.json) retains the exact body
snapshots and observed published-file hash. The procedure compares published bytes
with the pinned Git object, verifies the 19 numbered decisions and final paper-option
condition, checks the sole appended section, then reads back the current issue.

## Source gate result

[The finite inventory](research/0028-source-option-inventory.md) contains 583
source/extension/disposition records and 140 specification anchors, including two
converter graph-alias variants discovered after the initial 581-record gate. The eight JSON
chunks have unique IDs and explicit ownership/unknown states. These are traceability
records, not unique algorithms or implementation counts. Exact peer/checkpoint
reproduction is not claimed. Documented choices are ready for implementation;
unknown unpublished algorithms remain explicit.

## Local hardware

The inspected environment reports PyTorch `2.15.0+cpu`, 12 CPU threads, no CUDA
runtime and no available CUDA device. GPU execution evidence is NOT_RUN here.
Historical sandboxed Pyright logs contain dependency-resolution failures. The
exact source states and complete reproduction context for those exploratory runs
are NOT_RETAINED, so they are not exact-revision verification evidence. Subsequent
commit hooks ran typechecking at the identified commits. Local exploratory logs
remain under `docs/evidence/issue-28/`; a current typecheck is reproducible with the
repository wrapper and the explicitly selected interpreter/environment.

## Earlier checkpoints

The original causal-core, topology, nonlinear, activation-preference, exogenous,
factorized, importance-weighted, matrix-family, named numerical-preference and
shared categorical/matrix-preference and source activation-selection/calibration
entries are preserved verbatim in the
[checkpoint history](issue-28-implementation-history.md), with their original
findings and evidence links.

## Source activation causal-profile checkpoint

At `c4081e1`, eight source-activation profiles bind primitive and named rescale
preferences to Gaussian-affine causal structures across all three graph families.
Standard wrappers fit independent ancestral samples in parent order and retain
all constants and stream identities. JSON loading reconstructs that calibration
and rejects incoherent stage edits; task rows/context cannot refit the mechanism.
Plain profiles retain unwrapped primitives. Combined sampling/fitting costs and
calibration buffers are included in admission and task cards.
[Definitions and evidence](issue-28-source-causal-profiles.md).

Full checks pass **679 tests, one skipped**, all static gates. Independent NumPy
verification passes 27 ancestral/generation cases, 24 corruption/admission rejects,
six plain policies, six saved workflows across task types and 37 exact old-profile
files from twelve fixtures. Independent accounting checks exact budget boundaries;
array estimates remain distinct from measured process RSS. Standards finds no hard
violations and one nonblocking metadata-accounting duplication heuristic.

The unfiltered author census retains **1,507 valid / 1,536 attempted** tasks and
all 29 complete failed recipes. All failures are generation overflows; all 31
primitive kinds occur across attempted recipes. Valid degenerate draws remain.
Successful raw census archives are LOCAL_ONLY with hashes; all headers and failed
records are published. Matrix integration, remaining wrappers, wider graph/network/
tree/GP and structural options, typed states, interventions and actual browser/Colab
workbench delivery remain required. The full goal stays active.

## Joint source activation and matrix checkpoint

At `bb78d81`, eight joint profiles integrate source standard/plain node activations
with all five matrix families across three graph families. One activation scope
owns transformed-matrix and node choices. Complete matrix/stage bindings and
independent ancestral calibration survive JSON loading, generation and saved
backend replay. Matrix calibration follows the actual affine-then-noise equation.
[Laws and evidence](issue-28-source-matrix-profiles.md).

Full checks pass **738 tests, one skipped**. Independent scoped PASS covers 45
numerical cases, 33 corruption/admission rejects, six plain policy variants, six
saved task-type workflows, 24 profile/graph combinations and 16 joint global seeds.
Twenty old profiles retain 61 byte-identical files. Original numerical-oracle and
fixture stops, diagnoses and corrected full runs remain published. Standards finds
no documented violations and one advisory duplicated-planning concern.

The unfiltered census retains **1,509 valid / 1,536 attempted**, with 27 generation
overflows and complete sampled recipes for every attempt. All headers and failed
recipes are published; successful raw archives remain LOCAL_ONLY with hashes.
Pooled recipes exercise 31 primitives and five matrix families. Remaining wrappers,
broader exogenous/graph/network/tree/GP options, typed states, interventions and
actual browser/Colab workbench delivery remain required. The full goal stays active.

## Frozen L2 and Kumaraswamy authored-transform checkpoint

At `be0c793`, immutable transforms retain node-wide L2 factors or per-channel
Kumaraswamy ranges with scalar shapes, calibration identity and digest. Authored
activated/stochastic/matrix nodes use them at input/output stages through JSON,
generation, diagnostics, original-row traces and saved explorer workflows.
Stable arithmetic preserves representable extreme results; domain-clamp flags
remain distinct from out-of-calibration flags.
[Laws, boundaries and evidence](issue-28-calibrated-transforms.md).

Full checks pass **752 tests, one skipped**, all static gates. Initial `bf3a4c5`
remains **REJECT**: rounding at an epsilon boundary returned one instead of the
independent `1.419802591052941e-305`. The repair retains the subtraction residual.
Its unchanged 874-case oracle and 589 additional boundary/width cases pass with
1,000-digit Decimal references. Independent verification also passes 172
integration checks, 36 authored compositions, two dimension-changing compositions
and six local saved workflows. All 168 legacy proposals match byte-for-byte:
164 successes and four unchanged explicit failures. Six executed verifier programs,
96 imported-source hashes and original/repaired raw outcomes retain custody.

Standards originally found two evidence-retention omissions. The narrow delta
resolves both with raw development logs and a fresh retained alias-premise fixture;
unavailable historical snapshots/receipt remain explicitly NOT_RETAINED. The
source investigation preserves three distinct converter semantics: historical
tuple intent, actual pinned graph alias behavior and corrected readout-only.
Their graph integration remains required.

Authored caller samples do not prove independence or population moments. Sampled
pipelines must certify moments at the actual L2 input before independent fitting.
Shared shape sampling, node pipelines/converters, broader source mechanisms,
typed states, interventions and actual browser/Colab delivery remain required.
No supported-size, GPU or scientific acceptance is claimed. The full goal stays active.

## Authored node-pipeline and converter checkpoint

At `ae9b3b8`, ordered frozen stages, channel importance and L2 precede explicit
numerical converter state/readout branches; structural innovation and whole-node
importance follow. Three named variants preserve historical tuple intent, actual
historical graph behavior and corrected readout-only behavior. Descendants consume
state; converted columns retain their earlier readouts. Moments, normalization,
ancestry, original-row stage traces and saved replay honor that distinction.
[Laws and evidence](issue-28-node-pipelines.md).

Full checks pass **770 tests, one skipped**, all static gates. Independent scoped
ACCEPT covers 50 numerical comparisons (maximum absolute error 8.88e-16), 20
expected contract rejections, nine saved variant/task-type workflows, moment and
ancestry cases, alias independence and frozen-control invariance. Independent
resource arithmetic matches 87 parameters / 32,768 estimated bytes; all-node trace
buffers are summed. Four exact executed programs and raw outcomes retain custody.
All 168 legacy proposals are byte-identical: 164 successes and four retained
failures. Standards finds no hard violations and two advisory duplication risks.

The source inventory adds explicit graph-alias variants for both Kumaraswamy and
activation converters: 583 source records plus 140 anchors, not implementation
counts. The follow-up preserves source redraw behavior and separate root-point and
later-noise paths. Authored roots currently have deterministic pre-pipeline bias;
source root-law and independent sampled-pipeline integration remain required.
Shared shape/weight preferences, activation converters, source standardization,
source profiles, all wider mechanisms and actual browser/Colab/GPU/scale evidence
remain in scope. The full #28 goal stays active.


## Independent root input and activation-converter checkpoint

Source `c997bc2` adds a separately keyed root input before activation/pipeline,
with independent later structural noise and signal-dependent response. Gaussian,
uniform, Student-t, mixture, truncated, ball and factored laws compose through the
existing frozen record. One-channel fitted standard activations now support all
three converter variants; readout/state moments and ancestry follow each branch.
[Contract, source adaptations and evidence](issue-28-root-input-converters.md).

Full checks pass **797 tests, one skipped**, all static gates. Independent scoped
review passes 306 checks, including 209 numerical comparisons (maximum absolute
error 8.88e-16), 17 expected rejections and nine saved variant/task-type workflows.
Exact resource admission matches 21 parameters / 162,432 estimated bytes; one-below
limits reject. All 168 legacy proposals remain byte-identical, including four
retained numerical failures. Eight exact executed programs and 260 imported-source
receipts bind the evidence. Standards has zero hard findings and one nonblocking
decoder-duplication advisory. Initial verifier setup errors remain in the archive.

This advances authored `I2.converter.03` / `.03.graph` and the separate root-input
stage. Source RandomPoints functions, covariance/mixture parameter sampling,
standalone Standardize, independent ancestral fitting, shared choices and full
source profiles/census remain required. No browser/Colab, GPU, scale or scientific
acceptance is claimed. The full #28 goal remains active.


## Standalone Standardize checkpoint

Source `962f0ad` adds immutable population Standardize with epsilon 1e-4 outside
the square root. It supports node input/output placement and ordered pipeline
stages, retains calibration identity/digest, preserves one-row and constant-channel
behavior, and checks finite input second moments before construction. Generation,
original-row inspection and saved replay use the same frozen constants.
[Contract and evidence](issue-28-standardization.md).

Full checks pass **809 tests, one skipped**, all static gates. Independent scoped
review passes 174 checks: 38 numerical comparisons, 58 expected rejections, exact
fitter/equation resource limits, moment/ancestry cases and three saved task types.
All 168 legacy proposals are byte-identical, including four retained failures.
Five exact programs and imported-source custody preserve reproduction. Standards
finds zero hard violations and one nonblocking duplicated-admission-list advisory.
Earlier verifier mistakes remain classified and retained separately.

This advances node standardization; sampled source pipeline construction remains
required. The [integration design](research/0028-pipeline-sampling-integration.md)
recommends one shared preference context, complete transactional binding ownership,
explicit global-mode adaptation, source-name adapters and ancestral fitting against
final parent states. It preserves legacy families. Proposed design oracles are
NOT_RUN. Full #28 scope, browser/Colab, GPU, scale and scientific evidence remain open.

## Shared sampling context checkpoint

Source `662fb89` adds one transactional numerical/categorical scope across named
operations, nested ownership, explicit global policies, forced identities and
aggregate admission before draws/decoding. Existing samplers retain their default
behavior. [Contract and retained evidence](issue-28-joint-sampling.md).

Full checks pass **825 tests, one skipped**, all static gates. Independent review
passes 286 assertions including 165 expected rejections; all 156 legacy pairs are
byte-identical, including three retained failures. Source/import custody verifies
323 entries each. The original `2812d23` decoder REJECT remains retained: a string
path was silently coerced into character components. Repaired decoding rejects
string/dictionary paths. The separate Tensor-output verifier mistake is retained.
Standards accepts the narrow repair/documentation remedy, with no source-level
hard finding and one advisory transaction-copy duplication concern.

This is the shared context/ledger foundation, not the sampled graph integration.
Typed activation/matrix/root/converter plans, semantic destination bindings,
independent ancestral fitting against finalized parent states, source profiles,
all wider options and actual browser/Colab/GPU/scale evidence remain required.
The full #28 goal stays active.
