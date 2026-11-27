# Issue #28 implementation and evidence ledger

The [accepted specification](specs/0028-causal-network-generators.md) is the
authoritative scope. This ledger tracks execution, not a smaller replacement goal.

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

[The finite inventory](research/0028-source-option-inventory.md) contains 581
source/extension/disposition records and 140 specification anchors. The eight JSON
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

The original causal-core, topology, nonlinear, activation-preference, exogenous
and factorized entries are preserved verbatim in the
[checkpoint history](issue-28-implementation-history.md), with their original
findings and evidence links.

## Importance-weighted projection checkpoint

At `e6d8faa`, authored uniform and polynomial/log-Gaussian importance vectors feed
recorded row-normalized Gaussian projections. `causal-importance-v1` exposes this
through all three topology families and shared save/reopen/replay. Stable centered
arithmetic preserves decay under huge common perturbations and valid epsilon-zero
directions after weight underflow. Family provenance rejects factorized/importance
substitutions. [Definitions, source distinctions and evidence](issue-28-importance-weights.md).

Repository checks pass **613 tests, 1 skipped**; source/test bytes match `fc0265b`
where that check began. Independent scoped review passes 2,134 numerical checks,
384 sampled recipes, four saved backend sessions and 315 exact old-profile outcomes
(including two failures). Resource instrumentation precedes all 22 matched Gaussian
allocations in its declared example. The author census retains **191 valid / 192
attempted**, including the failed integer-power recipe. A disconnected-graph oracle
mistake is retained as a harness error, not a product rejection. Standards found no
hard violations; repeated profile registries remain a nonblocking heuristic.

This supplies positive-vector/weighted-Gaussian options within W01/W03, not all
weight priors. Shared source numerical preferences, kernel/transformed matrices,
wrappers, masks, networks/trees, typed states, interventions and the full workbench
remain required. CPU backend evidence does not establish browser/Colab/GPU or
scientific acceptance. The full goal stays active.


## Matrix families and kernel point checkpoint

At `7eea0bd`, the constructor pipeline retains Gaussian, importance, positive-vector
factorized, signed Laplace and flattened-activation matrices; explicit jitter and
normalization stages; and covariance/Gaussian-mixture kernel point constructions.
`causal-matrices-v1` samples all five families, while `causal-gaussian-matrices-v1`
restricts the source family. Both share the existing generation, inspection and
saved replay workflow. [Definitions and source distinctions](issue-28-matrix-families.md)
retain the paper/code point variants and explicitly local sampling choices.

Full checks pass **623 tests, 1 skipped**, all static gates passing. Independent
review passes 224 numerical checks, 72 sampled recipes (71 successes, one retained
square-overflow failure), six saved backend workflows, and 378 exact legacy outcomes
(including two failures). Standards found no hard violations; duplicated component-
probability normalization is a nonblocking heuristic. The corrected zero-coefficient
oracle assumption remains a separate harness error, not a product rejection.

Unfiltered author censuses retain **192/192 valid** mixed-profile tasks and
**190/192 valid** Gaussian-only tasks. Both Gaussian-only failed recipes and the
independent failed recipe are retained completely, with reconstruction hashes.
Timing and peak RSS are NOT_MEASURED. Browser/Colab/GPU evidence remains outstanding.

This advances I2 matrix options and their primitive point laws. Shared source
numerical/categorical preferences, complete source activation weighting, wider
graph/network/tree/GP mechanisms, typed states, interventions and the full workbench
remain required. No whole-issue or complete source joint-prior acceptance is claimed.
Earlier implementation entries were moved verbatim to the linked checkpoint history
to keep files within the repository size limit. The full goal remains active.

## Named numerical preference checkpoint

At `593c924`, a public retained sampler supports independent local draws, fixed
global values and shared named Beta hyperparameters. Records preserve unit draws,
linear/log mapping, endpoint mass, round/floor conversion, batch shape, effective
scope and explicit strict/quantile reuse policy. Loading validates realizations,
references and first-use provenance. Draw/value/retained-parameter admission occurs
before random sampling or allocation. [Laws and evidence](issue-28-parameter-preferences.md).

Full checks pass **628 tests, one skipped**, including all static gates. Independent
spec review passes 708 bounded numerical/contract checks with no scoped defects;
standards finds no hard violations and one nonblocking accounting-duplication
heuristic. Earlier mechanism/profile implementation files are byte-identical to
`12ad190`. This is a numerical API checkpoint: categorical shared/refreshed weights,
generator binding/admission and profiles remain required. All wider #28 stages,
including full source activation weighting, typed states, interventions and actual
browser/Colab delivery, remain in scope. Smoke/eval-smoke are NOT_AVAILABLE; no
scientific or full-issue acceptance is claimed. The full goal stays active.


## Shared categorical and matrix preference checkpoint

At `639dd2a`, categorical local/global/meta sampling retains named weights, fresh
or shared distributions, exact mass selection and complete decoding provenance.
All five matrix families bind controlled fields to retained numerical/categorical
draws with full parameter accounting. Four profiles expose shared, refreshed,
local and global matrix preferences through all three topology families and saved
replay. [Laws, admission, source distinctions and evidence](issue-28-shared-matrix-preferences.md).

Full checks pass **649 tests, one skipped**, including static gates. Independent
verification passes **167 bounded probes** and **504 exact legacy outcomes**,
including four failures on both old and new revisions. An initial SCOPED_REJECT
found loss of a valid positive subnormal category through an unnecessary probability
view requirement. The repaired construction and decoder preserve the exact mass;
independent integer and Decimal oracles pass. The rejection, RED/GREEN, final
receipt and original harness errors remain inspectable. Standards' missing
publication-path finding is resolved; request-factory duplication is nonblocking.

The unfiltered author census retains **764 valid / 768 attempted** across the four
profiles, with four complete overflow failures and all five matrix families.
All **768 recipe identities** reproduce after the repair; this qualifies the prior
census recipes, not fresh task generation. Backend evidence remains separate from
browser/Colab/GPU and scientific acceptance. Smoke/eval-smoke are NOT_AVAILABLE.
Complete source activation selection/calibration, wider graph/network/tree/GP and
root/noise/weight variants, typed states, interventions and the full workbench remain
required. The full goal stays active.


## Source activation selection and authored calibration checkpoint

At `d9193b8`, public retained selectors preserve the pinned source's four/eight
fixed-family slot multiplicities, 27 fixed primitives and distinct real/integer
power laws. Named local/global/meta and shared/refresh policies retain complete
selection references. A rejected hierarchical selection commits neither RNG
advancement nor partial records. Authored causal nodes now support an independently
fitted standardize/rescale/activate/standardize pipeline with frozen constants,
calibration identity, per-channel shifts and accounted storage. Shared generation,
JSON replay and row inspection use those same constants.
[Laws, source adaptations and evidence](issue-28-source-activations.md).

Full checks at the repaired source pass **653 tests, one skipped**, all static
gates passing. Initial `c7cf429` is retained as **REJECT** despite its passing
652-test suite: scaling before centering distorted a nearly constant large-offset
channel's population standard deviation. The repair preserves that spread and
keeps an overflow branch for opposite-sign extremes. Original fixture, independent
Decimal oracle, saved wrong outputs and RED/GREEN remain inspectable. Independent
verification passes ten Decimal fixtures, four coupled-function oracles, 480 exact
local selections, six policy combinations, malformed-record and rollback checks,
72 byte-exact legacy regression cases and new binary/multiclass calibrated cases.
Both standards stages find no hard violations; resource ownership remains a
nonblocking heuristic. All final programs and raw outcomes are retained.

This increment supplies selector APIs and authored frozen pipelines. Their binding
to sampled causal/matrix profiles, shared rescale sampling, remaining L2/Kumaraswamy
wrappers, wider graph/network/tree/GP and structural options, typed states,
interventions and actual browser/Colab delivery remain required. Smoke/eval-smoke
are NOT_AVAILABLE; scientific and supported-size acceptance are not established.
The full issue #28 goal stays active.


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
