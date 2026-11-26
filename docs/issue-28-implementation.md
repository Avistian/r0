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
| Topology, function, activation, weights, roots/noise and hyperprior sampling | NOT_STARTED | Every inventory option exercised with traceable recipe and evidence |
| Mixed typed variables and native/adapted targets | NOT_STARTED | Schema, probabilities, held-out-label separation and saved roundtrips |
| Matched/population interventions and graph/mechanism edits | NOT_STARTED | Independent invariants, versioned identities and saved comparisons |
| Diagnostics and coverage/filtering evidence | NOT_STARTED | Original/retained denominators, failure reasons and structural/functional diversity |
| Shared explorer/Colab workbench and exports | NOT_STARTED | Actual interactions and rendered artifacts in each environment |
| Generation/storage/parallel/restart integration | NOT_STARTED | Retained execution, failure/restart and resource measurements |
| Independent review, checks and publication | NOT_STARTED | Fixed-revision review, required repository checks, PR and approved-specification link |

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
authorized path. The approved specification link must be published and verified
with the implementation PR, preserving the rest of issue #28.

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
Sandboxed Pyright failed to resolve NumPy/Torch despite passing runtime imports;
an explicit-interpreter unsandboxed run resolved dependencies and exposed only the
new tests' `Tensor.tolist` typing, which was corrected. Retained logs live under
`docs/evidence/issue-28/`.

## First causal core slice

Implemented explicit affine scalar/vector DAGs through `recipe_from_dict` and
`generate_task`, with independently keyed structural innovations, original-row
trace reconstruction, partial-channel readouts and exact recipe replay. Fixed
independent calibration is optional; constant channels retain their raw value
with identity normalization and a diagnostic flag. Shared nonzero-edge channel
ancestry is labeled as potential dependence, not proof of predictive usefulness.

Five passing public-behavior cases and historical failing logs are retained under
`docs/evidence/issue-28/` (intermediate RED source snapshots are NOT_RETAINED):
`causal-authored-*`, `causal-trace-*`, `causal-vector-*`, `causal-calibration-*`,
and `causal-constant-*`. This is a foundation slice, not completed paper-family
coverage. Sampled topology/function priors, nonlinear mechanisms, typed categorical
states, interventions and the expanded workbench remain outstanding.

Independent review rejected the first core checkpoint for direct outcome exposure;
see [the review and reproduction record](issue-28-core-review.md). The first
regression fix has a committed RED state; the original review is retained.
