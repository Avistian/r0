# Issue #28: initial causal core review

Candidate `b0f1543de642b949de99e7af98ac75b77a7df4ab`, compared with
`af7f5b8965adfd236150ef21361c0797e1dbdb78`. Two fresh reviewers inspected standards
and specification independently. This is a checkpoint review, not overall #28
acceptance. The original **REJECT** remains part of the record.

## Standards

- **P2, reproduction evidence:** the ledger's first five historical RED/GREEN
  claims did not retain recoverable intermediate source snapshots or exact
  commands/environment alongside every log. `CODING_STANDARDS.md` requires a
  retained reproduction path. The logs are local, ignored execution artifacts.
- **Heuristic, duplicated code:** generation and inspection repeat the RNG
  identity construction. The reviewer identified no other hard source-standard
  violation; tests were NOT_RUN on this axis.

Two findings: one documented evidence limitation and one nonblocking duplication
heuristic. Worst standards finding: incomplete historical reproduction records.

## Specification

- **P1, direct outcome exposure:** identical observed and target channels were
  accepted. In an independent one-root regression probe, all eight query features
  equalled their held-out labels and diagnostics were empty. This violates the
  specification's held-out-label contract.
- **Partial scope:** sampled topology/mechanism/network breadth, typed variables,
  interventions, workbench, hosted execution and final evidence remain incomplete.
  Separate causal resource limits were absent: a 65,537-channel latent node was
  accepted by the constructor. Large tensor execution was NOT_RUN.
- No scope creep was found. Independent probes passed vector arithmetic, constant
  retention, JSON recipe replay and binary/multiclass trace reconstruction.

One confirmed correctness defect plus grouped incomplete requirements. Worst
implemented-spec finding: direct outcome exposure. Full #28 remains INCOMPLETE.

## Reproduction and follow-up

The historical first five failing runs were genuinely executed and their logs
remain in local `docs/evidence/issue-28/`. Intermediate source snapshots for those
failures are **NOT_RETAINED**; the logs alone do not prove exact-revision RED states.
They are not represented as independently recoverable committed RED checkpoints.

The five passing core behaviors are reproducible at `b0f1543` with:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_generation.py
```

This command ran in `/tmp/r0-28`, selecting its `src` through the repository
wrapper. The observed environment uses Python 3.12 and PyTorch `2.15.0+cpu`, with
CUDA unavailable. For another machine, run the repository's documented environment
setup or explicitly select a compatible existing environment. The home checkout
and its editable installation must not be changed to reproduce this work.

The direct-outcome regression has a separately committed RED state:
`9406302`, command below, exit 1 with `DID NOT RAISE ValueError`:

```sh
python3 scripts/run_check.py pytest tests/priors/test_causal_generation.py -k direct_outcome
```

Run the same command at the subsequent fix revision for GREEN. The test also
requires legitimate effect-to-cause prediction to remain available. The correction
reserves declared target channels from observed readouts; it does not reject
distinct descendant variables or infer leakage solely from high predictability.

Further independently reviewed source/runtime limits and final #28 acceptance
are still required. Later passing evidence does not erase this checkpoint's defects.

### Repair checkpoints

- Direct outcome exposure: RED `9406302`, GREEN `fb5d4d1`; the exact test command
  is above. Generation and inspection now also share the causal RNG identity
  helper, addressing the duplication heuristic without changing sampled streams.
- Resource admission: RED `1a8d889`, command
  `python3 scripts/run_check.py pytest tests/priors/test_causal_generation.py -k resource_limits`;
  it failed with `DID NOT RAISE ValueError` on the 65,537-channel constructor probe.
  The subsequent fix adds explicit configurable node/channel/parameter limits and
  a conservative working-array estimate checked before tensor generation.

Default admission limits are 1,024 nodes, 1,024 channels per node, 8,192 total
channels, 1,000,000 numeric parameters and 256 MiB estimated working storage. These
are initial configurable admission settings, not qualified throughput maxima or
permanent architecture ceilings. The estimate is not measured process RSS or an
OS memory limit. Actual device/shape resource qualification remains outstanding.


### Scoped independent recheck

At `b507f0c3686838e851a31edfa0975bc7bad0d836`, the independent spec reviewer returned
**PASS for the repair delta**. It independently checked direct-role rejection,
valid descendant and same-node/different-channel profiles, every resource-limit
boundary, invalid limits, calibration workload admission, arithmetic, historical
RNG identity, selected-row traces, resource records and JSON replay. Its retained
script exited 0 without large allocations, source edits or author-test reruns.
This is not acceptance of the remaining #28 scope.

The standards recheck confirmed the historical RED qualification and shared RNG
repair. Its remaining historical-Pyright evidence concern was addressed by marking
those exploratory source states/reproduction context NOT_RETAINED in the ledger;
they are not used as exact-revision acceptance evidence. The original reports and
scripts remain local evidence. Current public tests and commit hooks provide the
reproduction path for the identified implementation revisions.
