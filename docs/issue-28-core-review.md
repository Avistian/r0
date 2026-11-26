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
