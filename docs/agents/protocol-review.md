# Reviewing protocol completion

Use this review when an evaluation specification, manifest or evidence report claims to be
accepted, complete or frozen. The review concerns agreement and evidence, not whether every
field happens to be non-null. A linter cannot decide whether a choice was actually accepted
or whether a command substantiates the accompanying prose.

1. Read the accepted scope and inspect unresolved fields in its artifacts. Classify each:
   **decision** (someone must choose), **verification** (a fact must be established), or
   **authorization** (an authorized person must approve). Split mixed items into their parts.
2. Record who resolves it and which stage it blocks: documentation delivery, implementation,
   protocol freeze or execution. An accepted deferral is valid; keep it visible in the handoff.
3. Compare the completion statement with those entries. Use the narrowest supported status,
   for example: “Design direction accepted; sampling details remain for #26 before freeze.”
4. For each verification claim, locate its inputs, revision, environment, exact command/program
   and output. Run the retained procedure where permitted. State what was independently
   recalculated versus replayed and what remains unverified. A one-off session command becomes
   durable evidence only when its procedure is retained with the reviewed artifacts.

Example from the #25 candidate packet (not new scientific choices):

| Pending item | Kind | Resolver | Blocks |
| --- | --- | --- | --- |
| Class allocation/tie-breaking, predictor settings, tolerances | Decision | #26 proposes; owner accepts | Final protocol freeze |
| Exact data-file hashes, eligible support, collection/overlap findings | Verification | Owner after provisioning | Dataset admission and freeze |
| Package/weight selection and exact content hashes | Decision, then verification | Adapter author and owner | Reference pinning and freeze |
| CPU/thread choices and cap-enforcement method | Decision, then verification | Harness author and owner | Resource contract and execution readiness |
| Human protocol acceptance | Authorization | Owner | Protocol freeze |
| Checkpoint-selection record and compute/access approval | Verification and authorization | Run owner | Milestone execution |

The accepted #25 scope leaves these prerequisites pending. Calling the broad design accepted
is supported; calling every material choice resolved is stronger than the packet supports.
This review does not choose values, expand permissions, change protected files or add an
extra approval gate. Follow the existing owner and ticket boundaries.

For an example retaining a previously session-only arithmetic procedure, see
[the #25 reproduction record](https://github.com/Avistian/r0/blob/0e4801bde17896383862d0d695ae81334ebb7aee/docs/evidence/issue-25-retro/README.md). Its fixed-input checker
supports the authored calculations only; it is not an evaluation harness or asset validator.
