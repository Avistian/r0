# Issue 28: operation ownership and joint stream transactions

Source checkpoint: `de1bead86d51721fd84d84598b7bea2800ec8885`.
Base: `51684d98b719494c2a20d1a07e7a0470c6c3b4c3`.
This is one graph-construction component. Whole issue #28 remains incomplete.

## Operation ownership

`JointSamplingContext.freeze_registry()` records every successfully completed
operation in preorder, including uniform-weight plans and deterministic scopes
that create no named decisions. Each `JointOperationSpan` records its owner path
and exact half-open decision-event interval. Parent containment is valid; duplicate
paths, orphan events, missing parents, reopened subtrees and overlapping siblings
are rejected. Empty scopes retain empty spans. Only an idle outer context can
freeze the completed registry; active or closed child handles cannot do so.

Previously, event-free scopes were omitted from completion bookkeeping and their
paths could be reused. Successful operations now reserve their path regardless
of draw count. Failed children leave no phantom owner, and a failed outer scope
rolls back its completed descendants. `JointRegistryLimits` bounds operation
count and total UTF-8 path bytes, independently of preference-draw limits.

`JointOperationRegistry` contains the full decision ledger. Existing
`JointSamplingRecord` v1 serialization is unchanged. The new registry decoder
admits raw metadata with caller-owned `CausalLimits`, applies caller-owned
registry caps, checks exact record/array shapes throughout the nested ledger,
then delegates numerical-law/reference validation to the existing typed records.
Payload caps cannot enlarge caller authority. Standalone legacy ledger decoder
behavior is unchanged by this new boundary.

## Named choices and auxiliary streams

The context accepts an optional dictionary of explicitly injected auxiliary
NumPy generators. `stream(name)` is available only on a live idle operation and
returns its staged generator. Every registered bit generator must be distinct,
including the generator used for named choices. The input dictionary is copied;
mutating that dictionary later cannot redirect the context's streams.

Each transaction stages named choices, all auxiliary streams, preference caches,
decisions and operation ownership together. Child success advances its parent's
staged state. Only outer success advances the caller's original generators.
All bookkeeping copies are prepared and all original states checked before
commit. A body exception, downstream admission failure or detected external
mutation rejects the transaction. External draws themselves are not undone.
Closed child handles cannot expose streams; previously returned child generators
do not alias committed parent state. This is not a thread lock or a claim of
cross-thread concurrency safety.

`JointStreamLimits` defaults to 64 auxiliary streams and one MiB of serialized
state, including the named-choice generator. Both limits are caller-configurable.
These bound retained/copyable RNG state, not OS process memory. Distinct generator
objects do not prove statistically independent seeds.

Example construction pattern, using the existing root helper:

```python
context = JointSamplingContext(
    named_rng,
    streams={"construction": raw_rng, "calibration": calibration_rng, "fit": fit_rng},
)
with context.transaction("graph") as graph:
    plan = sample_joint_root(graph, "root", width=2, calibration_rows=32)
    root = materialize_joint_root(
        plan,
        graph.freeze(),
        rng=graph.stream("construction"),
        calibration_rng=graph.stream("calibration"),
        fit_rng=graph.stream("fit"),
        calibration_id="root-calibration",
    )
registry = context.freeze_registry()
root.validate(registry.decisions)
```

This context owns its RNGs and sampling records. It does not undo arbitrary caller
I/O or collections mutated inside the body, and it is not yet a graph-returning
builder. Raw-field/stream manifests, ownership of typed plan trees, ancestral
calibration and aggregate graph budgets remain required integration work.

## Verification and retained rejection

The first independent Spec review **REJECTED** source `1110228`: malformed
`decisions.events={}` and an empty string decoded as the valid empty ledger;
`decisions.events=[0]` leaked `AttributeError`. Two pinned independent runs retain
these outcomes. That source's full suite nevertheless passed **1172 tests,
one skipped**, 496.47s. Its green suite does not erase the structural rejection.

The repair adds a strict structural pass before legacy reconstruction. Author
checks cover all array/record paths in a ledger containing local/global/meta
numerical and categorical choices, including categorical weight hyperparameters.
The original three malformed cases now raise `ValueError`.

Author focused checks: **98 passed**, 2.23s. Static checks passed. Transaction
coverage includes real roots fitted before outer failure, exact fresh recovery
across all five NumPy generators and three preference modes, six alias pairs,
20 external-mutation cases, nested caught failures, stale handles, resource caps
and a real later-root admission failure after an earlier root was fitted.

Final full `make check` at source `de1bead`: **1180 passed, 1 skipped**, 484.52s;
all static gates passed. [Full check](research/0028-joint-operation-registry-check-de1bead.txt).

Independent repaired Spec review: **ACCEPT component**. Final-source probes passed
708 malformed container/field cases, 460 malformed scalar cases, 25 independent
preorder/span models, ten ownership corruptions, and 42 transaction scenarios.
Across 15 engine/mode combinations, isolated baseline/final processes produced
identical public plan bytes, decision-ledger bytes and subsequent named RNG draws.
The intentional event-free path-reuse rejection was separately confirmed.
Preparatory `96d8711` evidence remains separate from final `de1bead` execution.
[Final Spec report](research/0028-joint-operation-registry-spec-report.md).
Standards source review: zero hard violations; original bookkeeping-copy advisory
plus a repair-delta strict-record-validation duplication advisory. Both are
nonblocking heuristics, not demonstrated correctness defects.

No raw-manifest, graph-wide budget, graph fitter, browser, Colab, GPU, cloud,
training or predictive-quality acceptance is claimed here. `make smoke` and
`make eval-smoke` remain unavailable while train/eval CLI entry points are absent.


## Evidence and reproduction

- [Historical Spec rejection](research/0028-joint-operation-registry-spec-reject.md)
  and [old green suite](research/0028-joint-operation-registry-check-1110228.txt)
  are retained alongside the repair; neither supersedes the other.
- [Original Standards report](research/0028-joint-operation-registry-standards-original.txt)
  and [repair-delta Standards report](research/0028-joint-operation-registry-standards-repair.txt)
  distinguish the original scope from the later shape-validation change.
- [Independent archive metadata](research/0028-joint-operation-registry-spec-artifacts.json)
  retains both review bundles, their input/program hashes, historical failures,
  final probes and a fresh replay of the exact public verifier programs.
  [Recovery receipt](research/0028-joint-operation-registry-spec-recovery.json).
- [Development metadata](research/0028-joint-operation-registry-development.json)
  retains RED/GREEN, intermediate repairs, static checks and both full checks.
  Intermediate uncommitted source snapshots are not all retained.

Recover the archive into a new directory:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-joint-operation-registry-spec-artifacts.json /tmp/registry-evidence
```

The [retained commands](research/0028-joint-operation-registry-reproduce.txt) identify
all executions. At the pinned source checkout, run recovered `repaired/` probes
with `R0_REVIEW_ROOT` pointing to the checkout and `R0_REVIEW_OUTPUT` pointing to
a fresh output directory for each probe. The public copies under
`scripts/verify_joint_operation_registry/` are byte-identical to the final executed
programs. They need recovered `pins.json`; the scalar probe also needs a generated
`schema-final/valid.json`, and the comparator needs the baseline and final
compatibility directories. Source pins reject other heads and dirty source/tests.
The publication replay ran those public copies from outside both checkouts,
regenerated baseline and candidate outputs, then compared the resulting bytes.
This repeats retained cases rather than adding new coverage.

No source/test bytes changed in documentation publication. The graph integration
research and later implementation remain separate work.
