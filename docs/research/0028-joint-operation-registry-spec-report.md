# SPEC component verdict: ACCEPT

Final reviewed source de1bead86d51721fd84d84598b7bea2800ec8885; base51684d98b719494c2a20d1a07e7a0470c6c3b4c3.
Diff reviewed: git diff 51684d98b719494c2a20d1a07e7a0470c6c3b4c3...HEAD.
Commits906f99b (registry),1110228 (streams),de1bead (decoder shape repair).
Preparatory96d8711 has the same source tree; its evidence remains schema-repair/.
Final source preflight and exit status contain only intentional untracked .venv.

No unresolved missing/partial requirement, scope creep or incorrect behavior found
within the assigned registry and jointly staged stream component after repair.
Historical1110228 REJECT remains immutable in sibling bundle
/tmp/r0-28-operation-registry-spec-1110228. This repaired verdict does not erase it.

Independent exact-final-source execution:
- 708 wrong container/missing/unknown field mutations through100containers, including
  nested categorical hyperparameters and polynomial weights: all ValueError.
- 460 scalar-to-container corruptions: all ValueError.
- 25 generated operation trees checked against independently retained preorder and
  event-path ownership;10 explicit corrupted registries rejected.
- 42 transaction scenarios: five NumPy engines across three modes;20 caller-mutation
  cases; six aliases; real covariance/tree fit followed by abort. Raw values and
  states compared to direct NumPy draw oracles. Nested abort/retry, caches, literal
  spans, stale child/outer handles, parent-handle mutation, limits and UTF8/depth checked.
-15 isolated baseline/final cases: exact serialized public root/weight plans,
  JointSamplingRecord v1 and next16named draws identical. Baseline0c3a032 source/tests
  byte equality to51684d9 independently checked. Intentional empty-uniform-path reuse
  difference reproduced: baseline accepts; final rejects.

Research contract lines159-163 requires exact shapes and unique complete operation
ownership; lines175-180 requires jointly staged state while separating graph atomicity;
lines184-188 requires bounded decoder inputs and caller-controlled caps.
Evidence supports these component obligations. Raw-field manifest, typed plan-tree
ownership, ancestry calibration, aggregate graph resource admission, graph-return
builder, whole issue28, seed independence and thread locking remain outside this
verdict. Full repository makecheck belongs to the parent and is separate evidence.

No product execution failures on repaired sources. Preexecution Ruff B023 failures
are retained as lint-002.log and lint-004.log with exact corresponding program bytes;
subsequent executed programs are formatted/Ruff-clean. Gates use explicit exceptions.
Probe execution records pin source HEAD/tree, imported checkout and program hashes.
Every retained artifact is bounded to20000bytes.
