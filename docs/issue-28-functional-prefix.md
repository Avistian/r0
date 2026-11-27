# Issue 28: admitted ancestral prefixes

`FunctionalPrefix` describes frozen ancestor nodes, a fresh population size, limits,
and any surrounding retained construction metadata. It needs no task definition,
feature readouts or target readouts. This provides the evaluation seam identified
in [sampled graph ownership research](research/0028-sampled-graph-ownership.md);
the complete sampled builder and raw-draw recorder are still unfinished.

`evaluate_functional_prefix(prefix, rng=..., device="cpu")` returns ordered tuples
of node states and readouts. The caller injects a keyed generator provider and is
responsible for assigning a fresh population stream to each receiving fit. Calling
this function repeatedly with an identically reset provider repeats the population;
a new receipt identifier alone does not make a population independent.

Complete task generation and prefix evaluation share the same internal node loop.
It evaluates real root laws, ordered parent state concatenation, frozen functions,
noise scales, pipelines, structural innovations and converter readouts. Descendants
consume final states; captured converter values remain distinct from those states.
No artificial observed/target channels or dummy nodes are inserted for prefix fits.

Before drawing arrays, prefix construction validates acyclic parent order, widths,
normalization moment certificates and conservative aggregate resource estimates.
The frozen prefix fields bind nodes, population size and limits. Arbitrary caller-supplied retained dataclasses are not deeply frozen: their contents are recharged at evaluation, and valid same-size mutations remain allowed. Concurrent mutation is unverified.
Evaluation repeats admission before drawing. Limits include caller-supplied retained
metadata, but cannot discover external objects the caller omitted. Estimates are
not measured RSS or an OS cap. Existing complete-recipe admission uses the shared
arithmetic with its original task metadata and workspace charges.

The node record moved to a shared internal file; public imports and serialized
node/recipe shapes remain unchanged. Version-one task arithmetic and RNG order
must remain compatible. The checked candidate passed the full suite and bounded independent review. These results do not establish the complete sampled builder.

## Reproduce focused checks

From this worktree, using the existing Python 3.12 environment without installation:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest \
  tests/priors/test_functional_prefix.py \
  tests/priors/test_functional_causal.py \
  tests/priors/test_functional_calibration.py \
  tests/priors/test_functional_admission.py
```

The new literal tests exercise a dimension-changing tree/quadratic prefix with
reordered multi-parent input, converter capture versus noisy descendant state, and
rejection of malformed ancestry/rows/resource caps. The complete-graph tests cover
the existing task path. The current local run passed 29 tests; its raw output is
`/tmp/r0-28-graph-prefix-green.log`, now retained with exact-source independent verification in the evidence below. No browser, Colab, GPU, throughput or scientific evidence
is established by this component.


## Accepted checkpoint evidence

- [Full check](research/0028-functional-prefix-check-97cd023.txt): **1258 passed, one skipped**, 525.59 seconds, with all static gates. Checked candidate `97cd0233726fed230a570435ab34dfbaf8e63556` has the same source/test trees as publication source `0ceb2c042c91a99df91b57043fb82c8da91e4721`.
- [Candidate independent review](research/0028-functional-prefix-candidate-review.md) and [publication-source replay](research/0028-functional-prefix-publication-review.md): literal parent ordering, changed widths, converter captures versus noisy descendant states, reproducible population namespaces, malformed shapes/ancestry/rows, preallocation/resource rejection, moment gates and retained-metadata readmission. All 24 complete version-one/version-two snapshots match their baseline byte-for-byte, including recipe/card/trace/tensors and admission reports.
- [Standards review](research/0028-functional-prefix-standards.txt): no documented violations or actionable heuristic findings. Tests were not run by that reviewer.
- [Archive metadata](research/0028-functional-prefix-spec-artifacts.json) and [recovery receipt](research/0028-functional-prefix-spec-recovery.json): 1058 verified file identities. The original invalid-weight reviewer fixture and its correction remain retained; no product defect was found.
- [Public verifier](../scripts/verify_functional_prefix) is a linted copy with an explicit 24-snapshot guard and reachable equivalent baseline. Its separate author replay passed on publication source. Original independent bytes and executions remain sealed separately. [Publication gates](research/0028-functional-prefix-publication-gates.txt) cover document/script checks; source/tests are unchanged by publication.

Recover the archive and verify every checksum from the repository root:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/research/0028-functional-prefix-spec-artifacts.json \
  /tmp/functional-prefix-recovered --repo .
```

With the existing supported Python environment and a clean checkout (untracked
`.venv` is permitted), run the public verifier into a new directory:

```sh
python scripts/verify_functional_prefix/replay.py \
  --source-root "$PWD" \
  --output-dir /tmp/functional-prefix-fresh \
  --expected-head "$(git rev-parse HEAD)"
```

The fresh output identifies the actual checked source. Source-specific accepted
runs remain in the archive. Both destinations must be nonexistent. The full-suite
run used the same existing environment wrapper as the focused command above,
with `make check` (Python 3.12.3, NumPy 2.5.5, Torch 2.15.0+cpu). `make smoke` and
`make eval-smoke` remain NOT_AVAILABLE because train/eval CLI entry points are
absent. Full issue-28 delivery and measured/GPU/browser/Colab evidence remain open.
