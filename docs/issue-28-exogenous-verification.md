# Issue #28 exogenous checkpoint verification

This records partial implementation evidence for [the exogenous/noise laws](issue-28-exogenous-laws.md),
not completion of the [accepted scope](specs/0028-causal-network-generators.md).

## Standards

Independent standards review of `19dd757...f75c9cd` found **no hard documented
violations**. One nonblocking heuristic remains: the causal-profile names are
repeated in dispatch and validation registries, so a future profile addition
needs synchronized edits. No current mismatch was demonstrated. This checkpoint
leaves the existing registry layout intact; it is not a missing scope item.

The scoped review of `f75c9cd..684a3ec` found no new violations or heuristics.
It inspected the zero-scale/overflow repair and its public-interface regression
cases without running tests. Numerical verification belongs to the spec review.
These verdicts concern the partial checkpoint and do not accept all of #28.

## Spec

Independent review **REJECTED** `f75c9cd` after 54 preceding public-interface
checks passed. A root with bias `1e200`, base noise scale zero and response
`(offset=1,strength=1,power=2)` evaluated an overflowing power before multiplying
by zero. The mathematically finite deterministic task was rejected. This violates
the zero-path promise and accepted decisions 3/13 about retaining valid extremes.
The author identified the concern; the reviewer independently reproduced it
against an archived frozen revision. The [original result](evidence/issue-28/legacy-research/0028-exogenous-original-review.json)
retains that attribution, passing probes and failure trace. An initial verifier
NumPy-scalar descriptor error was a harness defect, retained separately by the
reviewer, not an implementation finding.

RED `71c365d` adds three failures: the zero-scale case and two representable final
scales whose intermediate powers/products overflow. GREEN `684a3ec` masks zero
structural scales before power evaluation and rescues nonfinite positive terms
through their logarithms. Ordinary finite arithmetic stays unchanged; genuinely
nonfinite final scales still reject. The [original regression output](evidence/issue-28/legacy-research/0028-exogenous-overflow-red.txt)
is retained. All 12 authored exogenous tests pass on the repaired source; command:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest tests/priors/test_causal_exogenous.py --tb=short
```

The first [full-check attempt](evidence/issue-28/legacy-research/0028-exogenous-check-f75c9cd.txt) stopped on
formatting in an earlier activation-prior documentation example. `a7d24ad` fixes
that documentation. Its [next attempt](evidence/issue-28/legacy-research/0028-exogenous-check-a7d24ad-interrupted.txt)
passed static gates and 172 tests before the author interrupted it after finding
the scale defect. Neither attempt is reported as a complete full-check pass.


A second independent recheck **REJECTED** `684a3ec` after 60 passing probes.
The earlier overflow fixes passed, including mixed vectors and genuine final
nonfinite-scale rejection. The new finding was symmetric underflow: bias
`1e-200`, scale `1e200` and response `(0,1,2)` produced scale zero rather than
`1e-200`. Its independently calculated 100-digit Decimal result, actual import
paths and executed program hashes remain in the [receipt](evidence/issue-28/legacy-research/0028-exogenous-underflow-review.json)
and [probe report](evidence/issue-28/legacy-research/0028-exogenous-underflow-probes.json). The original
combined artifact is retained locally; publication separates its report without
changing its values to satisfy the text-file limit.

RED `731b454` adds three underflow cases, covering power and coefficient-product
underflow. Source repair `882637f` detects zero/subnormal intermediates before
offset addition and reconstructs only affected positive terms in logarithms.
The [RED result](evidence/issue-28/legacy-research/0028-exogenous-underflow-red.txt) remains distinct from
passing verification. The full check at `684a3ec` was interrupted when this defect
was confirmed; its [partial output](evidence/issue-28/legacy-research/0028-exogenous-check-684a3ec-interrupted.txt)
is not a completed pass.

The first two repaired test attempts failed overly strict target comparisons:
[exact equality](evidence/issue-28/legacy-research/0028-exogenous-underflow-tolerance-attempt-01.txt), then
[relative error near cancellation](evidence/issue-28/legacy-research/0028-exogenous-underflow-tolerance-attempt-02.txt).
They remain retained. `a47b14e` propagates the independently asserted scale error
through tanh's 1-Lipschitz bound. The scale assertion still has zero absolute
tolerance, so losing `1e-200` to zero fails. The output bound scales with the
innovation magnitude and is capped for large values. This is a numerical
tolerance correction, not removal of the underflow check. The [15-case focused
output](evidence/issue-28/legacy-research/0028-exogenous-focused-a47b14e.txt) passes with the command above.

Scoped standards review at `a47b14e` accepts the repair shape and error-bound
rationale, finding no new hard violations or heuristics; it did not execute tests.
The [three original standards reports](evidence/issue-28/legacy-research/0028-exogenous-standards.md) retain
exact revisions, procedures and the remaining nonblocking registry heuristic.


## Final scoped verification

Independent spec recheck **PASS** at `a47b14e`: all **135 checks** pass. These
include all four base laws with and without truncation; prefix/replay/selected-row
reconstruction; ball radial moments; factor orientation and analytic covariance;
nonlinear noise equations; moment boundaries; invalid descriptors; admission
arithmetic; sampled profile serialization; and the two numerical repairs.
Eighteen canonical recipe/card/input/score snapshots exactly match base `19dd757`
across all three older profiles and all three topology families.

The independent 100-digit Decimal boundary grid has 73 cases: 64 finite cases,
including 13 positive mathematical scales correctly rounding to zero at float64
precision, and nine genuinely overflowing final scales explicitly rejected.
It includes positive offsets hiding underflow, subnormal powers/products,
coefficient overflow, mixed zero/nonzero channels and minimal final subnormals.
These bounded checks do not certify every possible floating-point composition.

The [receipt](evidence/issue-28/legacy-research/0028-exogenous-recheck-receipt.json) identifies exact head,
base, imports, commands and executed script hashes. Its ordered three probe parts
retain all 135 check records without changing their values. To reconstruct the
original report object, concatenate the arrays named in `checks_parts` in order,
replace `checks_parts` with `checks`, then remove the publication-only
`unsplit_original` entry. The original combined JSON remains at its recorded local
path with a SHA-256 fingerprint. Both historical REJECT records remain separate.

The [runner](../scripts/verify_causal_exogenous.py), [public-interface probes](../scripts/verify_causal_exogenous_probes.py)
and [Decimal probes](../scripts/verify_causal_exogenous_repair.py) are byte-identical
to the independently executed final package. The runner archives both Git
revisions, isolates source imports, records actual HEAD before and after, and
refuses to overwrite an existing output. From an unchanged checkout, using a new
output path:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py python scripts/verify_causal_exogenous.py \
  --repo . --expected-head HEAD \
  --base 19dd757bcbc978bd56d1c6ed76e1e71a7fd824c5 \
  --output /tmp/r0-28-exogenous-independent-reproduction.json
```

At source `a47b14e`, complete `make check` passes **597 tests, 1 skipped**, with all
preflight/static gates passing. The [complete output](evidence/issue-28/legacy-research/0028-exogenous-check-a47b14e.txt)
is retained verbatim. The skip is the existing GPU agreement check because CUDA
is unavailable. Command/environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

Smoke/eval-smoke remain NOT_AVAILABLE because T7/T12 train/eval CLI commands have
not landed. This is local CPU verification, not hosted CI, browser/Colab proof or
predictive-performance evidence. Publication adds documentation and retained
verification programs; package and test source remain identical to `a47b14e`.
The full #28 implementation goal stays active.
