# Issue 28: independent target calibration 

This extends [authored functional graphs](issue-28-functional-graph-integration.md)
with `CalibratedFunctionalCausalRecipe` (`functional-causal-v2`). It implements the
independent target-normalization part of accepted specification decisions 12–13;
full sampled-graph construction and ancestral function/pipeline fitting remain open.
Version one retains its original unnormalized schema and stream namespace.

## Contract

The version-two recipe selects 32–65,536 calibration rows (default 256). Generation
uses a separate `target-calibration-v1` keyed population from the same frozen graph.
The target readouts include the existing converter-capture and structural-noise
ordering. Features, context/query rows, labels and emitted row count do not determine
these statistics. This independent fit is an r0 adaptation, not a claim of exact
first-batch fitting behavior from a paper implementation.

Each target channel receives its independent population's empirical mean and
population standard deviation (`ddof=0`). The graph's conservative moment analysis
must certify finite second moments before normalization. Finite sampled values do
not bypass that mathematical gate. Scaling before moment estimation avoids squaring
extreme raw values. Arithmetic failures raise; there is no clipping or replacement
draw. Constant calibration columns retain identity scaling (center zero, scale one)
and their indices are recorded; constancy in a calibration sample is not a proof
that the population is constant.

The mechanism card retains row count, center, scale, constant-channel indices,
stream/version, a canonical little-endian float64 calibration-population digest,
and a fingerprint of the frozen nodes and target readouts. It does not store the
full calibration population. Seed/task identity plus the frozen recipe and named
stream make that population reproducible. This digest is provenance, not a signature
or proof against coordinated alteration of saved artifacts.

Inspection applies retained constants without refitting and checks the graph
fingerprint, row count, record shape and reconstructed scores. Repeating generation
with the same recipe/seed/task identity deterministically recomputes the independent
fit. Changing the task identity may change that fit; changing emitted row count does
not. Version two does not yet expose reusable sampled-graph construction receipts.

Calibration numeric metadata and its maximum row count are charged to graph
admission. These are conservative declared bounds, not measured process RSS or an
OS memory limit. Calibration runs on CPU; no GPU evidence is claimed.

## Checks and remaining scope

Public tests cover fixed constants across emitted row counts, selected-row inspection,
finite-second-moment rejection, constant-target identity, malformed records, resource
admission, and cancellation of linear scales at `1e300` and `1e-300`. The shared
session test covers create/save/reopen/replay for both graph versions, all four
current root bases, and regression/binary/multiclass adapters. These are session/API
checks, not browser or hosted Colab checks.

Development checks exposed two decoder regressions: non-object graph records leaked
`AttributeError` after version dispatch, and missing calibration cards leaked
`KeyError` in inspection. Both now reject with `ValueError`; original failures remain
in the [development archive](evidence/issue-28/legacy-research/0028-functional-calibration-development.json). Intermediate uncommitted source states are not all retained; those historical runs alone are not a complete reconstruction of each intermediate revision.

The candidate passed `make check`: **1255 passed, one skipped**, 526.98 seconds, with all static gates. Independent bounded review passed, then passed again on publication source `09cdbbab88de5cc9b04702bb7433aea47f8ae160`, whose source/test trees match checked candidate `ca50743`. No overall issue-28 completion,
browser/Colab readiness, throughput, scale, scientific or GPU acceptance follows
from these candidate checks. Sampled-graph operation ownership, complete raw-draw
retention, fresh upstream populations for every receiving fit and final atomic
commit remain separate work, guided by
[the ownership research](research/0028-sampled-graph-ownership.md).


## Retained evidence and reproduction

- [Full check](evidence/issue-28/legacy-research/0028-functional-calibration-check-ca50743.txt), [candidate independent review](evidence/issue-28/legacy-research/0028-functional-calibration-candidate-review.md), and [publication-source independent replay](evidence/issue-28/legacy-research/0028-functional-calibration-publication-review.md).
- Independent checks include a separate scalar NumPy oracle for nonlinear downstream values, structural noise and converter capture; changes in emitted rows/split/label noise; strict moment boundaries; multiclass targets; malformed metadata and admission; retained-constant inspection/session replay; `1e300`, `1e-300` and finite normalization despite overflowing subtraction. Twelve version-one cases across four roots and three task adapters match 129 retained baseline files byte-for-byte.
- [Standards review](evidence/issue-28/legacy-research/0028-functional-calibration-standards.txt): zero documented violations and one nonblocking repeated-version-check heuristic. No tests are attributed to that reviewer.
- [Archive metadata](evidence/issue-28/legacy-research/0028-functional-calibration-spec-artifacts.json) and [recovery receipt](evidence/issue-28/legacy-research/0028-functional-calibration-spec-recovery.json): 2641 verified file identities. Original reviewer scripts, source snapshots, raw outputs, environment pins and the corrected archive-ID harness error remain immutable.
- [Public programs](../scripts/verify_functional_calibration) are linted copies with a reachable equivalent baseline and an additional nonempty-baseline guard. Separate author replays passed on the candidate and publication source. Final repository lint then inserted two blank import-separator lines; token sequences are unchanged and these whitespace-only copies were not rerun. The executed copies remain in the archive. These copies do not replace the original independently executed bytes. [Publication gates](evidence/issue-28/legacy-research/0028-functional-calibration-publication-gates.txt) cover the document/script delta; product source and tests are unchanged.

From the repository root, recover every retained file and validate its checksum:

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-functional-calibration-spec-artifacts.json \
  /tmp/functional-calibration-recovered --repo .
```

Use the existing supported Python environment, a clean checkout (untracked `.venv`
is permitted), and a new output directory for a fresh public-program run:

```sh
python scripts/verify_functional_calibration/replay.py \
  --source-root "$PWD" \
  --output-dir /tmp/functional-calibration-fresh \
  --expected-head "$(git rev-parse HEAD)"
```

Both destinations must be nonexistent. The fresh run reports the source actually
checked; accepted source-specific results remain in the archive. The retained
full-suite command used Python 3.12.3, NumPy 2.5.5 and Torch 2.15.0+cpu:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv make check
```

`make smoke` and `make eval-smoke` remain NOT_AVAILABLE because their train/eval
CLI entry points are absent. No model loss or evaluation table is claimed.
