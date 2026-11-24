# Single-table protocol candidate

The [accepted specification](../../specs/0025-evaluation-protocol.md) owns decisions.
[candidate.json](candidate.json) is a review packet, **not an executable suite or a frozen
benchmark**. Its null fields and PENDING states block real milestone evaluation. No assets
were acquired, no training/TabPFN call was made, and no dataset is admitted by this PR.

## Read the packet

Each dataset has a role, task type, source DOI/URL, collection family, published dimensions,
target meaning and eligibility concern. Published counts are metadata, not eligible support.
The target-column names and feature allowlists await inspection of actual owner-provisioned
files. FLIR's feature count is null because a final allowlist has not been verified.

The candidate seeds are fixed implementation choices: query seed 0 and context seeds 1, 2, 3.
Do not search seeds for favorable scores. Sampler version, ordering, class allocation and
row-identity recipe must be specified before final row selection. The final receipt retains
actual row IDs as well as seeds. Seeds alone do not define a split or prove reproducibility.

Exact package and weight identities are intentionally absent: 'TabPFN v2' is a model-family
choice, not a reproducible asset pin. The five predictor roles are requirements for #26,
not installed dependencies or implemented adapters. Its settings, preprocessing and numerical
tolerances must be frozen before access. Five minutes/8 GiB are unmeasured requirements,
not an assurance of success. With five roles, six datasets and three draws there are 90
attempts (up to 450 wall-clock minutes if serial and all reach the cap), excluding retries
and reporting. This is not a weekly human-work estimate or run/spend authorization.

## Provenance and overlap findings

Official metadata was inspected on 2026-11-24 without downloading dataset payloads. Follow
the per-dataset source links; retain their versions and notices during owner provisioning.

| Dataset | Attribution to retain | Metadata observation requiring follow-up |
| --- | --- | --- |
| WDBC | Wolberg, Mangasarian, Street and Street | 569 rows leaves little margin above 512 |
| ILPD | Ramana and Venkateswarlu | Prose count 584 differs from table/class total 583 |
| CMC | Tjen-Sien Lim; underlying Indonesian survey | Check survey clusters and label vocabulary |
| Dry Bean | Koklu and Ozkan | Individual grains; check duplicate/acquisition identities |
| Abalone | Nash, Sellers, Talbot, Cawthorn and Ford | Published filtering/scaling precedes this protocol |
| FLIR oral | Wang et al., PhysioNet v1.0.0 | Repeated measurements summarized; camera files overlap |

UCI metadata for the first five declares CC BY 4.0; the original PhysioNet source declares
CC0 1.0. These statements do not substitute for checking embedded notices and publication
rights for row-level targets/predictions. Preserve attribution and owner evidence custody.

The six displayed names and described collection families did not show an evident match in
[TabArena-v0.1 Table B.2](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf).
This is metadata screening only, not cleared overlap. Exact aliases, file shapes, versions,
source relationships and content comparisons remain PENDING. Pin the compared TabArena
inventory in the final overlap receipt. No reviewed description establishes mandatory grouped
or temporal splitting for these scoped tasks; inspect provisioned identifiers before admission.
TabPFN pretraining/development exposure remains UNKNOWN for all six; no novelty claim follows.

## Identity recipe

candidate.sha256 records SHA-256 of UTF-8 JSON parsed and reserialized with sorted keys,
compact separators, ensure_ascii=False and allow_nan=False. Object whitespace/order does not
matter; array order does. The manifest embeds the SHA-256 of the accepted specification's file
bytes, so narrative rules cannot silently drift away from the packet. A specification change
requires refreshing that reference and the candidate digest. Freeze tooling belongs to #26.

Recompute locally with the standard library (no environment sync or network):

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
p = Path('docs/protocols/single-table-v0')
m = json.loads((p / 'candidate.json').read_text())
s = m['specification']
assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest() == s['sha256']
b = json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
               allow_nan=False).encode()
actual = hashlib.sha256(b).hexdigest()
assert actual == (p / 'candidate.sha256').read_text().split()[0]
print('Candidate identity verified:', actual, '(human freeze PENDING)')
PY
```

This digest is not a frozen protocol hash, asset verification or a human approval. When all
prerequisites are satisfied, the human freezes the fully resolved manifest with a separate
acceptance receipt referencing its content hash. Keep receipt identity outside its own hash
input. Preserve the old candidate/attempts in Git history. Never reuse caches across different
protocol identities; a mismatch rejects comparison in #26's future harness.

## Handoff and freeze checklist

Review one dataset per 2–3-hour weekly session; the other work can be reviewed in separate PRs.
For each candidate, the owner supplies an acceptance or rejection receipt containing:

1. Exact files, revisions, hashes, acquisition provenance, terms and attribution.
2. Target column/vocabulary, feature allowlist, identifiers, source processing and collection
   review. Inspect grouping, missing targets and duplicate/conflicting records before splitting.
3. Exclusion IDs/reasons, original/eligible counts and proof that 512 distinct rows remain.
4. Cleared overlap against development/milestone families and the pinned TabArena inventory;
   retain uncertainties/exclusions. UNKNOWN TabPFN exposure remains separately disclosed.
5. Canonical row recipe, explicit sampler/allocation rules and seeds; 256 query IDs and three
   sets of 256 context IDs, all-class coverage and context/query disjointness.

Before final freeze also record exact baseline assets/configurations, permitted preprocessing,
probability/invariance tolerances, CPU/thread/environment identities and cap enforcement.
These are unresolved prerequisites, not optional fields an adapter may silently default.
Record owner acceptance of the worked scoring example and a fully resolved protocol hash.
The learner selection and compute/access receipts are required before the later milestone run;
training a learner is not a prerequisite to #25's protocol design/freeze.

Keep milestone labels outside development and retain a one-use access receipt. An exposed
holdout cannot be made fresh by renaming a manifest. Preserve #24's private-case custody and
verify the owner's receipt separately; issue closure is not that receipt.

Human handling is required for eval/suite_v0.toml, eval/suite_v0.lock.json,
src/r0/modules/evaluation/, scripts/check_ratchet.py, results/best.json and ADR 0003. This PR
does not edit them. Reviewer should label needs-a-human for the freeze handoff. #25 remains
open; #26 may build fixture-backed interfaces from the accepted design but must not claim
real benchmark acceptance until the freeze is recorded.
