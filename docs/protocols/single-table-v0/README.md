# Single-table protocol candidate: owner-provisioned assets

The [revised specification](../../specs/0025-evaluation-protocol.md) requires Piotr's
re-acceptance. [candidate.json](candidate.json) remains CANDIDATE_NOT_FROZEN; human_acceptance
and every dataset admission remain PENDING. Proposed ELIGIBLE findings are not admission.
No model, training or TabPFN run was performed. The manifest is not an executable harness.

## Owner decisions and evidence boundary

On 2026-11-24 Piotr replaced ILPD with Spambase because ILPD occurs in BeyondArena (same UCI
collection/DOI). This was before any scoring. The binary pair is now WDBC development /
Spambase milestone. Record a new candidate digest; no frozen protocol or historical score
is being replaced. The previously accepted spec revision is not approval of this edit.

Piotr also approved the exact 115-feature FLIR list, including image-derived T_OR*/T_OR_Max*,
Ethnicity and banded Age. published_features is now 115 for that accepted allowlist. The
thermal mouth-region features are image measurements, distinct from the excluded oral
thermometer reading aveOralF. Subject/date/time IDs and spacer columns remain excluded;
only FLIR_groups1and2.csv is used. The feature decision does not admit the dataset.

This packet transcribes owner-provisioned codex-brief.md, asset-report.md, asset-check.json
and SHA256SUMS.txt, accessed 2026-11-24. [owner-receipt.json](owner-receipt.json) identifies
those source files by hash and records the reported checksum result: 34 files, no mismatches.
The update cross-checks these receipts; it does not reopen data/weights or rerun check_assets.py.
The raw datasets, weights, rows, class counts, target statistics and query/context allocations
are not committed. The owner's external report remains necessary for a full audit.

## Dataset findings

| Dataset / role | Raw → eligible | Exclusions | Attribution / material finding |
| --- | --- | --- | --- |
| WDBC / dev binary | 569 → 569 | none | Wolberg, Mangasarian, Street and Street; unique source IDs, repeated patients still uncertain |
| Spambase / milestone binary | 4601 → 4204 | 391 duplicates; 6 conflicting rows in 3 groups | Hopkins, Reeber, Forman and Suermondt (HP Labs); source-identity shortcuts george/650 retained |
| CMC / dev multiclass | 1473 → 1296 | 48 duplicates; 129 conflicting rows in 62 groups | Tjen-Sien Lim / Indonesian survey; disclose 177 exclusions (12%) and residual survey clustering |
| Dry Bean / milestone multiclass | 13611 → 13543 | 68 duplicates | Koklu and Ozkan; embedded citation request, residual acquisition clustering |
| Abalone / dev regression | 4177 → 4177 | none | Nash, Sellers, Talbot, Cawthorn and Ford; upstream missing-row removal/scaling, residual site clustering |
| FLIR oral / milestone regression | 1020 → 1020 | none | Wang et al.; one row per unique SubjectID, four measurement rounds within columns; undetectable repeat visits remain possible |

These are owner-reported support counts, not newly measured results. All proposed verdicts
are ELIGIBLE, awaiting Piotr's dataset admission. UCI pages declare CC BY 4.0 for the first
five; PhysioNet LICENSE.txt declares CC0 1.0. Embedded terms, collection limitations and
source URLs/DOIs are recorded per dataset. Spambase's single-mailbox non-spam provenance
makes word_freq_george and word_freq_650 known source shortcuts; this limits generalization
claims and is disclosed rather than silently filtered or used to change the task.

Each exclusions_receipt retains the counts and rule order: exclude invalid targets, keep the
first exact duplicate over allowlisted features plus target, then quarantine conflicting
identical-feature groups. No invalid-target rows were reported. Nonempty excluded_row_ids
are hash-bound companion files under exclusions/, grouped by reason; each integer is a
zero-based data-row position after headers. Empty sets are inline. All 642 excluded IDs
are retained, without row values. This layout keeps every file below the repository size cap.
Header conventions belong to the owner's parser (FLIR's real header is row 3); source row
positions are not a frozen sampler or canonical row-identity recipe.

## Overlap receipt and remaining limits

The owner screened names, aliases, source URLs and DOIs against:

- autogluon/tabarena at 39c3b4a8dcd18633551f9b1ff0585948250d8435: the 51-dataset
  curated_tabarena_dataset_metadata.csv and 143-dataset BeyondArena_core2k_tasks.csv.
- HF TabArena/BeyondArena at 2ecfe882ccfb814fc27c4de10a64ceefd5d7655c.

Full inventory hashes are in owner-receipt.json. ILPD matched and was replaced. The six
current collections had no match in that screening; no dataset content-hash comparison was
performed. BeyondArena's ljubljana_breast_cancer is UCI 14, a different collection from WDBC.
Abalone appears as a repository test fixture, not an entry in those benchmark inventories.
Development and milestone pairs come from different collections. These are recorded owner
findings, not a new automated overlap proof. TabPFN exposure stays UNKNOWN for every dataset.

## Reference pins and acknowledged caveat

The candidate records the report's recommended tabpfn==2.2.1 wheel hash and exact classifier /
regressor repository revisions, filenames and weight hashes. Final package selection remains
PENDING: the report also offers 10.2.0 with explicit ModelVersion.V2. The package is part of
identity; neither option may inherit upstream model defaults. Executors must use explicit
local model_path assets, verify their hashes and disable telemetry with TABPFN_DISABLE_TELEMETRY=1.
No installation, download or inference was performed for this update.

Piotr OKd the supplied Prior Labs License v1.1 caveat. The owner receipt calls for the license
copy and “Built with PriorLabs-TabPFN” attribution for public results, and flags the naming
clause if outputs improve a distributed model. Keep TabPFN a descriptive baseline: no outputs
as training/selection signals. This records the supplied caveat and owner's acknowledgment;
it is neither a new legal interpretation nor acceptance of the complete protocol.

## Identity and verification

candidate.sha256 is SHA-256 of parsed JSON reserialized as UTF-8 with sorted keys, compact
separators, ensure_ascii=False and allow_nan=False. Object whitespace/order does not matter;
array order does. specification.sha256 binds the edited spec's file bytes. The manifest also
binds owner-receipt.json and nonempty exclusion files by file-byte hashes. Refresh referenced
hashes before the candidate digest whenever a bound artifact changes. A formatting change to
a byte-hashed companion requires refreshing its reference; the top-level canonical JSON
identity ignores only its own formatting.

Run the [retained verification command](verify.md) against the owner's four receipt files.
It checks the full metadata transcription and every excluded ID without reading raw assets.
Neither the candidate digest nor that check is a frozen protocol identity or human acceptance.

## Pending decisions, verification and authorization

- **Piotr:** re-accept the edited spec, admit/reject each dataset, accept the fully resolved
  protocol and publish its final freeze receipt. FLIR-feature and license-caveat approval
  have already been supplied; they do not clear these other approvals.
- **#26 and Piotr:** choose/freeze package configuration, preprocessing, sampler/version,
  canonical row identity, allocation tie-break, CPU/thread/environment, numerical tolerances
  and enforcement. The largest-remainder allocation in the owner checker is feasibility only,
  not the accepted sampler. Query/context row IDs therefore remain null. Candidate seeds
  remain 0 for queries and 1/2/3 for context; do not search them for favorable results.
- **Harness/run owner:** establish resource feasibility and retain the future learner-selection,
  compute and one-use access receipts. Per-attempt caps remain five minutes/8 GiB including
  preprocessing, loading, fitting and prediction; scoring/reporting is recorded separately.
  Five predictors × six datasets × three draws permits 90 attempts (450 serial minutes at
  the cap, before retries/reporting), not an authorization to spend or execute.

Review one dataset per 2–3-hour weekly session. Keep milestone labels/scores outside development;
public data is not secret, so record access/influence. Preserve #24's private-case custody.
#25 stays open. Protected evaluation files and ADR 0003 remain untouched and require human
handling. #26 owns harness enforcement; this packet does not certify it or unblock real runs.
