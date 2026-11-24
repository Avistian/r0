# Verify the owner-receipt transcription

From the repository root, run the following standard-library command. Its argument is the
folder containing the four owner receipts; it reads those files only, not raw datasets,
weights, milestone rows or targets. It prints only a bounded success summary. The command
checks reported hashes against the supplied SHA256SUMS, not the underlying raw files.
The source receipt hashes identify the exact inputs used for this candidate revision.

```sh
python3 - "$HOME/r0-assets/issue-25" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

assets = Path(sys.argv[1])
folder = Path('docs/protocols/single-table-v0')
manifest = json.loads((folder / 'candidate.json').read_text())
canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False).encode()
assert hashlib.sha256(canonical).hexdigest() == (folder / 'candidate.sha256').read_text().split()[0]
reference = manifest['specification']
assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256']
assert reference['acceptance'] == 'PENDING_PIOTR_REACCEPTANCE_AFTER_SPAMBASE_REVISION'
ref = manifest['owner_receipt']
assert hashlib.sha256((folder / ref['path']).read_bytes()).hexdigest() == ref['sha256']
receipt = json.loads((folder / ref['path']).read_text())
for name, expected in receipt['provisioning_receipt']['files'].items():
    assert name in {'codex-brief.md', 'asset-report.md', 'asset-check.json', 'SHA256SUMS.txt'}
    assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == expected
checks = json.loads((assets / 'asset-check.json').read_text())
sums = {line.split(None, 1)[1].strip(): line.split()[0]
        for line in (assets / 'SHA256SUMS.txt').read_text().splitlines()}
assert checks['_sha256sums'] == {'files': 34, 'mismatches': []}
assert manifest['state'] == 'CANDIDATE_NOT_FROZEN'
assert manifest['human_acceptance'] == 'PENDING'
assert {d['id'] for d in manifest['datasets']} == {
    'wdbc', 'cmc', 'abalone', 'spambase', 'dry-bean', 'flir-oral'}
assert len(manifest['datasets']) == 6
count_keys = ('raw_rows', 'excluded_invalid_target', 'excluded_exact_duplicates',
              'quarantined_conflict_rows', 'quarantined_conflict_groups', 'eligible_rows')
excluded_total = 0
for data in manifest['datasets']:
    source = checks[data['id']]
    assert data['admission'] == 'PENDING' and data['tabpfn_exposure'] == 'UNKNOWN'
    assert data['query_row_ids'] is None and data['context_row_ids'] is None
    for field in ('target_column', 'feature_allowlist', 'eligible_rows'):
        assert data[field] == source[field]
    assert data['class_vocabulary'] == source.get('class_vocabulary')
    assert data['asset_files']['data_member'] == source['data_member']
    assert data['asset_sha256']['downloaded_file'] == source['primary_file_sha256']
    assert sums[source['primary_file']] == source['primary_file_sha256']
    member_hash = source.get('zip_members_sha256', {}).get(
        source['data_member'], source['primary_file_sha256'])
    assert data['asset_sha256']['data_member'] == member_hash
    assert member_hash in sums.values()
    exclusion = data['exclusions_receipt']
    assert all(exclusion[k] == source[k] for k in count_keys)
    assert exclusion['raw_rows'] - exclusion['excluded_invalid_target'] - exclusion[
        'excluded_exact_duplicates'] - exclusion['quarantined_conflict_rows'] == data['eligible_rows']
    rule_order = ['invalid_target', 'exact_duplicate_keep_first', 'conflicting_target_group']
    assert exclusion['rule_order'] == rule_order
    ids = exclusion['excluded_row_ids']
    if ids:
        path = folder / ids['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == ids['sha256']
        excluded = json.loads(path.read_text())
        assert excluded['dataset'] == data['id']
        assert excluded['rule_order'] == rule_order
        ids = excluded['excluded_row_ids']
    positions = [position for group in ids.values() for position in group]
    assert len(positions) == len(set(positions))
    assert all(type(i) is int and 0 <= i < source['raw_rows'] for i in positions)
    actual = {str(i): reason for reason, group in ids.items() for i in group}
    assert actual == source['excluded_row_ids']
    excluded_total += len(actual)
    assert Counter(actual.values()) == Counter(source['excluded_row_ids'].values())
    for field in ('embedded_terms_review', 'collection_review', 'overlap_review'):
        assert data[field].endswith('proposed: ELIGIBLE; awaiting owner')
    assert not {'class_counts', 'target_stats', 'query_allocation', 'context_allocation'} & data.keys()
flir = next(d for d in manifest['datasets'] if d['id'] == 'flir-oral')
assert flir['published_features'] == len(flir['feature_allowlist']) == 115
assert not {'aveOralM', 'aveOralF', 'SubjectID', 'Date', 'Time'} & set(flir['feature_allowlist'])
for key in ('sampler_version', 'row_identity_recipe', 'allocation_tie_break'):
    assert manifest['sampling'][key] is None
assert excluded_total == 642
for weight in manifest['reference']['weight_sha256'].values():
    assert weight in sums.values()
for name, digest in receipt['overlap_inventory']['files'].items():
    assert sums['tabarena-inventory/' + name] == digest
print('PASS: six dataset receipts, 642 excluded IDs, FLIR allowlist, source/spec/packet hashes')
print('Admission PENDING; human acceptance PENDING; edited spec needs Piotr re-acceptance')
PY
```

Use ordinary Python execution (not optimization that disables assertions). Passing verifies
transcription and receipt consistency only. It does not run the owner checker, assess legal
terms, establish dataset independence, execute the sampler or approve the protocol.
