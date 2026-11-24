# Reproducing the issue 25 review evidence

This record fixes two review-process gaps in the #25 documentation delivery. It is independent
of that PR stack and can merge against main without importing its proposed protocol. It does
not change the accepted dataset/scoring choices or complete #25's human freeze.

## Findings and changes

1. **Completion language exceeded the packet.** The interview handoff said no further material
   design questions remained. The [candidate at 9e46d64](https://github.com/Avistian/r0/blob/9e46d6424f61847e40153c334f3883f708bcae57/docs/protocols/single-table-v0/candidate.json)
   still leaves sampling allocation, predictor settings and numerical tolerances unresolved.
   These include choices, not only unavailable facts. The new review guidance classifies
   unresolved work and its owner/stage; accepted deferral remains distinct from resolved choice.
2. **An executed verification lacked its retained procedure.** The
   [record at c38ba8a](https://github.com/Avistian/r0/blob/c38ba8a02ff8cd54704f3106d733f17649d53adf/docs/evidence/issue-25/verification.md)
   reports a one-off arithmetic calculation and PASS outputs, but only the candidate-hash
   command was retained. Independent review recomputed the scores, which established arithmetic
   confidence without supplying the original executable recipe. The command below retains a
   reproduction of those checks and also checks the documented secondary scores.

The existing Make, commit-hook and CI checks run formatting, lint, typing, structural guards
and tests. They do not establish that narrative completion claims or verification procedures
cover the accepted scientific scope. These are judgement checks assigned to reviewers through
CODING_STANDARDS.md. No new generic linter, runtime harness or global AGENTS instruction is added.

## Reproduce the pinned packet

Use Python 3's standard library and Git, in a local repository containing commit
c38ba8a02ff8cd54704f3106d733f17649d53adf (PR #65's reviewed documentation stack).
Replace the quoted repository path below. The command reads only four explicit Git blobs;
it neither checks out a branch nor reads mutable working files. It fails if the commit is
unavailable. Obtain the repository revision through the normal Git workflow first; this
command performs no network access, data download, environment sync or model call.

```sh
python3 - "/path/to/r0" <<'PY'
import hashlib
import json
import math
import subprocess
import sys
from statistics import mean

revision = 'c38ba8a02ff8cd54704f3106d733f17649d53adf'
repository = sys.argv[1]

def read(path):
    return subprocess.check_output(
        ['git', '-C', repository, 'show', f'{revision}:{path}'])

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()

def same(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)

folder = 'docs/protocols/single-table-v0/'
packet = json.loads(read(folder + 'candidate.json'))
digest = hashlib.sha256(canonical(packet)).hexdigest()
assert digest == read(folder + 'candidate.sha256').decode().split()[0]
specification = packet['specification']
assert hashlib.sha256(read(specification['path'])).hexdigest() == specification['sha256']
assert hashlib.sha256(canonical(json.loads(json.dumps(packet, indent=4)))).hexdigest() == digest
changed = json.loads(json.dumps(packet))
changed['sampling']['query_seed'] = 99
assert hashlib.sha256(canonical(changed)).hexdigest() != digest
expected_roles = {(p, t) for p in ('development', 'milestone')
                  for t in ('binary', 'multiclass', 'regression')}
assert len(packet['datasets']) == 6
assert {(d['partition'], d['task_type']) for d in packet['datasets']} == expected_roles
assert all(d['admission'] == 'PENDING' and d['asset_sha256'] is None
           for d in packet['datasets'])
print('PASS packet binding, canonical identity, changed-split sensitivity, roles and pending assets')

fixture = json.loads(read('docs/evidence/issue-25/comparison.json'))
expected = {
    'binary': ([.05, .25, .29], [.125]*3, [.5]*3, 43/600),
    'multiclass': ([.10, .24, .375], [.14]*3, [2/3]*3, 59/600),
    'regression': ([math.sqrt(.5), 1, math.sqrt(.5)], [math.sqrt(2)]*3,
                   [math.sqrt(5), 2, math.sqrt(5)],
                   (-2/math.sqrt(10)+(1-math.sqrt(2))/2)/3),
}
assert len(fixture['cases']) == 3
assert {c['task_type'] for c in fixture['cases']} == set(expected)
for case in fixture['cases']:
    kind = case['task_type']
    targets = case['scoring_targets']
    query_ids = {row['row_id'] for row in case['query_features']}
    assert len(query_ids) == len(targets) == len(case['query_features']) == 2
    assert len(case['draws']) == 3
    scores = [[], [], []]
    secondary = [[], [], []]
    gaps = []
    for draw in case['draws']:
        context = draw['context']
        context_ids = {row['row_id'] for row in context}
        assert len(context_ids) == len(context) and not context_ids & query_ids
        labels = [row['y'] for row in context]
        vocabulary = case['class_vocabulary']
        if kind == 'regression':
            baseline = [mean(labels)] * len(targets)
        else:
            baseline = [[labels.count(c)/len(labels) for c in vocabulary]] * len(targets)
        predictions = draw['supplied_predictions']
        for index, values in enumerate((predictions['candidate_fixture'],
                                        predictions['reference_fixture'], baseline)):
            assert len(values) == len(targets)
            if kind == 'regression':
                assert all(math.isfinite(x) for x in values + targets)
                errors = [p-y for p, y in zip(values, targets)]
                primary = math.sqrt(mean(e*e for e in errors))
                other = mean(abs(e) for e in errors)
            else:
                assert all(len(row) == len(vocabulary) and
                           all(math.isfinite(x) and 0 <= x <= 1 for x in row) and
                           math.isclose(sum(row), 1) for row in values)
                primary = mean(sum((p-(c == y))**2 for p, c in zip(row, vocabulary))
                               for row, y in zip(values, targets))
                chosen = [vocabulary[max(range(len(row)), key=row.__getitem__)]
                          for row in values]
                other = mean(p == y for p, y in zip(chosen, targets))
            scores[index].append(primary)
            secondary[index].append(other)
        gap = scores[0][-1] - scores[1][-1]
        gaps.append(gap/scores[2][-1] if kind == 'regression' else gap)
    for calculated, literal in zip(scores, expected[kind][:3]):
        for actual, wanted in zip(calculated, literal):
            same(actual, wanted)
    secondary_expected = ([[.5, 1, .5], [1]*3, [2]*3] if kind == 'regression'
                          else [[1]*3, [1]*3, [.5]*3])
    for calculated, literal in zip(secondary, secondary_expected):
        for actual, wanted in zip(calculated, literal):
            same(actual, wanted)
    same(mean(gaps), expected[kind][3])
    assert mean(scores[0]) < mean(scores[2])
    print(f'PASS {kind}: mean={mean(scores[0]):.10f}; gap={mean(gaps):+.10f}; secondary checked')
print('Authored evidence only; model execution NOT_RUN; human freeze PENDING')
PY
```

The expected fractions/radicals come from the retained
[hand derivation](https://github.com/Avistian/r0/blob/c38ba8a02ff8cd54704f3106d733f17649d53adf/docs/evidence/issue-25/README.md).
The calculations above use the row-level literals independently of the displayed result table.
The command intentionally verifies one immutable packet, not arbitrary future protocol schemas.
Use normal Python execution, not optimization that disables assertions.

## Observed reproduction

The exact fenced command was executed locally against the pinned Git blobs. Result:

```text
PASS packet binding, canonical identity, changed-split sensitivity, roles and pending assets
PASS binary: mean=0.1966666667; gap=+0.0716666667; secondary checked
PASS multiclass: mean=0.2383333333; gap=+0.0983333333; secondary checked
PASS regression: mean=0.8047378541; gap=-0.2798541044; secondary checked
Authored evidence only; model execution NOT_RUN; human freeze PENDING
```

This repairs the reproduction gap without claiming that the original PR retained this command.
The earlier bounded review verdicts remain historical evidence. Real dataset eligibility,
TabPFN/model performance, resource enforcement and final human acceptance remain unverified.
