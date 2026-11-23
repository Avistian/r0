# Retained local observations

Implementation checked: `412ba3b` (evidence-only packaging follows).

Local `make check`: **82 passed**, formatting/lint/Pyright/import boundaries/checkout/module/file guards passed. Training/evaluation smoke targets are not implemented yet. Hosted CI is separate and was not run for this receipt.

Walkthrough: **PASS**, 15 scenarios; observed duration 1.274s, not a hardware-independent limit.

| Scenario | Acceptance |
| --- | --- |
| default manifest | PASS |
| biased filtering | PASS |
| request-order replay | PASS |
| same proposals under selection | PASS |
| paired noise and legitimate noisy tasks | PASS |
| exact content with new partition | PASS |
| same mechanism new observations | PASS |
| row permutation equivalence | PASS |
| bijective class renaming | PASS |
| corrupted target | PASS |
| lost proposal | PASS |
| query-target isolation | PASS |
| independent worked rows | PASS |
| generation failure retention | PASS |
| representative variants | PASS |

## Worked numeric examples

Each regression example includes the recorded nonzero target-noise realization. Full layers, paths, ancestors, kernel construction and classification adaptations are in the archive.

| Family | Expected target | Observed target | Difference |
| --- | ---: | ---: | ---: |
| linear | 0.692195856667 | 0.692195856667 | -1.11e-16 |
| gp | -0.705433244845 | -0.705433244845 | 0 |
| mlp | 1.34194524534 | 1.34194524534 | 2.22e-16 |
| scm | 0.945392003322 | 0.945392003322 | 0 |
| tree | 0.874868387867 | 0.874868387867 | 0 |
| interaction | -1.5063301037 | -1.5063301037 | 0 |

## Sparse/high-noise trace

Proposal `2b663c4ab37dba56e90021fed8373b4e08714684f087bdc8e97e65c15d350a84` belongs to `linear/added/regression`, with 1 relevant and 7 nuisance features. Its cell has proposed=2, generated=2, valid=2. Unfiltered selected=2, share=2/72. Under the biased filter selected=0, share=0/36; within-cell retention=0/2. Both valid tasks are explicitly rejected by the demonstration noise rule, not declared invalid.

## Archives

Extract all three accepted archives in one disposable directory, for example `for f in accepted-*.tar.gz; do tar -xzf "$f"; done`. `accepted/reference/manifest.json` checks every run file; `accepted/walkthrough.json` links the controls and run identities. Historical evidence preserves the original missing-fork rejection and its counts. Its baseline proposals are unchanged in the accepted raw records; only the supplemental example and stricter verification were added.

- `accepted-biased.tar.gz`: 740714 bytes; SHA-256 `d60d9420878a540acc24845332ab033d0897c0c67c0945e099238a011a521c08`.
- `accepted-controls.tar.gz`: 58860 bytes; SHA-256 `d729d02bc8c8d4add0aef42488cd7050a466f987fd4a0a843fcd4aaf06a759f9`.
- `accepted-reference.tar.gz`: 740593 bytes; SHA-256 `493f31b8d25a2ebdee2f91730ad0919976e4016bfcb4e0ce3466cd004a6058ca`.
- `historical-fork-gap.tar.gz`: 2936 bytes; SHA-256 `eb7256a99bf056d1896829dd7950fef8ba7a187b6786fa4edbcf434554f3cb42`.
