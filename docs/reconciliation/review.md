# Acceptance walkthrough

## Validation

On the authoring checkout at 51a3bebb plus this documentation delta, UV_OFFLINE=1 make check passed. Formatting and lint passed, pyright reported zero errors and warnings, all three import contracts held, module/file-size checks passed, and the full suite passed 23 tests. Explicit file-size and whitespace checks cover the added documents. These results validate repository consistency; the audit claims use the separate pinned source and direct walkthrough.

Training smoke and evaluation smoke were not run because those capabilities are absent. No source behavior changed. No new tests were added for the documentation audit.

A final git ls-remote origin refs/heads/main returned 807da066ef3d0df1090e1ef99b7340b11eca4dc4, identical to the audit baseline. No intervening remote changes required reconciliation.

## Independent review history

The standards reviewer initially marked PENDING because this receipt was still a placeholder and the historical 64/256/512 grid had not been written into the crosswalk. The specification reviewer initially returned REJECT and found missing dispositions for query independence, config caps, generator reporting, discretization, many-class oracle behavior and version provenance. Those findings prompted the additional crosswalk rows and a more concrete recovery proposal in the curie review.

The independent reader reproduced version 0.1.0 with exit 0 through the public Python entry point at pinned 807da066. They traced T1 configuration and command work to #21 and capability owners, T8 draft protocol status to #25/#41, T16 arithmetic versus learned comparison to #22/#32, and curie's global RNG restoration to r0's explicit-generator recovery requirement. Original source excerpts were used to challenge crosswalk completeness.

Both reviewers returned PASS for the bounded correction delta. The standards reviewer confirmed its two documentation findings were resolved. The specification reviewer confirmed every reported omission and the recovery explanation were corrected. Neither re-ran author validation or independently checked the recorded remote lookup. The original PENDING and REJECT above remain part of the review history.
