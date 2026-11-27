# Issue 28 local evidence retention

The user requires generated evidence to be gitignored. Generated execution logs,
receipts, saved examples, archives, review reports and screenshots are therefore
local files under `docs/evidence/`, not committed artifacts.

The first cleanup untracked 164 historical files already under that directory.
The subsequent cleanup moved another 883 evidence files (134,260,898 bytes) from
`docs/research/` to `docs/evidence/issue-28/legacy-research/`. Their bytes, including
original rejections and failed attempts, were preserved. A SHA256-verified mapping
and the exact migration procedure are retained in the ignored
`docs/evidence/issue-28/evidence-migration/` directory of the publication worktree.
Authored source research, all eight source-option inventories and test fixtures
remain tracked. No scientific or implementation requirement was removed.

Documentation links now point to the local ignored locations. Historical reports
and manifests keep their original bytes and may name their earlier locations or
publication revisions. Their old descriptions of committed/portable artifacts
refer to those historical revisions; the current checkout does not publish them.
Reproduction commands that consume those artifacts require the retained local
files. A fresh clone alone cannot reproduce such historical checks.

Code, authored contracts and concise verification outcomes remain versioned.
The cleanup does not turn missing hosted/GPU/resource evidence into acceptance,
and it does not erase any earlier rejection. No history rewrite was performed.
