# Integrated generator verification

On 2026-11-25, the remaining #27 branches were assembled in a separate verification
checkout at [`147909568da719eac414924447c7bc9c0292421f`](https://github.com/Avistian/r0/tree/147909568da719eac414924447c7bc9c0292421f).
This combines PRs #115–123 on main `b703a7b` without merging those PRs on GitHub.
The later one-sentence correction to the local explorer documentation is outside
this candidate; runtime code is unchanged by that correction.

| Check | Result |
| --- | --- |
| `make check` | 492 passed, one skipped; formatting, lint, types, boundaries and repository guards passed |
| General explorer browser walkthrough | Passed all ten families, graphs and component expansion, response curves, saved reopen/replay, classification, full exports and 20,000-row/256-feature display |
| Sampling-profile browser walkthrough | Passed broad/reference/fixed modes, selected settings, next-task variation, exact replay and recipe download |
| GP browser walkthrough | Passed conditional means/bands/draws, explicit approximation labels, deterministic replay and export |
| Paired browser walkthrough | Passed all four interventions, saved reopen, matched axes, strict/estimated variance distinction and ZIP/JSON/SVG/PNG downloads |
| Numerical graph browser walkthrough | Passed centered Fourier values, weak-signal curve, inactive polynomial basis and graph reconstruction |
| Browser errors | Zero reported exceptions across all five walkthroughs; mobile overflow checks passed where present |
| Notebook cells, executed locally | Eight sampled polynomial tasks created and verified; resume generated zero replacements; larger run stayed disabled |
| `make smoke` then `make eval-smoke` | Both exit 2: `train` and `eval` CLI commands are absent. No loss summary or evaluation table exists; these commands belong to their separate implementation tickets |
| Actual hosted Colab | NOT_RUN: local notebook execution does not verify Colab output callbacks, downloads, Drive persistence or runtime restart |

The full check and browser runs used the shared existing environment through
`scripts/run_check.py`, with numerical libraries restricted to one thread. Browser
walkthroughs drove actual Chromium over its local debugging interface; they did
not substitute HTTP-only tests. Screenshots, downloads, receipts and logs remain
ignored under `docs/evidence/issue-27-integrated/` in the verification checkout.
The recorded run is author execution, distinct from independent source review and
the separate [Modal receipt record](https://github.com/Avistian/r0/blob/eead75abafd308212fb17826304a3e1a94ccd82e/docs/modal-device-results.md).

## Reproduction

Use the source above and an installed environment. Start the explorer with a fresh
output directory and a dedicated Chromium debugging profile as described in
[local explorer checks](local-dataset-explorer.md#reproducing-browser-checks).
Then run these scripts sequentially against that browser and server:

```bash
node scripts/profile_explorer_browser_smoke.mjs 9226 http://127.0.0.1:8781/ /tmp/profile-proof
node scripts/explorer_browser_smoke.mjs 9226 http://127.0.0.1:8781/ /tmp/explorer-proof
node scripts/gp_explorer_browser_smoke.mjs 9226 http://127.0.0.1:8781/ /tmp/gp-proof
node scripts/pair_explorer_browser_smoke.mjs 9226 http://127.0.0.1:8781/ /tmp/pair-proof
node scripts/numerical_explorer_browser_smoke.mjs 9226 http://127.0.0.1:8781/ /tmp/numerical-proof
python3 scripts/run_check.py python scripts/colab_notebook_walkthrough.py \
  --out /tmp/notebook-proof --family Polynomial --profile broad-v1
```

Use new evidence directories. The notebook walkthrough checks that runtime source
matches the notebook's installation pin before executing its generation/estimate
cells; it does not execute the hosted install or Drive mount. The three repository
commands were run in the required order: `make check`, `make smoke`, `make eval-smoke`.

This closes the local combined-branch verification gap. It does not turn the
candidate into merged main, certify a production sampling mixture, or close #27
without its actual embedded Colab acceptance. Raw evidence remains out of Git.
