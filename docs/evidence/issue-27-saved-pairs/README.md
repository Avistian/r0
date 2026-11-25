# Saved controlled comparisons

Source: `257739969be1fe9385ab2aa8c8cc2142e055d377`. CPU only, using the
existing r0 verification environment. No cloud run or hosted Colab interaction.
This evidence covers shared session actions and exported files, not the pending
side-by-side comparison panel or completion of #27.

## Observed behavior

The standalone walkthrough created ten pairs: five regression, two binary and
three multiclass cases. Every pair has 32 rows, 20 context rows and 12 query rows.
The ten task identifiers are 0–9, all with root seed `9223372036854775807`.
These are ten coupled source draws with twenty variants, not twenty independent
draws. The largest exported pair is 103,389 bytes; all ten total 462,015 bytes.

| Family | Task | Intervention | Variance policy |
| --- | --- | --- | --- |
| Linear | Regression | Active count 4 to 2 | Analytic unit variance |
| Grouped linear | Binary | Total columns 8 to 12 | Unchanged mechanism |
| Polynomial | Regression | Noise ratio 0.2 to 0.8 | Unchanged mechanism |
| Quadratic | Multiclass | Active count 4 to 2 | Explicit calibration estimate |
| Random Fourier features | Binary | Separation 1 to 0 | Unchanged mechanism |
| Periodic | Multiclass | Separation 1 to 0 | Unchanged mechanism |
| Smooth radial | Regression | Total columns 8 to 12 | Unchanged mechanism |
| Smooth sinusoidal | Regression | Noise ratio 0.2 to 0.8 | Unchanged mechanism |
| Exact GP | Multiclass | Total columns 8 to 12 | Unchanged mechanism |
| Composition | Regression | Active count 4 to 2 | Explicit calibration estimate |

A separate `verify` process opened each saved manifest, verified both runner
archives, recomputed the comparison card and regenerated each side directly from
its recipe. Inputs, targets, scores, splits, classification probabilities and
mechanism JSON replayed exactly. It checked exported recipes, CSV row counts,
query inputs, context targets, truth scores and truth targets against the saved
arrays. Query labels are absent from predictor exports. No guarded worker attempt
was added during verification; direct in-memory replay was intentional.

`created.json` records the cases and ZIP hashes; `verified.json` records verification
results. `pair-0.zip` through `pair-9.zip` retain the actual exported bundles, each
containing `card.json`, `before.zip` and `after.zip`. The separately readable
`card-*.json` files match their exported cards. `files-*.json` records paths, sizes
and SHA-256 hashes for all 212 original run artifacts, including archives, receipts,
manifests and attempt records. Retained dataset artifacts are byte-identical copies of the raw outputs under
`/tmp/r0-saved-pairs-final`. The check log only has trailing whitespace removed.

## Reproduction

Check out the source revision above. Choose a fresh unused output directory;
`create` refuses an existing destination. `verify` deliberately requires the same
HEAD as `create`, so evidence cannot silently be attributed to different code.
With the existing verification environment selected:

```sh
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_check.py python scripts/saved_pair_walkthrough.py create --out /tmp/r0-saved-pairs-reproduction
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_check.py python scripts/saved_pair_walkthrough.py verify --out /tmp/r0-saved-pairs-reproduction
```

Both final commands exited 0; their output is in `create.log` and `verify.log`.
The ten-family walkthrough is persistence/replay evidence, not a full family ×
task × intervention matrix. The previous controlled-pair API evidence covers 90
cases. Run measurements here were collected alongside checks and are not dedicated
throughput benchmarks or 100-million-task estimates.

The revision-guard counterfactual used a fresh directory containing only this
`created.json`:

```json
{"source":"0000000000000000000000000000000000000000","cases":[]}
```

Running the same `verify` command against that directory exited 1 before opening
any datasets or writing results. `revision-mismatch.log` retains the refusal.

## Checks, failures and limits

At the recorded source, `make check` passed: **466 passed, 1 CUDA skip in 134.73 s**.
New public-session tests exercise save/reopen/resume, exact large identifiers,
combined ZIP size limits, stale identities, retained timeout failures, altered cards,
corrupt archives, path containment, explicit estimated-variance permission and
clearing comparisons when the base changes. The runner helpers are shared with
ordinary dataset generation; existing session tests also pass.

Required commands ran in order: `make check`, `make smoke`, `make eval-smoke`.
The latter two exited 2 because this checkout has no `train` or `eval` CLI command.
Their logs are retained; no training loss summary or evaluation table exists.

The initial public-session tests failed on the absent `pair` and `pair_export`
actions. The first integration run exposed a missing `asdict` import, corrected
before final checks. The initial standalone walkthrough incorrectly requested
32 calibration rows for grouped linear, which requires at least 256. That run
failed visibly (`initial-create-failure.log`); the corrected script only reduces
GP calibration to 32. The initial full check also rejected Markdown fence formatting,
which was corrected. Historical failures are not counted as successful final runs.

Independent source review found no Spec defects. The Standards review caught the
verifier's revision attribution; the final script rejects a mismatch and the
reviewer confirmed closure. Neither review substitutes for these executed checks.

The comparison UI, actual Colab execution, additional CPU/GPU agreement, broad
sampling profiles and remaining resource/scale evidence are still pending.
