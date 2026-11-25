# Side-by-side dataset comparisons in the explorer

Application source: `4e551eb960e3da3858252be64dbefe29c6813492`.
The final regression walkthrough uses harness commit
`e11e3a1e837304558cdbe9cd3b8ae1f53ad805ea`; that delta only consumes an HTTP response
body in the browser harness. Runtime assets and backend are identical. The Colab
notebook pins `e11e3a1`, including the shared comparison panel. Hosted Colab remains
**NOT_RUN**; no cloud run was launched for this evidence.

## Browser observations

The dedicated walkthrough used actual Chromium through CDP, with CPU generation
on loopback port 8777. It checked all four interventions, common axes and matching
original row coordinates, classification channels/counts, preserving the base,
timeout preservation of the previous
comparison, clearing on a new dataset, explicit estimated-variance refusal/consent,
reopening a saved pair and a 390-pixel mobile viewport without horizontal overflow.
Keyboard row inspection used CDP Enter events on a focused point.

Chromium completed SVG, PNG, JSON and ZIP downloads; the exact card retained the
seed `9223372036854775807`. The retained 91,394-byte `comparison.zip` contains both
128-row datasets and their card. Independent NumPy inspection found identical
common inputs and latent scores and a fourfold residual change, maximum scaling
error **4.440892098500626e-16**. Predictor arrays exclude query targets.

- `noise.png`, `classification.png`, `mobile.png`: actual browser screenshots.
- `pair-after-plot.png`: the downloaded chart, with applied control, row count and
  split/axis legend inside the image.
- `receipt.json`, `pair-files.json`: browser observations and hashes/sizes of all ten
  original pair-browser artifacts. Larger SVG remains in the raw output; its
  completed download and labels were checked by the harness.
- `comparison-card.json`, `comparison.zip`, `download-verification.json`: exact
  downloaded data and independent array-check result.
- `full-browser-receipt.json`, `full-files.json`: the broader regression walkthrough,
  including all ten families, ten generation graphs, nine analytic response curves,
  GP conditional views, expandable 64-component composition, a 20,000 × 256 task,
  replay, saved opening, exports and responsive layout.

Retained binary and JSON artifacts are byte-identical copies; `comparison.zip`
renames the original identity-bearing filename. Logs only strip trailing whitespace.
Raw outputs are `/tmp/r0-pair-browser-4e551eb` and
`/tmp/r0-explorer-browser-pairs-final`. Both browser exception lists are empty.

## Reproduce

Use source `e11e3a1` and the existing verification environment. Start a fresh explorer
output directory and an owned Chromium CDP instance. The recorded environment used
Node v24.21.0 and the Chromium version in the receipts; no npm packages are needed.

```sh
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts/run_check.py python scripts/explore.py --out /tmp/r0-pair-ui-reproduction --port 8777
```

For a fresh reproduction, start the recorded Chromium executable with a new profile:

```sh
/home/avist/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux-arm64/chrome-headless-shell --no-sandbox --disable-dev-shm-usage --remote-debugging-port=9226 --user-data-dir=/tmp/r0-pair-browser-profile about:blank
```

In another terminal, run the browser procedures sequentially against that server:

```sh
node scripts/pair_explorer_browser_smoke.mjs 9226 http://127.0.0.1:8777/ /tmp/r0-pair-browser-reproduction
node scripts/explorer_browser_smoke.mjs 9226 http://127.0.0.1:8777/ /tmp/r0-full-browser-reproduction
```

Choose fresh evidence paths and available ports. If changing any served asset,
restart the explorer: its HTML is assembled at server startup. The final server
was started after the feature-label and keyboard-focus changes.

Run the independent downloaded-array verification from this evidence checkout:

```sh
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv python3 scripts/run_check.py python docs/evidence/issue-27-pair-explorer/verify_download.py
```

## Checks and historical failures

`make check` at application source `4e551eb` passed: **466 passed, 1 CUDA skip in
131.95 s**. `make smoke` and `make eval-smoke` followed in order and exited 2 because
`train` and `eval` CLI commands are unavailable. Logs are retained. The later harness
change passed Node syntax checking and the complete browser regression procedure;
no application code changed, so the Python suite was not duplicated.

`browser-red.log` records the initial real-browser failure because comparison
controls did not exist. The first implemented pair walkthrough passed; the final
one additionally exercised active-feature defaults/roles and actual CDP key events.

The first broader walkthrough failed in Node's Undici HTTP parser before reaching
the UI (`http-harness-failure.log`). Consuming the fetched body before closing the
idle connection allowed the full browser procedure to pass.
This was a harness correction; the failed run is not counted as a completed UI run.

The Spec review found no scoped defects. Standards review found no hard violation
and one optional suggestion to consolidate intervention metadata; the current small
maps and direct conditions remain. Both reviewers confirmed the harness-only delta.

This is local browser and CPU evidence. Shared assets in the notebook do not prove
hosted Colab interaction, Drive persistence or GPU agreement. Broader exploration
profiles, remaining numerical fixes and resource/scale evidence remain separate
#27 work. This PR does not claim complete issue acceptance or trained-model results.
