# Analytic response curves: local browser evidence

Application and browser procedure source:
`f6e089aa21f0d76e209ef4d7ef3cd020516c172b`. Retained receipt and screenshots are from
a fresh local server started at that source, with fresh output in
`/tmp/r0-response-curves-f6e089a`. Raw browser outputs were retained in
`/tmp/r0-curves-browser-f6e089a`; `files.json` lists their hashes and sizes. The two
committed PNGs were visually inspected. These are Chromium results, **not hosted
Colab evidence**. Hosted interaction remains **NOT_RUN**.

Before implementation, the public-API test failed on missing `inspect_curve` and
the browser procedure against the prior UI failed with `response-curve controls
must exist` (no such control). After implementation, `make check` passed with
442 passed and one CUDA skip. The curve tests use an independent matrix calculation
from recorded linear weights, check the constant irrelevant-feature response,
preserve original task values, and reject unsupported inputs and direct GP curves.

The actual browser procedure generated all ten starting recipes. It evaluated
129-point curves for the nine analytic families and checked the explicit GP
conditional-inspection error without drawing a fake curve. It also exercised a
65-point composition curve anchored to row 63, clearing stale curves when inputs
change, constant response on an irrelevant feature, and actual SVG/PNG downloads.
Existing generation/replay/export, class counts, maximum shape and mobile checks
also passed. See `receipt.json` for identities, environment and run measurements.

Reproduce from the source above using a fresh output and evidence directory. In
separate terminals, with Chromium already available:

```bash
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/explore.py \
  --out /tmp/r0-response-curves-f6e089a --port 8768

chromium --headless --remote-debugging-port=9224 \
  --user-data-dir=/tmp/r0-response-curves-profile about:blank

node scripts/explorer_browser_smoke.mjs 9224 http://127.0.0.1:8768/ \
  /tmp/r0-curves-browser-f6e089a
```

The recorded run used Node v24.21.0 and the installed Chromium headless shell
`/home/avist/.cache/ms-playwright/chromium_headless_shell-1243/chrome-headless-shell-linux-arm64/chrome-headless-shell`.
The receipt records the exact browser version. No npm package is required.

After `make check`, `make smoke` and `make eval-smoke` each exited 2 because the
repository CLI does not implement `train` or `eval`; no loss summary or evaluation
table exists. Generation graphs, conditional GP views and pairs remain pending.
