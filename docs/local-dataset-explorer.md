# Explore and create datasets locally

From an installed r0 checkout, run:

```bash
python scripts/explore.py --out outputs/explorer
```

Open the printed `http://127.0.0.1:8765/` address. For a deliberately reused
verification environment, prefix the command with `python3 scripts/run_check.py`
and set `UV_PROJECT_ENVIRONMENT` as described in the repository README.

Select a mechanism, edit rows/features/task type, and click **Generate dataset**.
The advanced section exposes the complete versioned recipe, device, time limit
and output-storage limit. Ten starting recipes cover linear, grouped linear,
polynomial, quadratic, random Fourier features, periodic, smooth radial,
smooth sinusoidal, exact GP and composed functions. These are editable examples,
not a production mixture or bulk-training eligibility claim. Input distributions,
kernels, axes, component strengths and other family-specific controls are editable
in the recipe JSON. Recipe validation rejects incompatible settings explicitly.

Every click uses the shared guarded generation runner for one task, saves and
verifies its archive, then loads the saved arrays into the explorer. CPU and CUDA
are explicit choices; unavailable CUDA fails without CPU fallback. The output
folder retains the run request, attempts, measurements, archive and receipt.
The UI reports the archive's path and task identifier so it can be reopened later.
Opening a saved task never invokes generation. Failed generation leaves the
previous displayed task intact. A parent-process request record is saved before
launch, and a separate result record preserves timeout, unavailable-device and
worker failures even when the child never initializes. These UI audit records
live under `OUTPUT/.attempts/`, outside the runner's output-storage limit; each
attempt keeps its own records. A started record without a result is incomplete.

**Replay recipe** accepts the full exported `recipe.json`; **Next task** increments
the task identifier. Exact seeds, task identifiers and split identities support
the generator's 63-bit range. The UI transports large identities as decimal
strings to avoid JavaScript rounding; raw exported recipe JSON preserves integer
values. In the advanced recipe editor, write split identifiers larger than
`9007199254740991` as quoted decimal strings. The pasted replay record needs no
such conversion. Editing recipe or identity creates a different request. Repeating
the same request resumes verified saved output and reports zero new generations.

The interface shows:

- Full-row feature/target histograms, active/irrelevant feature roles and split counts.
- Up to 512 reproducibly sampled rows in signal/probability scatterplots, with
  context/query colors and clickable points. These are observed points with other
  features varying, explicitly not response curves.
- The first 64 sampled rows in the table preview, with arbitrary full-task row lookup.
- [Analytic response curves](response-curves.md) with other inputs fixed to a chosen
  row, bounded evaluation points and inspectable values.
- Regression latent signal versus observed target and realized noise deviation;
  classification probabilities and observed/context/query/expected counts for
  every declared class, including absent classes.
- Source/episode identity, recipe, sampled parameters, actual device/software,
  elapsed time, peak RSS, GPU allocated memory and runner counts when generated.

Download the complete dataset as a [replay bundle](dataset-inspection.md), independent
of chart sampling. Predictor files exclude query labels and mechanism truth.
Each chart exports as SVG or PNG; the recipe also downloads separately. Output
limits bound persisted bytes, not process RAM. Dataset exports are built in memory.
The browser serves only on loopback, requires a per-process token for actions and
does not contact cloud services. One server session has one active dataset and
processes dataset actions sequentially while accepting browser connections independently.
Ctrl-C stops accepting connections and waits for an active bounded generation job
to finish, so its child process is not abandoned. Saved datasets remain on disk.

This slice supplies the local creation/inspection loop. Expandable generation
graphs, reconstructed component values, conditional GP views and paired
comparisons remain required #27 follow-up slices.
The [embedded Colab adapter](colab-explorer.md) shares this UI; its hosted proof
is tracked separately.

## Reproducing browser checks

Start the explorer with an empty output directory and Chromium with a separate
profile and a loopback debugging port:

```bash
python scripts/explore.py --out /tmp/r0-browser-run --port 8767
chromium --headless --remote-debugging-port=9224 \
  --user-data-dir=/tmp/r0-browser-profile about:blank
node scripts/explorer_browser_smoke.mjs 9224 http://127.0.0.1:8767/ /tmp/r0-browser-proof
```

The retained script uses Node's built-in WebSocket and Chromium's debugging API;
no npm packages are needed. It checks actual browser generation for all ten starting
recipes, row lookup, zero-new-generation replay, reopening, ZIP/SVG/PNG downloads,
classification counts, 63-bit identities, full versus sampled rows, visible failure
with preserved prior data, and mobile overflow. It writes screenshots and a receipt.
Browser checks are separate from Python tests and hosted Colab/GPU acceptance.
