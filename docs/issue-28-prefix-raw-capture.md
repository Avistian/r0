# Issue 28: replayable fresh ancestor populations

`sample_functional_prefix(prefix, rng=..., owner=(...), capture=...)` samples an
admitted frozen prefix from one injected calibration generator. Roots and structural
innovations consume it in node, column and draw order. Every receiving fit can call
it afresh; successful calls advance the stream, while any failure discards the whole
population's draws and records. Caller mappings and RNG states retain the raw ledger's
transaction contract. The existing keyed `evaluate_functional_prefix` and authored
generation/inspection APIs keep their behavior.

Capture records original root base values and ball radii; Gaussian, uniform and
Student-t innovations; mixture normals and component choices; and original directions
and radii for ball innovations before covariance factors. Truncation records every
proposal batch, including rejected proposals and unused accepted tails. It does not
mislabel post-truncation values as unconstrained draws. Laws, parameters, shapes and
states use the existing seven-law raw recorder. No retry policy changes.

Owners extend the caller's receiving-fit path with node/root/innovation/column
components. These paths identify draw sites; complete typed graph ownership and
plan-to-runtime bindings remain separate unfinished work. A nested graph transaction
must bind this generator as `calibration`. The prefix is readmitted before allocation;
raw-record limits additionally bound retained draws. Complete graph-wide aggregate
admission remains unfinished. Sampling on CUDA is explicit, without fallback;
GPU execution and measured resource use remain unverified.

Author checks include missing-API RED, five-engine capture/replay across all four
independent innovation laws with/without truncation and ball geometry, covariance
factors, original root ball radii, ordered parent states and literal mixture equations.
Successive calls produce distinct populations; late draw-budget and nonfinite pipeline
failures roll back the generator and ledger. Original keyed evaluation checks remain.
The initial expanded command named two nonexistent test files and ran no tests; its
log is retained separately from the corrected successful command. Full-suite and
independent acceptance are pending at this source commit.

This progresses [the full #28 specification](specs/0028-causal-network-generators.md)
and [sampled graph ownership design](research/0028-sampled-graph-ownership.md).
The graph builder still must request a fresh population for each receiving function
and pipeline fit, retain its receipts and bind it to frozen ancestors. This component
alone does not prove that graph schedule, browser/Colab delivery, broader paper options
or scientific performance.

## Exact author reproduction procedure

Inputs are `tests/priors/test_prefix_raw_capture.py`, `test_functional_prefix.py`,
`test_causal_root_input.py`, `test_causal_exogenous.py` and `test_functional_causal.py`
at source `4f80f9cab590cacf88f1bc463ca939a6ccccfc16`. Using the existing environment
Python 3.12.3, NumPy 2.5.5 and Torch 2.15.0+cpu, the successful command was:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -ra \
  tests/priors/test_prefix_raw_capture.py \
  tests/priors/test_functional_prefix.py \
  tests/priors/test_causal_root_input.py \
  tests/priors/test_causal_exogenous.py \
  tests/priors/test_functional_causal.py
```

It passed **87 tests in 1.74 seconds**. The exact stdout/stderr is retained in
`/tmp/r0-28-prefix-raw-expanded-green.log`; the earlier nonexistent-file invocation
is `/tmp/r0-28-prefix-raw-expanded.log` and ran no tests. The missing-API RED is
`/tmp/r0-28-prefix-raw-red.log`; its intermediate uncommitted source is not retained,
so that historical failing run is not presented as a fresh reproducible source build.
The full check at the same source used the two environment variables above and
`make check`: **1496 passed, one skipped, 532.81 seconds**; exact log
`/tmp/r0-28-prefix-raw-full-check.log`. All static gates passed. Source and tests at
integration `1b547ec0433d9fa7a6e623668dbd7a49bd22d328` are byte-identical.

Independent original-source archives, inputs and executable procedures are retained
at `/tmp/r0-28-prefix-raw-spec-4f80f9c`. Its `README.md` names each oracle and the
checks it supports; its `environment.json` and `SHA256SUMS` pin the environment and
exact bytes. To reproduce the successful self-contained replay, use new output outside the
packet and the interpreter above:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/avist/Projects/r0/.venv/bin/python \
  /tmp/r0-28-prefix-raw-spec-4f80f9c/replay.py \
  /tmp/r0-prefix-capture-replay
```

The script resolves source archives and oracle programs relative to itself, extracts
fresh sources and explicitly sets child `PYTHONPATH`; no original checkout is used.
The [published archive](evidence/issue-28/legacy-research/0028-prefix-raw-capture-spec-artifacts.json) retains
both the original packet and exact-main independent verification, including all
original executed programs, source archive identities, raw populations and commands.
[Author logs](evidence/issue-28/legacy-research/0028-prefix-raw-capture-development.json) preserve the earlier
failures and successful runs verbatim. Original temporary paths above identify the
executed procedures; use the recovery command below for a separate checkout.

```sh
python3 scripts/recover_joint_sampling.py \
  docs/evidence/issue-28/legacy-research/0028-prefix-raw-capture-spec-artifacts.json \
  /tmp/r0-prefix-capture-evidence --repo .
PYTHONDONTWRITEBYTECODE=1 /home/avist/Projects/r0/.venv/bin/python \
  /tmp/r0-prefix-capture-evidence/publication-independent/replay.py \
  /tmp/r0-prefix-capture-replay
```

The [candidate](evidence/issue-28/legacy-research/0028-prefix-raw-capture-candidate-review.md) and
[exact-main](evidence/issue-28/legacy-research/0028-prefix-raw-capture-publication-review.md) reviews verify
180 configurations and 360 populations, 346 proposal batches, 77056 rejected
proposals and 9280 unused accepted values. All original arrays/states and literal
prefix equations agree. Across 132 compatibility objects, authored causal-v1/v2,
functional generation, fixtures, keyed prefixes and inspection remain byte-identical.
The [full check](evidence/issue-28/legacy-research/0028-prefix-raw-capture-check-4f80f9c.txt) is parent evidence.
The [original Standards finding](evidence/issue-28/legacy-research/0028-prefix-raw-capture-standards-original.txt)
about missing reproduction instructions remains retained; the instructions above
address it. The duplicate device/admission guard heuristic remains nonblocking.

The evidence pack stores identical files once and restores duplicate paths. Large
whole-repository source archives are reconstructed byte-for-byte with `git archive`
from pinned commits, then checked against the original SHA256SUMS. Original candidate
history is published on `evidence/issue-28-raw-capture`; a single-branch clone must
fetch that branch before recovery. This avoids recursively republishing older evidence
archives. The same method is used for the function/root capture packet.
