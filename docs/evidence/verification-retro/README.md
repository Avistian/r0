# Verification-environment and artifact-check regression evidence

User-approved follow-up to the issue 24 session retrospective. This standalone change
is based on main `daf4b05f3d509c3cf0ea215b59ca74368c252fdc`, not the challenge PR stack.
Implementation tested at `8e6b6f3`.

## Environment preservation

The [RED run](environment-red.log) uses real uv and two temporary local projects, with
an editable installation in a borrowed virtualenv. `make preflight` originally returned
success but replaced both r0.pth and direct_url.json with the borrowing checkout's paths.

The regression now passes: `scripts/run_check.py` runs uv offline with no syncing,
selects this checkout's source, and compares editable metadata before and after each
command. Additional tests show a changed .pth fails verification and an inherited active
environment does not replace a missing checkout-owned .venv. No real user environment
is modified by the regression fixtures. Fixture package builds use the existing offline
uv backend cache populated by repository setup; no dependencies or data are downloaded.

## Token-check parity

The [RED run](tokens-red.log) stages an oversized JSON artifact whose filename contains
spaces. The original Make target succeeds because its extension filter omits JSON.

Make, CI's make check, and the actual commit-hook script now use the same indexed-blob
checker. Both local command paths reject that JSON. Replacing or deleting the working
copy cannot hide an oversized staged blob. The exact 20,000-byte text boundary passes;
existing binary and lockfile exclusions are preserved.

## Validation

`make check` at the implementation head passed: 143 tests and three import contracts.
The [check log](check.log) includes checkout preflight, formatting, lint, types, imports,
module coverage, token/artifact checks and tests. Logs retain output with trailing
whitespace removed. These are local results, not hosted CI evidence.

The changes add no dependencies, do not modify protected evaluation paths, and run no
training/cloud jobs. Training and evaluation smoke are NOT_RUN because those CLI
capabilities are still unimplemented. No loss summary or evaluation table is claimed.

The environment guard covers editable-install metadata, not arbitrary writes or network
access by tested code. Its no-sync/offline flags constrain uv. Missing environments or
dependencies require explicit setup. Direct bare uv commands remain outside this guard.

## Independent review

At `8e6b6f3`, specification review found no actionable defects and independently reran all
five regression tests successfully. Standards review found no documented violation; it
noted nonblocking duplication of index-blob discovery between the token and 1 MiB artifact
checkers. That maintenance suggestion is retained without expanding this fix into a refactor.
