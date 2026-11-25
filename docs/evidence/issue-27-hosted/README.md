# Modal CPU/GPU smoke and durable resume

Both runs used source `116c4977156cde96daed3f7b6bc14c29ae51b712` and factory workflow
`94ccc691b6c09228767f59b86e96e06290ccb6ff`. They were explicitly authorized with a
$5 aggregate budget and without a dashboard cap for this smoke. No unattended
larger run was authorized or launched. The compute switch was disabled afterward.

| Evidence | CPU | T4 |
| --- | --- | --- |
| Workflow | [45588566360](https://github.com/Avistian/r0/actions/runs/45588566360) | [45589037046](https://github.com/Avistian/r0/actions/runs/45589037046) |
| Newly generated / saved / failed | 8 / 8 / 0 | 8 / 8 / 0 |
| Resume: new / verified saved | 0 / 8 | 0 / 8 |
| Generation + serialization + persistence | 2.094 s | 2.062 s |
| Child elapsed including startup | 5.958 s | 4.945 s |
| Child peak RSS | 3,125,891,072 bytes | 3,724,341,248 bytes |
| PyTorch GPU peak allocated | Not applicable | 17,058,816 bytes |

Each saved workload has 128 rows, eight features, four active features and 96 context
rows. The run cap was eight tasks, 60 seconds and 10 MiB, with two workers inside a
one-core/4-GiB container. T4 additionally performed independent CPU/GPU execution of
64-row regression, binary and three-class diagnostic recipes. Identity, mechanism,
inputs, scores, targets, split and probabilities agreed at `rtol=atol=1e-10`.
The reported worker timing/memory excludes that parent-process diagnostic.

The second invocation used a separate single-use container. It reloaded committed
Volume output, verified group digests/member integrity and reported no new tasks.
This establishes the tested save/commit/reload/resume path, not crash-injection
recovery on Modal or distributed writer safety. Local failure/retry tests remain
separate evidence. No wider shape/family GPU agreement claim is made.

## Retained records

`cpu-*` and `gpu-*` are the downloaded workflow artifact files without edits.
`artifact-metadata.json` identifies their GitHub artifact IDs/digests and source
revision. `SHA256SUMS.json` records hashes of the local retained JSON files.
Raw logs remain attached to the linked workflows. Re-download the originals with:

```sh
gh run download 45588566360 --repo Avistian/r0 \
  --name generation-45588566360 --dir /tmp/issue27-cpu-receipt
gh run download 45589037046 --repo Avistian/r0 \
  --name generation-45589037046 --dir /tmp/issue27-gpu-receipt
```

Compare `reservation.json`, `local-smoke.json` and `result.json` with their respective
`cpu-` / `gpu-` copies. In each result, inspect the source SHA, generation counts,
resume counts, return codes, timeouts and requested limits. The T4 result retains
the agreement outcome and tolerance. Archived task arrays remain in Volume
`r0-generation-v1`, under `runs/issue27-{cpu,gpu}-smoke-20261125`.

## Rejections and limits

Two earlier dispatches failed before jobs were created; their complete selected
run metadata, including empty job arrays, is retained in `rejected-dispatches.json`.
They never reached the credential-bearing step. A subsequent API request using a
JSON number for `workers` was rejected with HTTP 422 and created no workflow run.
The accepted source explicitly converts the dispatched worker count to a number
for the reusable workflow. Both hosted runs then succeeded. These rejected attempts
are not erased or counted as successful CPU/GPU execution.

The successful CPU and T4 plans each reserved $1.25. Actual provider charges remain
**NOT_MEASURED**; reservation and resource estimates are not settled billing.
The tiny tasks spend substantial time on serialization and Volume persistence;
their roughly 3.8 tasks/s rates are not bulk-generator throughput or a reliable
100M-task estimate. Colab installation/Drive/browser interaction, larger workloads,
additional generator families and the explorer remain unverified or unfinished.

Local full checks passed 241 tests with one CUDA skip. Hosted CI also passed at
the executed source. The hosted GPU agreement above is distinct from that local skip.
`make smoke` / `make eval-smoke` still lack their train/eval CLI commands; these
generation runs supply neither a training loss nor an evaluation table.
