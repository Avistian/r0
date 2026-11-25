# Estimate a larger Modal generation request

Before proposing a larger run, download the retained result of a successful fresh
Modal smoke and estimate the proposed request without launching compute:

```bash
gh run download 45697210266 --repo Avistian/r0 \
  --name generation-45697210266 --dir /tmp/r0-estimate-receipts
python3 scripts/run_check.py python scripts/estimate_modal_generation.py \
  --result /tmp/r0-estimate-receipts/result.json \
  --config configs/generation/sampled-gpu.toml
```

For a larger proposal, copy the TOML request and edit its count, generation-time
limit and output-storage limit, then pass that path to `--config`. The command
prints the measured task count/rate, baseline revision and hardware, worker count,
proposed limits, and `requested count / measured rate`. It makes no writes and
needs neither CUDA nor Modal credentials.

The measurement must come from a complete fresh run: all requested tasks generated
and saved, with no failures or pending tasks. A resume with zero new generations,
a partial run, inconsistent rate, or mismatching source/device is rejected. Recipe,
profile, seed, start, grouping and device must match the measurement; only count,
time and output limits may change. Preserve the printed worker count. Changed
hardware, generator code or workload needs a new authorized smoke before relying
on its rate; the estimator reports these assumptions and does not inspect a
future provider environment.

The estimate treats every requested task as new. Existing verified groups can
reduce actual generation work on resume, but are not inspected by this command.
Time excludes startup, CPU/CUDA agreement checks, separate-container resume
verification and retries. Sampled mechanisms vary between tasks, so eight measured
tasks do not establish the whole profile's capacity or tail latency. Peak measured
memory is reported without extrapolating it as a bound. Output bytes are
`NOT_MEASURED`: the hosted summary does not include persisted archive sizes.

`fits_generation_time_limit` compares the projection with the proposed generation
deadline only. It is not an admission verdict. Before dispatch, the existing
[guarded lane](modal-device-agreement.md) must separately validate count, storage,
expanded workflow deadlines and the remaining authorized reservation. Current
repository limits still allow only eight generated tasks per run. The estimate
command does not raise these limits or authorize another paid run.

The Colab notebook already shows this measured-rate estimate before its disabled
larger-run cell. Both paths use the shared generator's measurement; neither treats
extrapolated 100-million-task time as measured execution.
