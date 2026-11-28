# Converter input channel allocation candidate

`allocate_observation_channels` binds each assigned feature to a contiguous,
half-open input span on its node. It traverses explicit group order and feature
order, reserving the caller-declared converter input width. These widths are
inputs consumed by converters, not output category counts or graph latent widths.

`ObservationChannelLayout` retains the observation realization, declared input
widths, spans, and per-node converter-channel totals. Validation checks the
observation decisions/raw bindings and derives the exact layout again. Duplicate
node assignments occupy separate consecutive spans. No latent channel is marked
observed by this allocation.

The totals feed `sample_joint_widths`, which appends the independently sampled
latent widths. Admission reserves at least one latent channel per node, rejects
oversized converter reservations, and bounds layout metadata before allocating
its lists. The eventual width sampler checks the actual added latent channels.
Neither component clips or retries over-budget geometry.

## Evidence and remaining integration

The public integration test builds a dense three-node graph from numerical
feature assignments, explicit identity converters, and one hidden channel per
node. Generation and repeated-row inspection pass. The final node's mechanism
receives all preceding node channels, including hidden channels. A separate test
covers heterogeneous converter input widths, tampered totals and budget rejection.

Reproduce in `/tmp/r0-28-observation-channels`:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_observation_channel_layout.py \
  tests/priors/test_joint_width_plans.py tests/priors/test_joint_observation_plans.py
```

Logs are under ignored `docs/evidence/issue-28-geometry/`: missing-API RED in
`channel-layout-red.log`, 17 passing focused cases in
`channel-layout-integration.log`, and static verification in
`channel-layout-static-final.log`. Earlier geometry logs were copied unchanged
into this directory to satisfy README.md's evidence-location rule. The original
standards finding remains in `r0-28-geometry-standards-a22eb63.txt`.

This is still a construction component. The integration test passes realized
geometry into the existing topology request; that archive does not yet retain
the observation and latent-width sampling provenance. The next complete
construction version must retain those records and their stream boundaries,
keep the shared named context across geometry and mechanisms, and expose them
through archive reopening, inspection, comparisons and the workbench. Native
categorical converters and sampled converter/pipeline calibration also remain
unfinished. No I2.graph.04–06 completion or whole-issue acceptance is claimed.
