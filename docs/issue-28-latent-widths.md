# Latent width planning candidate

`sample_joint_widths` adds the named `node_n_latent_features` draw to the shared
joint sampling context. The default prior uses log interpolation on bounds 1
and 32, then floor with upper clipping to 31. Each node reserves its explicitly
supplied converter-channel count in addition to the sampled latent channels.
The `JointWidthPlan` binds every realized width to exactly one ordered named
decision and exposes the combined widths for subsequent mechanism construction.

This is a planning component for inventory option I2.graph.04. It does not yet
sample feature-node assignments or converter specifications, integrate those
choices into the complete topology archive, or deliver hidden-channel uncertainty
experiments. Those remain required, along with I2.graph.05–06 and the rest of #28.

## Source contract

The local pinned source `_node_function.py` uses the named latent-width request
before adding converter widths; `_base.py` defines logarithmic numerical
sampling and floor/clipping for `randint`. The implementation composes the
existing r0 named parameter sampler and transactional context; no source code
was copied. Default bounds follow the source. Explicit alternative bounds are
exploration controls, not additional paper defaults. Existing scalar-quantile
and numerical policies apply; upstream bitstream equality is not claimed.

Files inspected under `/tmp/node-pipeline-sampling-research/`:

- `_node_function.py`, SHA256
  `3b2345f23d0013442f7390c907454dfd4c6f4b43836a1bd210423242a34964c6`.
- `_dataset.py`, SHA256
  `7e4237e2f8c34b98000c147b81383b33711b381073d78212191d0761685ec1e1`.

Feature-node assignment in `_dataset.py` is deliberately still pending. This
component accepts converter channel reservations; it does not invent them from
the number of task features or confuse observed columns with latent widths.

## Admission and evidence

Invalid reservations and metadata budgets reject before draws. A sampled width
that exceeds node or total channel limits rejects inside the context transaction,
restoring caller RNG, decisions, preferences, and operation registry. There is
no retry, clipping of realized widths to resource budgets, or graph filtering.

Author evidence in `/tmp/r0-28-latent-widths`:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -q tests/priors/test_joint_width_plans.py
```

The initial missing-API failures are retained in
`/tmp/r0-28-latent-widths-red.log`. Four focused cases passed in
`/tmp/r0-28-latent-widths-oracle.log`: global named sharing, an independent scalar
local log-uniform oracle, and early/late budget rollback. Static checks passed
in `/tmp/r0-28-latent-widths-static.log`. Full-suite and independent review of this
width component remain pending. These checks do not measure runtime memory or
scientific diversity, and do not establish integrated delivery of I2.graph.04.
