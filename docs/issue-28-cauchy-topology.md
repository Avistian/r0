# Issue 28: Cauchy topology candidate

Status: standalone sampler reviewed; topology/mechanism integration is a candidate
described in [topology assembly](issue-28-topology-assembly.md). Workbench controls
remain pending. This does not complete topology breadth or issue #28.

## Source and numerical contract

The pinned TabICLv2 source draws one global Cauchy term plus an explicit offset,
receiver and source Cauchy propensities, then independent uniform edge proposals.
A proposal is accepted by a logistic probability; only edges from earlier to
later node indices survive. The entire square proposal matrix is drawn before
masking. Source variable names use `output_importances` for receivers and
`input_importances` for predecessors in that array orientation.
[Graph source](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_graph.py).

The cached same-revision configuration sets offset to zero and node bounds to
2–32. This API requires an explicit offset and admits configurable node counts;
these are exploration controls, not a claim to reproduce the full prior.
[Configuration](https://github.com/soda-inria/tabicl/blob/9530bd65a7b806f6ad4f9f4335af745577e9d3cc/src/tabicl/prior/graph_lib/_config.py).

Implementation is a NumPy float64 adaptation. It uses NumPy Cauchy and uniform
draws through the raw ledger rather than the source's mixture of global NumPy
and Torch generators. It does not claim bitstream identity with upstream Torch.
Stable scalar sigmoid evaluation avoids exponential overflow without clipping
logits. Nonfinite connectivity arithmetic rejects the draw; no retry conditions
on connectivity, usefulness or target predictability.

Local source copies read:

- `/tmp/r0-28-source-read/I2graph.py`, SHA256
  `36e9488d481d562387d11ef655510ffb1ad0a28c2d3bb49b48c1edba2f4acabc`.
- `/tmp/node-pipeline-sampling-research/_config.py`, SHA256
  `789f5af746f7a96e3efd2433652c9a2fcdc2bbfe62988b391d3b8572be9baf01`.

## API and retained evidence

`CauchyGraphPrior(nodes, offset)` and `sample_cauchy_graph(prior, rng=..., limits=...)`
produce a `CauchyGraph`. Its four raw records retain the global term, receiver
propensities, source propensities and complete square proposal matrix. Derived
parents are sorted distinct earlier-node indices. Uniform entries discarded by
the acyclic mask remain recorded and participate in replay.

`validate()` binds record laws, roles, shapes, generator chain and derived parents.
It does not prove raw values came from an original RNG. `replay(rng)` requires the
matching caller pre-state, executes all retained laws and checks values/states,
then publishes the caller state only on success. No seed is reconstructed from
a digest. The five supported concrete NumPy engines remain supported.

Geometry and conservative retained/working storage estimates are checked before
square arrays. Estimates are not measured RSS. The raw ledger's additional
limits remain enforced. Invalid parent types, nonuniform proposal values,
inconsistent bindings and nonfinite arithmetic are rejected. Valid highly dense
and disconnected draws are retained.

## Author verification

- Public API RED: `/tmp/r0-28-cauchy-topology-red.log`, one missing-API failure.
- First scalar equation and generator-tail oracle: one pass in 0.97 seconds,
  `/tmp/r0-28-cauchy-topology-green.log`.
- Replay/validation RED: six failures and two passes,
  `/tmp/r0-28-cauchy-topology-replay-red.log`.
- Repaired topology/raw-capture compatibility: 39 passes in 1.17 seconds,
  `/tmp/r0-28-cauchy-topology-compatibility.log`.
- Static gates: `/tmp/r0-28-cauchy-topology-static-second.log`, passed.

Reproduce using the existing environment:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest -ra tests/priors/test_cauchy_topology.py \
  tests/priors/test_raw_draw_capture.py tests/priors/test_raw_draw_streams.py
```

At standalone checkpoint `557bec83aeb66ebb0fbed07634441bb532ad825b`,
independent scoped standards and specification reviews passed. Its full check
passed with 1,749 tests and one skip in 618.99 seconds
(`/tmp/r0-28-cauchy-check.log`). Resource measurement and portable evidence
publication remain pending. Source options I2.graph.01–03 have this standalone
implementation; final integration review remains pending. I2.graph.04–06 (latent widths, partial observations and hidden
channel uncertainty) are not delivered by this topology sampler. Motifs,
rewrites, preferential attachment and other required graph families remain in
scope. No family-level substitute closes the option-level inventory.
