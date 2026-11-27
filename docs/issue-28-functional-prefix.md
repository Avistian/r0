# Issue 28: admitted ancestral prefixes (candidate)

`FunctionalPrefix` describes frozen ancestor nodes, a fresh population size, limits,
and any surrounding retained construction metadata. It needs no task definition,
feature readouts or target readouts. This provides the evaluation seam identified
in [sampled graph ownership research](research/0028-sampled-graph-ownership.md);
the complete sampled builder and raw-draw recorder are still unfinished.

`evaluate_functional_prefix(prefix, rng=..., device="cpu")` returns ordered tuples
of node states and readouts. The caller injects a keyed generator provider and is
responsible for assigning a fresh population stream to each receiving fit. Calling
this function repeatedly with an identically reset provider repeats the population;
a new receipt identifier alone does not make a population independent.

Complete task generation and prefix evaluation share the same internal node loop.
It evaluates real root laws, ordered parent state concatenation, frozen functions,
noise scales, pipelines, structural innovations and converter readouts. Descendants
consume final states; captured converter values remain distinct from those states.
No artificial observed/target channels or dummy nodes are inserted for prefix fits.

Before drawing arrays, prefix construction validates acyclic parent order, widths,
normalization moment certificates and conservative aggregate resource estimates.
The immutable record binds nodes, population size, retained metadata and limits.
Evaluation repeats admission before drawing. Limits include caller-supplied retained
metadata, but cannot discover external objects the caller omitted. Estimates are
not measured RSS or an OS cap. Existing complete-recipe admission uses the shared
arithmetic with its original task metadata and workspace charges.

The node record moved to a shared internal file; public imports and serialized
node/recipe shapes remain unchanged. Version-one task arithmetic and RNG order
must remain compatible. No full-source or independent acceptance is claimed yet.

## Reproduce focused checks

From this worktree, using the existing Python 3.12 environment without installation:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest \
  tests/priors/test_functional_prefix.py \
  tests/priors/test_functional_causal.py \
  tests/priors/test_functional_calibration.py \
  tests/priors/test_functional_admission.py
```

The new literal tests exercise a dimension-changing tree/quadratic prefix with
reordered multi-parent input, converter capture versus noisy descendant state, and
rejection of malformed ancestry/rows/resource caps. The complete-graph tests cover
the existing task path. The current local run passed 29 tests; its raw output is
`/tmp/r0-28-graph-prefix-green.log`, pending evidence packaging and exact-source
independent verification. No browser, Colab, GPU, throughput or scientific evidence
is established by this component.
