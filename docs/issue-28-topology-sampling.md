# Issue #28 sampled topology checkpoint

This extends the explicit affine core. It is **partial implementation** of the
[accepted specification](specs/0028-causal-network-generators.md), not completion
of the graph inventory, mechanism portfolio or workbench.

## Delivered interface and laws

Public functions live at `r0.modules.priors`:

- `sample_causal_graph(CausalGraphPrior(...), seed=..., task_id=..., limits=...)`
  returns immutable parents, prior settings and draw identities.
- `sample_causal_recipe(graph, task=..., seed=..., task_id=..., widths=..., limits=...)`
  returns an ordinary `CausalRecipe`. Empty widths select scalar nodes; a tuple
  supplies each node's channel count. Serialized recipes preserve all realized
  weights, biases, readout channels and topology provenance.
- `sample_recipe(family, profile="causal-affine-v1", ...)` exposes `forward`,
  `layered` and `redirection` through the existing bounded runner and explorer
  session. This is an explicitly limited r0 exploration profile, not a published
  peer prior or the final broad #28 profile.

`forward` partitions `nodes` into `components` contiguous groups with boundaries
`floor(nodes*k/components)`. Each possible earlier-to-later edge within its group
is an independent Bernoulli draw with `edge_probability`. Groups may themselves
be disconnected; the parameter counts partitions, not guaranteed connected
components. Zero-edge draws are preserved, with no connectivity repair.

`layered` repeats `layer_widths` in each partition. Their sum times `components`
must equal `nodes`. Only consecutive layers are eligible for independent
Bernoulli edges. Empty edge masks and disconnected nodes remain valid.

`redirection` constructs a rooted tree per partition. A new vertex chooses an old
vertex uniformly, then with `redirection_probability` attaches to that vertex's
predecessor when it has one. All edges are oriented old-to-new. The first vertex
is a root; exactly one parent is added for every other vertex. Probability one
therefore gives a star; zero gives a uniform recursive tree. `edge_probability`
is inapplicable and a nondefault value rejects. This family is distinct from the
still-outstanding configurable preferential-attachment extension.

The [TabPFN-v2 Methods](https://www.nature.com/articles/s41586-024-08328-6)
names growing-network redirection and disconnected graph unions. It also describes
density changing with redirection, which the single-parent reconstruction above
does not reproduce. The exact unpublished DAG construction remains unresolved;
this tree is not evidence for its full graph distribution. The Gamma probability
and log-uniform size hyperpriors are also still outstanding. No silent clipping
or invented exact peer fidelity is claimed.

The affine construction (`gaussian-affine-v1`) draws independent weights
`Normal(0, 1/fan_in)` where the second argument denotes **variance**, biases with
standard deviation 0.25, and independent Gaussian innovations of standard
deviation 1 for roots and 0.1 for nonroots. Node widths and concatenated parent
widths can differ. Parameters use separate keyed streams; generated task rows,
query labels and context partition never determine them. All node/channel pairs
are uniformly permuted; the first feature-count channels are observed, and the
following score-count channels are targets. This is the mixed-observation profile,
without replacement or target-feature overlap. Multiclass requires one score
channel per class and uses the existing adapter, not native categorical nodes.
`signal_features` and `active_features` in the shared task controls do not override
actual graph ancestry; the card reports the realized causal relationship.

The convenience profile samples 2–16 additional nodes beyond the required
feature/score channels and 1–3 partitions. For layered graphs, it rounds up to equal
partition sizes, selects 2–5 layers bounded by partition size, and chooses distinct
uniform cut points. Bernoulli probabilities are uniform over
`{0, .05, .15, .35, .65, 1}`; redirection probabilities over `{0, .25, .5, .75, 1}`.
It samples scalar nodes; explicit widths are available through the direct sampler.
These numeric choices are r0 exploration choices awaiting independent assessment,
not source-attributed settings or measurements of production suitability.

## Admission, replay and inspection

Graph geometry is bounded before weight allocation using `CausalLimits`.
The full scalar/vector dimensions enter the same parameter and estimated-storage
admission as authored causal recipes. The topology sampler additionally bounds its
adjacency representation. Estimates are not measured peak RSS or hard OS limits.
Large combinations can reject; this does not assert support for the Cartesian
product of 20,000 rows, 256 features, 100 classes and maximum graph complexity.

Changing an edge in an authored edit requires clearing or replacing `topology`
because saved parents must agree with the realized structural equations.
Mechanism edits should mark `construction="authored"`; construction identifies a
sampling origin, not a cryptographic verification of hand-edited parameters.
Stored actual node parameters are authoritative for replay.

The explorer session can generate, inspect selected original rows, save, reopen
and replay these recipes through its existing JSON request interface. This is
backend integration evidence. New family controls, architecture views, causal
response/intervention UI and actual browser/Colab demonstrations remain pending.
An observational SCM is not generally a deterministic function of the observed
features, so the existing deterministic response-curve operation is not evidence
for a causal intervention.

## Reproducible checks

Each RED checkpoint is retained as an actual commit. Run its named test at that
revision, and at its succeeding implementation commit for GREEN:

| Behavior | RED | GREEN | Public test |
|---|---|---|---|
| Forward partitions and replay | `708d680` | `bf7a57e` | `test_forward_graph_keeps_disconnected_components_and_extreme_probabilities` |
| Layered boundary law | `22383a7` | `6ebdf8a` | `test_layered_graph_respects_layer_boundaries_in_each_component` |
| Vector task, trace and JSON replay | `e6e45f3` | `c01ddbe` | `test_sampled_vector_graph_is_a_replayable_task_with_saved_topology_provenance` |
| Predecessor redirection law | `f281fcd` | `3bbd74d` | `test_growing_redirection_tree_preserves_its_actual_attachment_law` |
| Saved explorer family sampling | `7070bc8` | `80bedc4` | `test_sampled_causal_session_saves_reopens_and_replays_its_graph` |

Tests are in `tests/priors/test_causal_graph.py` and
`tests/explorer/test_causal_session.py`. Use a matching environment and checkout:

```sh
UV_CACHE_DIR=/tmp/r0-28-uv-cache \
UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv \
python3 scripts/run_check.py pytest \
  tests/priors/test_causal_graph.py tests/priors/test_causal_generation.py \
  tests/explorer/test_causal_session.py
```

These tests establish the listed behavior only. Full-inventory evidence,
independent topology statistics, resource measurements, GPU and browser/Colab
acceptance are not established by their passing.

## Source coverage status

The finite source inventory remains authoritative. `SPEC.G08.01` (forward DAGs),
`SPEC.G01.01` and `P1.graph.01` (layered sparse DAGs), and `P2.graph.01`/`.02`/`.03`
(redirection and graph unions) have implementation foundations here, **not closed
option-level acceptance**. P1's neural mechanisms and dropout variants are still
missing; P2's exact distribution and hyperpriors are still missing. Every other
documented option remains required. No paper-wide coverage count is inferred from
these three families.
