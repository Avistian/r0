# Topology hyperprior graphs: explicit mechanism templates

This slice attaches the retained size/redirection construction from
[the topology hyperprior core](issue-28-topology-hyperpriors.md) to an executable
functional causal task. `HyperpriorGraphRequest` names a topology prior, task,
root template and one-parent edge template. `sample_hyperprior_graph` freezes the
complete topology receipt in `HyperpriorGraph` (`sampled-hyperprior-graph-v1`).
The resulting functional graph is derived from those exact templates and parents;
it is not stored as an independently editable second copy.

The root template has no parents and an explicit root law. Its output width is
the graph state width. The edge template has placeholder parent `(0,)` and equal
input/output state width; each realized node substitutes its actual predecessor.
Both templates accept the existing admitted frozen function interface, including
sampled mechanisms with retained receipts and one-child parent compositions.
Their innovation laws and numerical pipelines remain attached. Each root/node
uses the ordinary graph's distinct keyed row innovations despite sharing equation
parameters. Template sharing is an **r0 extension**, not a claim about P2's full
mechanism sampling law.

Readouts use the explicit `prefix-features-suffix-targets-r0-v1` layout: features
are the first flattened node channels, target scores are the last channels, in
node/channel order. Every size in the prior support must have enough distinct
channels for both. Multiclass uses one target score per declared class; binary
and regression use one. Classification labels and probabilities still come from
the existing target adapter, not from the forest's predecessor-selection draws.

Construction admits the worst possible node count, complete retained metadata,
both template geometries, nested sampled function/root replay, original-row
emission and topology raw/snapshot budgets before opening the topology RNG
transaction. Summed all-root and chain reservations conservatively dominate any
mixture of those node types. These are estimates, not measured RSS. Strict
archive decoding performs the same raw geometry admission before restoring any
construction stream; caller limits only tighten admission. No rejected sample
is retried, and no node count is clipped.

`recipe_from_dict`, `generate_task`, `explain_generation`, ordinary generation
runs and task ZIP save/load preserve the complete envelope and exact identities.
Inspection uses original row indices and checks the topology card against its
retained construction. `with_task` changes task controls without redrawing the
forest; a genuinely new construction requires a new sampler call. Existing
fixed-size graph formats and their source samplers are unchanged.

## Required follow-ons

This is the explicit-template integration slice. Jointly sampled mechanism priors,
geometry/observation selection, native and converted task wrappers, composition
with intervention envelopes, and a dedicated guarded workbench constructor are
still required. In particular, the existing joint mechanism sampler currently
resolves named plans before full family admission; this slice does not silently
relax its own before-RNG worst-size contract to reuse that ordering. No complete
P2 reproduction, scientific equivalence, hosted/GPU result, or full-suite result
is claimed here.

## Outer-envelope preflight repair

Independent review of `20332375b166de6e5fd41b3b18c19731bcdad5f4` found that an invalid layout restored four sampled source-root snapshots before rejection. That historical REJECT remains retained. Graph decoding now first parses and validates the complete topology receipt without restoring its construction stream, including snapshot headers, raw draw records, derived Gamma/topology values and request/cap identity. Discarded authored template previews pass through the existing node, request and graph constructors to validate outer versions/profiles, ownership, geometry, readouts and runtime compatibility. The original complete template records remain authoritative for subsequent source replay.

The standalone topology decoder shares that metadata parser, then performs its existing exact numerical replay. Snapshot header checks share the restoration validator. Worst-size and complete original metadata admission still precede preview construction; no declared cap is increased. Focused public RED/GREEN, the original outside-test sentinel and complete five-engine source/Conv-template artifact comparisons are retained in ignored `docs/evidence/issue-28-hyperprior-preflight/`. Independent repair acceptance and the remainder of the stopped core/graph review are separate gates; this repair does not claim complete issue28 acceptance.
