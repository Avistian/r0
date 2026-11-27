# Native graph variables

`NativeCausalRecipe` (`native-causal-v1`) is an authored mixed-domain DAG. Each
`NativeCausalNode` contains a frozen `FunctionalCausalNode` equation and a
`VariableDomain`. Equations use the existing frozen function families, root laws,
structural noise and activation pipelines. Parent-before-child ordering and
encoded input widths are checked before evaluation. Numerical readout converters
are rejected because they would create a second, conflicting variable readout.

Continuous nodes retain their vector states. A binary or categorical node emits
one integer variable and passes a full declared-width one-hot state to its
children. Cardinality is fixed by a unique, nonempty vocabulary, including levels
that happen not to appear in a finite sample. Domains support:

- `softmax`: one score per category, positive temperature, one retained uniform
  draw per row; inverse-CDF sampling with a final endpoint of one.
- `argmax`: one score per category; ties select the first declared category.
- `threshold`: one score and exactly K−1 strictly increasing fixed cutpoints;
  equality belongs to the upper interval.

A target selects a variable/channel directly. Its task type and class count must
match the domain; target adapter controls must remain at defaults. No label
adapter runs. Native targets are recorded as `native-variable-v1`, and discrete
targets retain integer codes and their declared classes. Continuous targets use
the selected channel directly without empirical normalization.

## Shared outputs

`generate_task`, `recipe_from_dict` and `explain_generation` support the native
recipe. Inspection replays original rows in requested order, including duplicates,
and checks saved native emission uniforms and final values. Source identity
excludes episode split/context size while retaining the complete equation/domain
configuration. Saved mechanism cards retain all native emission uniforms for
future matched interventions. Admission charges one-hot states, retained scores,
probabilities, values, uniform draws, certificate sets and serialized copies.
This remains an estimate rather than a measured RSS limit.

`TrainingTask.feature_schema` stores column kinds and declared vocabularies;
`inputs` remains a numeric transport buffer. `episode()` produces binary integers,
categorical strings and continuous floats. Group archives retain the schema and
target origin in predictor metadata; loading checks discrete codes. Default old
schema/origin fields are omitted, preserving old archive member bytes.

## Source mapping and remaining work

[TabAbax §3.3](https://arxiv.org/html/2611.00562v1#S3.SS3) describes discretization,
softmax category sampling and consistent cardinality/one-hot width. This increment
provides authored versions of those emission families; it does not reproduce its
complete prior or guarantee every declared category occurs in each finite table.
Its reported frequency-distribution family, including Dirichlet frequencies,
still needs sampled construction and provenance integration.

[Kumo Tabular Appendix A.4](https://arxiv.org/html/2611.23001v1#A1.SS4) describes
argmax categories for features and targets. The authored deterministic option
here exposes that behavior; the paper's selection probability is not installed as
a default mixture.

Still open within #28: heterogeneous vectors containing multiple discrete/native
variables per node, sampled category/cardinality/frequency/threshold priors in the
shared preference registry, geometry constructor/UI wiring, and native equation or
domain pair interventions. The old adapter-pair interface explicitly rejects native
recipes. These limitations do not narrow the approved broader specification.
