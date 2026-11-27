# Joint native vector causal interventions

`CausalInterventionRecipe` and `generate_causal_pair` accept authored
`NativeVectorRecipe` and retained `SampledNativeVectorGraph` bases, including
parent-composed joint functions. The base construction archive is retained and
replayed unchanged. Existing scalar and vector recipe versions remain unchanged.
The logical intervention semantics below are an explicit **r0 extension**, not a
claim that an undisclosed paper intervention law has been reproduced.

`NativeVectorHardIntervention(node, variable, value=...)` clamps one continuous
logical emission. For a discrete variable use `category=...`, naming its declared
vocabulary label. Its version is `native-vector-hard-intervention-v1`. Flattened
score-channel clamps are intentionally invalid for this family: one logical
variable can occupy one score slot and several encoded-state slots.

The clamp replaces the emitted value and its entire descendant-state block.
Continuous states have one column; discrete states retain their declared one-hot
cardinality. A partial clamp evaluates the one joint function, pipeline and
correlated innovation before replacing selected emissions. Other logical
variables therefore retain their joint dependencies. When every logical variable
at a node is clamped, its original root/function/pipeline/noise/emission draws are
bypassed, including mechanisms whose unneeded evaluation would overflow.

Forced probabilities are K-wide category indicators, not original probabilities.
A forced softmax/argmax score block contains the same indicator; a forced threshold
score retains its single column and explicitly contains the category index,
not an original threshold score. Inspection labels and mechanism metadata expose
this distinction, while typed target values and encoded parent geometry remain
valid. Full bypass does not mean the baseline of a paired comparison can be
invalid: the baseline still must generate successfully. Direct after-generation
can bypass a numerically invalid original equation.

`NativeVectorDomainIntervention(node, variable, domain)` uses
`native-vector-domain-intervention-v1`. Replacement must preserve kind,
vocabulary, score span and encoded-state span. Frequency weights, temperatures,
threshold cutpoints and same-width emission changes are available. Frequency
weights do not guarantee empirical class marginals.

Existing `SoftIntervention` replaces a whole joint function and/or innovation
parameters; `StageIntervention` replaces the joint root law or pipeline. Scalar
stage-domain fields are ambiguous on vector nodes and reject; use the explicit
logical-variable record. Disjoint joint-stage changes, variable domain changes
and partial clamps can be combined. Duplicate joint fields or duplicate variable
locations reject, independently of record order. Numerical readout converters
remain incompatible with native vector domains, as in the original vector API.

Matched mode retains original vector RNG keys, downstream innovations and
unchanged per-variable emission uniforms. Changed root laws preserve initial
stream identity without asserting equality of transformed root values. Population
mode changes the exogenous population while retaining split/row positions; these
are not the same units. Pair comparison regenerates both tasks, checks typed
arrays and provenance, and uses a canonical contiguous reduction layout for
vector summary statistics so save/load does not change floating-point summaries.
Admission reserves both runtimes and retained traces before evaluation.

The explorer exposes logical-variable selection, native vocabulary choices,
continuous values, domain JSON, joint function/noise JSON, and root/pipeline JSON.
Target metadata uses the effective domain while retaining the original domain;
a clamp on a target sibling is not mislabeled as a target clamp. Guarded create,
reopen, repeated-row traces and export preserve the original active dataset and
retain failure receipts. Source construction and calibration are not rewritten.

Public tests cover independent one-hot arithmetic and softmax formulas, correlated
innovations, full/partial bypass, joint pipeline changes, unequal cardinalities,
parent composition, decoding, tampering, admission and persistence. Actual
Chromium covers both vector presets, typed controls, comparison traces,
reopen/export and failed-state preservation. Author evidence is ignored under
`docs/evidence/issue-28/vector-interventions/`. These checks are bounded; hosted
Colab, GPU/scientific reproduction, stochastic per-row assignments and the broader
accepted paper-option inventory remain required or unverified as documented.
