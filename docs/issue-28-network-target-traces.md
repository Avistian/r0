# Original-row network target traces

Standalone networks and raw/matched input views now expose their actual target
adapter dependencies. Existing architecture, masking, matching, signal and final
`targets` IDs remain. The final `targets` node points to the shared `target`
adapter node. Regression includes the original keyed Gaussian `noise-draw`, scaled
`noise`, and signal-plus-noise target; a zero noise ratio remains explicit.
Classification includes separation/offset logits, sigmoid or softmax probabilities,
original keyed uniform labels and inverse-CDF targets. Repeated indices refer to
original task rows, never newly drawn rows. The shared adapter arithmetic is also
used by existing generation inspection; generation itself is unchanged.

Before emitting architecture nodes, inspection replays and checks semantic task
metadata, source/episode identities, class/schema/origin declarations, mechanism
card, execution implementation and all saved arrays including probabilities and
dtypes. JSON whitespace is insignificant. Historical execution environment fields
are retained provenance, not independently verified by CPU replay. Tensor values
are compared on CPU as in the existing inspection contract; no GPU acceptance is
claimed here. The trace includes a measured target reconstruction error.

This is an inspection repair, not a new generator version. Complete generated task
records and tensors remain unchanged. Trace bytes intentionally change because
stochastic dependency nodes and the target-error header are added. Evidence and
old/new generation hashes are ignored under
`docs/evidence/issue-28-network-target-traces/`. Independent review and full release
checks remain separate from author verification.
