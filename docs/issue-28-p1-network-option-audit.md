# P1 network option audit

This reconciles the ten `P1.net` records in the finite source inventory against
source `211a4df`. It uses the recorded P1 source contracts in the
[hyperprior](issue-28-network-hyperpriors.md),
[innovation](issue-28-network-innovations.md) and
[structural-network](issue-28-structural-networks.md) notes. This is an
implementation/evidence audit, not a new upstream source extraction or an
independent reviewer verdict. All ten have executable implementation paths;
that does not establish a complete TabPFN prior or checkpoint reproduction.

The implementation files below are under `src/r0/modules/priors/lib/`; public
entry points are exported through `r0.modules.priors`. Test names identify
specific assertions, not merely a directory containing related tests.

| ID | Implemented behavior and entry point | Inspected checks and limits |
| --- | --- | --- |
| P1.net.01 | `sample_network(NetworkSamplingRequest(...))` builds frozen random weights; `NetworkRecipe` emits tasks. Seven architecture families are available. | `test_sampled_network_standalone_replay_and_prefix` covers all seven. `test_network_generation_archive_roundtrip` covers saved datasets. Generic skip/gated/irregular families are r0 extensions, not P1 source claims. |
| P1.net.02 | `network_sampling.py` samples retained elementwise masks; `mask_profile="p1-individual"` exempts the first layer and applies the named P1 scaling law. | `test_p1_named_mask_scaling_and_first_layer` checks raw weights/masks/scales. The TabICL scaling variant has a distinct profile. Frozen weight masks are not per-row dropout. |
| P1.net.03 | `mask_profile="p1-block"` samples disjoint diagonal blocks and retains realized keep-fraction scaling. | The same arithmetic test checks block masks and both allowed exponents. Remainder rows/columns outside floor-sized blocks stay masked. The complete upstream mixed selector is not implied. |
| P1.net.04 | `NetworkInnovationPlan.importance` and sampled `NetworkInnovationRequest.importance` multiply outgoing input-weight columns. | `test_importance_is_outgoing_weight_not_input_rescale` checks the literal nonlinear equation and rejects the alternative input-rescaling interpretation. The source does not disclose the full importance law; the requested law is an explicit adaptation. |
| P1.net.05 | `sample_network_innovations` with shared mean/scale scopes reuses frozen statistics across participating channels. | `test_sampled_noise_statistics_replay_and_rollback`, plus the saved three-scope calculation below. Shared base scale and optional half-normal channel multipliers are separately named controls. |
| P1.net.06 | Mean and scale scopes independently support `per-block` and `per-channel`; sampled records retain each realized value. | The saved calculation verifies one, two and four scale records for shared, block and channel scopes on a 3+1-channel network. Undisclosed mean laws must be supplied explicitly for paper placement. |
| P1.net.07 | `network_source_hyperprior(source="p1-paper")` supplies the paper TNLU depth/width laws to `sample_hyperprior_network`. | `test_independent_source_quantile_oracle`, `test_extension_width_scope_bias_and_round_before_offset` and seven-family replay exercise the scalar engine and realized architecture. Operational caps reject a draw; they do not truncate it or retry. |
| P1.net.08 | The same hyperprior samples weight scale with independent log-uniform mean and sigma, then positive truncated-normal realization. | The independent SciPy oracle checks three RNG engines; effective network weight scale must equal the retained scalar value. Pinned TabICL relative-sigma behavior remains a separate law. |
| P1.net.09 | `network_source_innovations(source="p1-paper", mean=...)` uses the paper TNLU scale law with bounds 0.0001–0.3; the compiled runtime consumes the scales. | The saved SciPy calculation checks the actual noise records, archive replay and generated-value traces. `test_fixed_innovation_response_numpy` independently checks pre/post-activation arithmetic. This is not an unused metadata-only parameter. |
| P1.net.10 | `dropout_profile="p1-beta"` samples two uniform shape parameters, retained gamma draws and `0.9 * ga / (ga + gb)`. | `test_p1_beta_is_bound_to_retained_raw_gamma_draws` checks parameters and resulting dropout. Invalid gamma sums fail without retry; the profile requires individual-sparse architecture. |

## Fresh evidence

At source `211a4df`, the combined existing checks in
`test_network_hyperpriors.py`, `test_network_innovations.py` and
`test_structural_networks.py` pass. They include literal forward calculations,
raw-draw binding, archive reconstruction, altered-record rejection and rollback.
This focused run does not replace the outstanding current full-suite result.

An additional saved calculation uses a two-block network of widths 3 and 1,
explicit mean 0.1, the paper noise-scale law, and statistics seed 31. It compares
each retained scale with SciPy's truncated-normal quantile, independently of the
implementation's `statistics.NormalDist` realization:

| Scope | Realized scale count | Values | Maximum quantile / generation-trace error |
| --- | ---: | --- | --- |
| Shared | 1 | 0.1382546094639454 | 0 / 0 |
| Per block | 2 | 0.1382546094639454, 0.13816584030025894 | 0 / 0 |
| Per channel | 4 | 0.1382546094639454, 0.13816584030025894, 0.13825685329942078, 0.13787601928772106 | 0 / 0 |

Each case saves a dataset archive, reloads it, compares the episode and decoded
recipe, and traces rows 0, 15 and 0. The retained script, exact outputs, three
archives and test log live under gitignored
`docs/evidence/issue-28-p1-network-audit/`. These are bounded author checks;
zero error in these examples is not a universal floating-point guarantee.

## Remaining qualification

These ten records no longer justify an implementation backlog saying that
importance, noise scopes or noise-scale consumption are absent. They still do not
prove the whole paper prior: source-specific compositions and selectors, explicit
undisclosed-law adaptations, across-seed diversity/resource qualification and
independent review remain separate obligations. The full #28 inventory and its
other source IDs remain active. GPU and downstream predictive benefit are not
established by these CPU calculations.
