# Issue #28: causal-graph and random-network generators

Status: accepted design on 2026-11-26; implementation authorized by the active user goal.
Authority: this document is the single authoritative accepted scope for issue #28.
Ticket: https://github.com/Avistian/r0/issues/28

## Accepted decisions

1. Deliver very broad coverage of configurable causal-graph and standalone
   random-network generator families. Maintain option-level coverage
   of the source corpus; the final paper-option completeness requirement below
   governs inclusion, adaptations and any proposed exclusions. Family choice
   and production mixture selection remain distinct decisions.
2. Extend all existing shared generation and delivery paths, including the public
   generation interface, local interactive exploration and creation/save, and Colab
   interactive exploration and creation/save. Extend inspection beyond these existing
   capabilities as specified in round 2 below. Library-only
   delivery is insufficient.
3. Preserve numerically valid weak-signal, saturated and disconnected draws by
   default, with diagnostic flags. Reject broken task contracts, numerical failures
   and declared resource-limit breaches. Explicit inspection filters must expose
   their effect on the retained distribution and remain separate from default
   generation. Production selection/filtering of valid tasks remains #30 policy.


## Accepted family and workbench scope (round 2)

4. Include forward-ordered random DAGs; layered graphs; growing graphs with
   redirection; Cauchy connectivity graphs; hierarchical motif assembly;
   acyclicity-preserving graph rewrites; disconnected components; hidden common
   causes; and nuisance branches. Include separately identified preferential-
   attachment extensions. Coverage must be very broad across the relevant papers,
   not limited to these examples or the existing #23 representatives. Expand the
   source-backed inventory before closing the design. Distinguish published recipe
   reproductions, adaptations and r0 extensions. Compare structural diversity.
5. Support composable linear/identity, neural, tanh, sigmoid, ReLU/smooth ReLU,
   absolute-value, polynomial/power, sine, domain-safe logarithm, step and modulo
   node mechanisms. Support joint-parent and separate per-parent functions with
   sum, product, average, maximum, logsumexp or random neural aggregation.
   Include heterogeneous mechanisms within a graph, configurable internal noise,
   explicit complexity limits and visible domain safeguards. Reuse applicable #27
   mechanisms. Retain inspectable equations and realized values.
6. Include dense, individually sparse, block-sparse, skip-connected/residual,
   irregular feed-forward and gated-mixture standalone networks, with a broad
   family of options. Expose depth, width, biases, weight distributions/scaling,
   activation placement and feature connectivity; support homogeneous and mixed
   activations. These networks sample functions; their weights are not fitted to
   generated labels.
7. Include cause-to-effect, effect-to-cause, shared-hidden-cause and mixed-observed-
   subset prediction profiles, including irrelevant variables and internal-node
   targets. A target descendant is not automatically leakage in an observational
   prediction task. Held-out-label exposure is a contract violation; trivial
   predictability and near-copy features have separate diagnostics.
8. Extend the explorer into an interactive mechanism workbench, both locally and
   in Colab: node/unit drill-down with inputs, noise, outputs and equations;
   ancestry/hidden-variable/inactive-unit/saturation overlays; editable edges,
   mechanisms and feature/target assignments; saved intervention/edit comparisons;
   and across-seed coverage dashboards with original-versus-filtered distributions.
   Support save/reopen/replay and export for the new views. Distinguish causal
   graphs, network architectures and generation-process diagrams.
9. Include hard and soft interventions with inspectable, replayable comparisons
   and worked examples. Production mixtures, observation effects and selection
   policy remain #30; temporal/dynamic mechanisms remain #39. Investigate unresolved
   specialized priors before finalizing the inventory, with explicit dispositions.

## Accepted sampling, intervention and scale scope (round 3)

10. Support scalar and vector-valued causal nodes, partially observed channels,
    dimension-changing edges and residual mechanisms throughout generation,
    visualization, editing and interventions. Expose channel-level relationships
    where a node-only diagram would hide them.
11. Support independently sampled node choices and shared per-task preferences
    over mechanisms, activations and weight families. Expose preference values and
    concentration, including homogeneous, loosely related and heterogeneous graphs.
    These are generator hyperpriors, distinct from production mixture selection.
12. Training-ready mechanisms remain fixed after sampling and calibration. Use
    analytic quantities where available or independent reproducible calibration
    samples, retaining the resulting constants. Query rows and labels must not
    determine calibration. Explicitly label deviations from published recipes;
    any batch-dependent reference mode must be named separately with its limits.
13. Include Gaussian, uniform, bounded, heavy-tailed and multimodal roots;
    configurable additive and signal-dependent structural noise; and Gaussian,
    sparse, factorized, kernel-derived and transformed weight matrices. Keep
    structural noise distinct from #30 added observation noise. Define mathematical
    compatibility rules, including required finite moments for normalization.
    Preserve valid extreme draws; expose domain restrictions and numerical failures
    without silently clipping values.
14. Support matched and population intervention comparisons. Matched comparisons
    preserve unaffected mechanisms, row identities and original exogenous draws
    wherever the intervention permits, retaining the original structure except for
    the intervention's declared changes. Population comparisons allow newly sampled
    exogenous draws and report distribution changes. Distinguish mechanism editing,
    causal intervention and observational conditioning in the UI and saved evidence.
15. Inherit the current #27 workload envelope: 16-20,000 rows, 1-256 observed
    features, 2-100 classes, and generation infrastructure designed toward 100
    million independently sampled source tasks. These are the current envelope,
    not permanent architectural ceilings; allow future expansion. Graph size,
    latent dimensions and network complexity have separate explicit limits based
    on resource measurements. Support configurable larger exploration workloads;
    distinguish measured bulk-generation recipes from expensive references.
    Document supported combinations rather than promising the Cartesian product
    of maximum settings. Separate measured throughput from extrapolation. This
    design target does not authorize a full or paid run.

## Accepted completion scope (round 4)

16. Support genuinely binary/categorical variables, including mixed continuous/
    discrete graphs, correctly typed observed features and native categorical-node
    targets. Extend shared schemas, exports and exploration, rather than treating
    integer-valued continuous features as categorical support. Preserve existing
    probabilistic target adapters as separately identified constructions. Show
    whether the predicted outcome is a graph variable or a label sampled through
    an adapter. Observation corruption and missingness remain #30.
17. Deliver reusable inspectable random tree functions needed inside causal
    mechanisms within #28, using shared interfaces for #29 reuse. The broader
    standalone tree/ensemble generator portfolio remains #29. Completion of all
    of #28 must not depend on waiting for #29's separate design. The final all-paper-
    options requirement includes both directly sampled trees and structural tree/
    forest mechanisms fitted on synthetic random data where documented; do not
    substitute one for the other. This includes documented DT, XGBoost, RF,
    ExtraTrees and GBDT component variants, while retaining the standalone boundary.
18. Acceptance requires an inspectable evidence package for every included family
    and meaningful composition alongside automated checks: source-to-implementation
    coverage distinguishing reproductions/adaptations/extensions; worked numerical
    traces and independently checked interventions; valid/flagged/failed examples
    and complete accounting; across-seed structural and functional diversity and
    filtering effects; replay, CPU/GPU agreement, resource and supported-size
    measurements; and actual local-browser and Colab demonstrations of generation,
    editing, comparison, save and reopen. Unexecuted environment checks remain
    PENDING. Correctness and diversity do not establish predictive superiority;
    downstream model comparisons require separate evidence.
19. The implementation author specifies remaining parameter ranges, distributions,
    diagnostic thresholds, numerical tolerances and per-recipe resource limits in
    explicit versioned recipes, supported by sources and measurements and checked
    by independent review. Preserve known published reference settings and provide
    broader exploration recipes separately. Diagnostic thresholds flag properties,
    not silently reject valid draws. This delegation does not authorize removing
    agreed families, reducing delivery scope or making temporary limits permanent.

## Paper-option completeness requirement

The user's final scope condition is: "make sure that it covers all options from
papers." Every documented option relevant to #28 in the source corpus must have
an option-level entry and a delivered implementation or explicitly identified
adaptation with evidence. A representative per paper or family is insufficient.
The corpus includes the #23 audit's relevant papers and the additional papers in
the inventory below; documented in-scope options discovered during implementation
must be added rather than silently omitted. Before implementation, complete the
source-option extraction and record the finite implementation inventory plus
justified sibling-ticket boundaries; source gaps must be recorded explicitly.
Publication dates and versions, source locations and code commits where available
must accompany the inventory.

This final condition supersedes any earlier suggestion that the implementation
author can freely defer a documented in-scope option. Removing or deferring such an
option requires an explicit scope decision from the user. Established ownership
boundaries (#30 observation/selection/mixtures, #39 temporal/dynamic behavior and
#29 standalone tree breadth) still apply: cross-reference those options and provide
the #28 integration surface, rather than count them as implemented in #28.

Track each option independently through source extraction, implementation,
inspection artifacts and verification. Distinguish paper descriptions from source
code, preserve documented disagreements as named variants, and label necessary r0
adaptations. Undisclosed algorithms/settings remain UNKNOWN; do not invent them or
claim exact full-prior/checkpoint reproduction. An UNKNOWN item needs a documented
resolution or explicit user disposition before a full-coverage claim. Coverage is
relative to the recorded corpus; do not claim every paper in existence was surveyed.

## Required evidence from the ticket

- Sampled causal graphs and structural equations; network diagrams with feature
  and target assignments.
- Across-seed depth, output dispersion, failure counts and reasons.
- Inspectable sampled mechanisms and parameter distributions; diagnostics for
  degenerate graphs, saturated outputs and target leakage.
- Saved valid and failed/flagged examples for each candidate.
- A worked causal example traced from parent values and saved node noise to target;
  a network response slice; saturated-versus-accepted comparisons; and the effect
  of inspection filtering on retained graph/network variety.

## Remaining work and owners

There are no unanswered product-scope decisions from the interview. The following
work is delegated or pending execution, rather than implicitly completed.

| Work | Owner | Completion condition |
| --- | --- | --- |
| Extract and implement every documented in-scope source option | Implementation author | Option-level matrix with exact source pointers, named adaptations and evidence; no silent omissions |
| Resolve undisclosed source details or source contradictions | Author and independent reviewer; user for scope changes | Explicitly documented reconstruction/variant, or user disposition where complete coverage cannot be established |
| Set recipe ranges, distributions, calibration, tolerances and resource limits | Author, independently reviewed | Source-backed versioned recipes and measurements preserving accepted breadth |
| Build shared generator, workbench, exports and execution paths | Author | Accepted behaviors available through the shared API, local explorer and Colab |
| Verify mathematical and intervention behavior, coverage and replay | Independent verifier | Retained worked calculations and checks tied to the delivered revision |
| Verify CPU/GPU, browser, Colab and scale behavior | Verifier in each actual environment | Measured evidence; unexecuted checks remain PENDING |
| Approve any paid/full-scale launch | User through the existing guarded workflow | Concrete run configuration and authorization; design agreement alone is not launch authority |
| Publish and link the accepted specification | Implementation publisher | Verified commit-pinned Approved specification link in #28, preserving the remaining issue body |

## Handoff

The design is accepted. The user subsequently authorized implementation of all of
#28 through an active persistent goal. Work proceeds on `feat/issue-28-generators`.
The specification and glossary are preserved from the design interview; the initial
byte-for-byte copy fingerprint is retained in local preflight evidence.
Publication and the issue's Approved specification link are PENDING. Complete that
step when publishing an authorized implementation PR: link a published revision of
this document, verify it contains the accepted decisions, identify it as the
authoritative accepted scope, and verify the issue body contains that link without
copying the decisions or changing the rest of the issue body.

## Supporting source evidence

The [initial paper-option audit](../research/0028-paper-coverage-audit.md) retains
the source register, option groups and extraction gaps found during the interview.
The [finite source-option inventory](../research/0028-source-option-inventory.md)
records the completed bounded extraction gate, required variants, neighboring
interfaces and remaining undisclosed details. It is not implementation evidence.
This document remains the authoritative accepted scope; research notes support it.
