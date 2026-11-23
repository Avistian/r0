# Issue 22 predictive oracles

Status: all three interview rounds accepted; implementation authorized by the user with $implement 22. Implemented locally. Evidence and review history are in ../evidence/issue-22/. Publication and hosted CI remain pending.

Issue: https://github.com/Avistian/r0/issues/22

## Accepted decisions: round 1

1. Establish reference predictions and uncertainty under declared priors, independently of a learned model. This is a correctness foundation, not evidence of competitive generator quality. Generator-quality diagnostics and challenges belong to #23 and #24; learned comparisons follow later.
2. Support beta–Bernoulli binary prediction and Dirichlet–categorical multiclass prediction, declared asymmetric prior parameters, and classes absent from context. Retain the uniform-prior example: three heads and one tail give next-head probability 4/6.
3. Support one-dimensional linear and squared-exponential/RBF GP references with declared parameters and constant prior mean. Retain other kernels and mixtures as future options; neither these examples nor their parameters select a production prior. No general kernel framework is required.
4. The oracle receives context observations, query inputs and an explicit reference specification. Query targets and sampled latent truth remain outside prediction. Record prior assumptions so later comparisons cannot silently give the reference additional information.
5. Report latent-function and new-observation uncertainty separately. Declare observation noise as a variance. Record numerical stabilization separately from modeled noise.
6. Independently calculate classification answers with rational arithmetic and GP answers with scalar or small-matrix derivations. Retain inputs, equations, expected answers, observed answers and differences. Shared implementation helpers are not an independent reference.
7. Reuse episode/predictor-view boundaries and replay/provenance conventions. The six-row fixture and fixture-v1 limits from #21 do not constrain these references or future generators. Preserve raw coordinates and declared kernel units explicitly.
8. Deliver readable reference cards and a bounded CPU regeneration command. Automated checks supplement inspectable calculations. No training, bulk generation or live W&B integration; tracking integration remains with its planned ticket.

## Accepted decisions: round 2

9. A deterministic oracles module consumes a PredictorView and a separate validated reference specification. It never consumes full episodes or source tasks. Priors owns episode construction; replay owns orchestration and reporting. Unsupported specifications fail explicitly.
10. A separate prior-predictive calculation supports zero-observation reference cards. Scored episodes preserve #21's nonempty-context contract.
11. Classification labels share one unknown class-probability vector within an episode; features are uninformative. Prior parameters are finite, strictly positive and aligned with the declared class vocabulary. Return posterior parameters and per-query class probabilities. Identify the heads label explicitly for coins.
12. GP output includes means, marginal variances and full joint query covariance in declared query order. Classification exposes marginal probabilities and posterior parameters, without a joint-label scoring API. Query additions/reordering cannot change other queries' marginal predictions; this does not imply independent outcomes.
13. Use a constant mean m. Linear kernel: k(x,x') = s²xx'. RBF kernel: k(x,x') = s² exp(-(x-x')²/(2ℓ²)). Signal variance and RBF length scale are strictly positive. Independent Gaussian observation noise has one finite variance σ² >= 0 shared by context and future observations. No parameter fitting or marginal-likelihood optimization.
14. GP input is exactly one fully observed continuous feature in raw coordinates. Targets retain original units; oracle calculations bypass context standardization. Reject missing inputs, extra features and unsupported kinds. Record feature/target units.
15. Use float64 Cholesky solves, never explicit inversion or automatic jitter escalation. Jitter defaults to zero. Explicitly requested jitter is recorded and identifies a stabilized system. Repeated coordinates are allowed when solvable. Reject singular noiseless systems clearly; pseudoinverse support is deferred.
16. Declare case-specific tolerances. Small well-conditioned cases default to atol=1e-10 and rtol=1e-10. Retain observed absolute/relative differences and separate symmetry/variance checks. Exceptions require prior justification. Correct tiny negative variances only within a declared numerical bound, and retain the correction; larger violations fail.
17. Mathematical agreement is the correctness criterion. Query-label diagnostics include classification Brier score, negative log probability and AUC (explicit user addition), plus regression MSE and Gaussian negative log density using observation variance. Zero-variance density scoring is undefined with a reason. No per-sample requirement that the oracle beat a baseline. These metrics do not select the benchmark protocol. AUC convention and undefined-case handling are specified in round 3.
18. Evidence covers: coin 4/6, asymmetric prior and successive observations; multiclass absent context class and asymmetric prior; GP prior-only, one-observation calculation, RBF near/far, small linear case and noisy repeated coordinates; invalid parameters, unsupported inputs and singular noiseless failures; query-label isolation; context/query reorder invariance within tolerance. Cards show assumptions, intermediate calculations, expected/observed outputs and differences.
19. Author explicit tiny observations, recording that they were not randomly sampled. General-purpose stochastic generators remain with their generator tickets.

## Accepted decisions: round 3

20. AUC means ROC AUC. Binary problems declare the positive class. Multiclass reports per-class one-versus-rest AUC and the unweighted macro-average across the declared vocabulary. Preserve class ordering and give tied scores half credit. PR AUC is a separate future option.
21. Score AUC per episode, never by pooling queries across episodes. Binary AUC requires both classes in query targets; each multiclass component requires positives and negatives. The full macro-average is undefined if any declared class has an undefined component. Retain valid components and reasons for missing values. Constant within-episode oracle probabilities yield AUC 0.5 whenever defined. Demonstrate both ties and undefined cases.
22. Add r0 oracle --config … --out … with its own strict, versioned TOML format. Declare family, parameters, authored observations, query locations and numerical policy. Reject unknown keys and incompatible combinations. Prior-only cards are explicit and do not create empty-context episodes. Query targets live in a separate scoring section; predictor inputs are constructed independently. Existing fixture replay behavior is preserved.
23. Explicit jitter affects only the context conditioning covariance diagonal. It is neither added to new-observation noise nor disguised as a probabilistic parameter. Retain the effective conditioning matrix and label stabilized results. Mandatory exact-reference examples use zero jitter; a separate example demonstrates explicit stabilization.
24. The numerical correction bound is 1e-12 times max(1, largest prior query variance). Within that bound, symmetrize covariance and clamp tiny negative diagonal entries to zero, retaining original values and corrections. Larger violations fail. Check covariance positive-semidefiniteness within the same bound; never silently repair eigenvalues.
25. Reports retain resolved configuration, episode/reference identities, environment versions, raw observations, actual predictor inputs, posterior parameters or GP matrices, predictions, separate scores, tolerances and numerical differences. Independently derived expected answers belong to versioned reference cards, not outputs generated by the oracle. Output paths and timestamps do not affect semantic identity.
26. The reference command accepts at most 64 context rows, 64 query rows and 16 classes per case. Reject oversized requests before covariance allocation. These are reference-command bounds, not limits on future generators. Use CPU only; record walkthrough duration as an observation, not a hardware-independent pass threshold.
27. Completion requires every mandatory card to regenerate, every documented invalid case to fail clearly, and independent reviewer calculations for the coin 4/6 result, a multiclass posterior, one-observation GP mean and both variances, and tied-score AUC. Retain arithmetic and observed differences. Repository checks supplement this evidence.

## Consolidated implementation scope

- Add deterministic predictive references in the oracles module, consuming the existing predictor view and a separate validated reference specification.
- Add explicitly authored tiny reference cases in priors and orchestration/reporting in replay, exposed through r0 oracle.
- Preserve independent hand-derived cards and bounded CPU walkthrough evidence. Include failure diagnostics, query-label isolation, reorder invariance and explicit stabilized-system evidence.
- Preserve existing episode validation and fixture replay behavior. No learner training, general-purpose stochastic generator, production prior selection, benchmark-protocol selection or live W&B integration is included.
- No new ADR is required by these decisions: the scoped reference interfaces and numerical conventions remain reversible and are documented here.

## Implementation notes

All design questions Q1–Q27 have accepted answers. The user confirmed implementation by invoking $implement 22.

An optional --expected path supplies a versioned JSON reference card to r0 oracle. It maps report paths to independently derived numeric answers; absent a card, verification is explicitly NOT_RUN. A mismatch retains a FAIL report and exits nonzero. This keeps expected answers outside the predictor and preserves comparison evidence.

Run records are created atomically and never replaced with different contents. Repeating an identical report is allowed. Prior-only GP configurations reject nonzero jitter because they have no conditioning matrix. Units live in the reference configuration/report, preserving #21's feature schema unchanged.

## Mathematical reference

Rasmussen and Williams, Gaussian Processes for Machine Learning, chapter 2:
https://gaussianprocess.org/gpml/chapters/RW2.pdf

The reference distinguishes posterior latent-function covariance from new-observation covariance, which additionally includes observation noise.

ROC AUC conventions:
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
