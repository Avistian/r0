# Prototype convex-mixture root core

This prerequisite implements P2's prototype construction idea with explicitly named r0 conventions. It is not a reproduction of the complete paper prior. Public graph, sampled source-profile, normalization, intervention and explorer integration remain required follow-ons.

The primary source is Hollmann et al., *Nature* 637, 319–326 (2025), [Initialization data sampling](https://pmc.ncbi.nlm.nih.gov/articles/PMC11711098/), DOI 10.1038/s41586-024-08328-6. It describes M = rho n prototypes and a convex combination with multinomial weights. It does not fully specify integer M, multinomial trial count or probabilities, or the mapping from temperature beta to trials. Those choices below are r0 reconstructions, not uniquely specified paper laws.

`PrototypeRootPlan` declares a joint or source-coordinate root base, reference population size, rho, integer convention and maximum prototype count. `ceil-r0`, `floor-r0`, and `nearest-even-r0` compute M without silently repairing a zero count or clipping excess. Joint bases preserve covariance and coordinate dependence. Source-coordinate bases preserve their explicit laws and optional independent calibration. Zero and noiseless prototypes are valid; normalization is separate.

`sample_prototype_root(plan, rng)` freezes an independent bank. Its reference size determines construction, not the size of future emissions. `draw_prototype_root(bank.root, rows, rng, capture=ledger)` uses one row weight vector across every coordinate: values = weights @ prototypes. Each row lies in the same frozen convex hull. The frozen bank has bounded conditional moments; its construction law can have smaller or unbounded unconditional moments. These are separate record properties, not interchangeable certificates.

`sample_prototype_batch_reference(plan, prototype_rng, weights_rng)` is the explicit alternative that constructs M from the declared batch size and emits that entire reference batch. Changing batch size constructs a new bank and may change M. It cannot be decoded or emitted as an independent frozen-bank record. Neither lifecycle reads query labels or silently normalizes from a query batch.

`PrototypeWeightLaw` offers:

- `multinomial-trial-temperature-r0`: positive integer baseline trials, effective N = ceil(trials * beta), explicit normalized probabilities or uniform defaults. Raw integer counts are retained; weights are counts / N. Larger beta increases the trial population under this declared reconstruction.
- `multinomial-probability-temperature-r0`: explicit extension with fixed N and probabilities proportional to p ** (1 / beta), evaluated stably. Zero probabilities remain zero.
- `dirichlet-concentration-r0`: explicit extension retaining independent raw Gamma draws with alpha_j = beta * M * p_j and row normalization. It requires `trials=None`; no unused trial controls are sampled.

All modes reject nonfinite products, excessive counts, invalid probabilities and numerical normalization failures without clipping, retry or fallback. Caps cover bank geometry, emitted weights and values, metadata, raw draws/numbers/bytes and RNG state before construction. Caller-owned raw ledgers account for prior consumption. Failed sampled construction rolls back every supplied RNG, including failures after the bank has been drawn. Batch streams must be distinct.

Frozen records retain plans, source and lifecycle labels, bank values, raw integer counts or Gamma variates, and initial/final generator snapshots. Strict decoders bound arrays before conversion and reconstruct the bank and batch to compare all raw values, outputs and final states. Public direct emitters use caller-owned ledger custody; sampled records provide standalone replay envelopes.

Author verification is retained under ignored `docs/evidence/issue-28-prototype-roots/`. It includes literal NumPy arithmetic across five supported engines, count conventions, cross-coordinate dependence, row prefixes, both lifecycles, heavy-tail construction versus conditional moments, corrupted records, caps and rollback. Full-suite and integrated graph/UI acceptance are separate.
