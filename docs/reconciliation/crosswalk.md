# T1–T20 promise crosswalk

Original wording is retained in [T1–T10](original-tickets-1.md) and [T11–T20](original-tickets-2.md). Each row below names materially distinct promises. “Absent” is source inspection at the pinned r0 revision, not an inference from failing tests. All issue numbers refer to Avistian/r0.

Evidence keys
- CLI = src/r0/modules/cli/__init__.py and the observed CLI walkthrough in reconciliation.md.
- ML = empty module entrypoints and lib/__init__.py under src/r0/modules. No implementation in the corresponding module.
- Files = git ls-files configs eval results scripts at the pinned revision.
- Policy = AGENTS.md, CODING_STANDARDS.md and docs/adr, cited in reconciliation.md.
- Make = Makefile targets calling future subcommands.
- Ownership = .github/CODEOWNERS.

| Original | Promise | Observed state and evidence | Disposition and owner |
| --- | --- | --- | --- |
| T1 | Version command | Implemented, CLI observation 0.1.0 | Retain; #21 integrates existing CLI |
| T1 | train, eval, data commands | Absent, CLI only exposes flags | Add with capabilities in #32, #26/#41 and #26/#41 respectively |
| T1 | leaderboard, ledger, summary commands | Absent, CLI | Retain outcome in #34; choose interface there |
| T1 | TOML round-trip and unknown-key rejection | Absent, CLI/ML | Strict config validation #21; frozen dataclasses remain an implementation option |
| T1 | seed_everything | Absent, ML | Replace global-helper prescription with explicit generators under CODING_STANDARDS.md; #21 owns replay contract, #32 recovery |
| T2 | Table, RelDB, TaskRows; PK, FK and time validation | Absent, reldb ML | #39 owns schema and each validation rule |
| T2 | Valid three-table fixture | Absent, reldb ML | Retain inspectable fixture in #39 |
| T3 | Sample relational task, respect shape caps, deterministic seed | Absent, priors ML | #39 owns relational generation; #23 defines coverage diagnostics |
| T3 | Label windows and invariance outside window | Absent, priors ML | #39 retains activity/value/property cases and window evidence |
| T3 | Ensure every class by resampling | Absent, priors ML | Candidate filtering policy #30/#39; measure induced bias before adoption |
| T4 | Shared DFS-lite aggregation | Absent, featurize ML | Candidate aggregate baseline #40 |
| T4 | Future and exactly-at-cutoff invariance | Policy exists, implementation absent | Binding strict cutoff; #40 owns evidence |
| T4 | 40-feature cap and NaN only for absent history | Absent, featurize ML | Historical representation choices; #21/#40 decide supported shapes and missingness |
| T4 | Vectorized million-event processing under two minutes | Absent, featurize ML | Profiling objective #40; set workload/hardware budget before interpreting timing |
| T5 | Earlier context times, latest queries, disjoint sets | Absent, batching ML | #21 disjoint episode contract; #40 temporal sampling and label availability |
| T5 | Pad to 40 and mask | Absent, batching ML | Mask semantics #21/#31; defer fixed size to supported shapes |
| T5 | Context-only normalization | Binding standard, implementation absent | Retain #21 and #40 |
| T6 | Row tokens, context labels and context-only attention | Absent, model ML | Reference candidate #31; alternatives #35 |
| T6 | Classification logits and absent-class mask | Absent, model ML | #31 basic behavior; #35 class-range alternatives |
| T6 | Regression bar logits and shared backbone/two heads | Absent, model ML | Candidate head design #31/#35 |
| T6 | Context permutation invariance and no query-label access | Absent, model ML | #31 retains demonstrable invariance/isolation; tolerance agreed there |
| T7 | AdamW, one-cycle, clipping | Absent, training ML | #32 optimizer choice and finite updates; specific schedule remains candidate |
| T7 | JSONL losses, steps, wall limit, runtime record | Absent, training ML/Files | #32 bounded execution and logs; #34 durable record |
| T7 | Checkpoint and resumed trajectory | Absent, training ML | #32 retains state and next-task continuity |
| T7 | Tiny config, 150 steps, 10% loss decline, under 60 seconds | Make target exists; config/training absent | #32 defines measured bounded smoke and oracle criteria; numeric recipe deferred |
| T7 | smoke in CI | Future Make wiring | #32 integrates smoke once implemented |
| T8 | Freeze suite and protocol hash | Draft ADR 0003; suite absent, Files | Human-owned #25 single-table and #41 relational |
| T8 | CODEOWNERS and human review | Ownership file exists; actual protocol approval absent | Preserve ownership; record approval with #25/#41 |
| T8 | Hash verification | Protocol absent | #25/#41 demonstrate manifest identity and mismatch rejection |
| T9 | Fetch/cache real data and write lock checksums | Absent, adapter ML/Files | #26 single-table, #41 relational; provisioning follows guarded rules |
| T9 | Offline loading, missing/corrupt cache behavior | Absent, adapter ML | Retain #26/#41 |
| T9 | RelBench fixture and conversion round-trip | Absent, adapter ML | Retain #41 |
| T10 | Common classifier interface and base-rate/logistic/LightGBM | Absent, evaluation ML | #26 includes regression equivalents; #41 relational integration |
| T10 | Deterministic predictions and protocol-sourced settings | Absent, evaluation ML | Retain #26/#41 |
| T10 | LightGBM full-training reference | Absent, evaluation ML | #41 explicitly distinguishes reference data access |
| T11 | Suite × shots × seeds × classifiers | Absent, evaluation ML | #26/#41 choose and record grid |
| T11 | Record protocol, code, runtime | Absent, Files | #34 |
| T11 | Repeatable fixture results and context-only fit | Absent, evaluation ML | #26/#41 walkthrough with saved row identities |
| T11 | AUROC agreement and offline dev suite | Absent, evaluation ML | #26/#41 independently recompute metrics and demonstrate offline use |
| T12 | Checkpoint wrapper and query chunks | Absent, model/evaluation ML | #31 behavior, #38 and #42 public checkpoint interface |
| T12 | eval-smoke and chunk equivalence | Make target exists; execution absent | #32 and #26 integrate bounded evaluation; #31 establishes chunk assumptions |
| T13 | Record, append ledger, summary | Absent, leaderboard ML/Files | #34 |
| T13 | best.json and protocol-scoped ratchet | Absent, Files; protected path | #34 human-owned protected-path work; distinguish exploratory losses from release regression policy |
| T13 | Append-only rows and complete ledger columns | Absent, Files | #34 schema and inspectable provenance |
| T14 | First guarded run, config and run identity | Caller exists; run/config evidence absent | #37 scale review then #38 single-table milestone; #42 relational milestone |
| T14 | Results PR, ledger, leaderboard and checkpoint release | Policy in ADR 0009; artifacts absent | #34 records; #38/#42 releases |
| T15 | GraphSAGE with PyG/PyTorch Frame optional dependencies | Absent, ML and pyproject dependency list | Candidate full-data reference #41; dependency authority specified there |
| T15 | Tiny fixture and labeled full-training reference results | Absent, evaluation ML | #41 fixture, #42 separately identified reference evidence |
| T16 | GP prior and exact posterior | Absent, priors/model ML | #22 exact arithmetic, #27 generator |
| T16 | Tiny learner trained by shared loop | Absent, training ML | #32 |
| T16 | Held-out log-likelihood gap versus posterior | Absent, ML | #32 retains comparison; threshold chosen before run |
| T16 | Broken attention mask invalidates outcome | Absent, ML | #32 counterfactual; #24 independently authored challenges |
| T16 | Under-two-minute CPU gate before GPU work | Absent, Make/Files | #32 sets machine-specific bound; #37 requires correctness evidence before scaling |
| T17 | SCM graph, neural/tree/GP/linear mechanisms | Absent, priors ML | #27 smooth/linear, #28 SCM/neural, #29 trees |
| T17 | Categorical features, missingness, filters, weights | Absent, priors ML | #30; #23 diagnoses realized distributions |
| T17 | Replay, caps and checklist coverage | Absent, priors ML | #21 contract, #23 diagnostics, #30 mixture audit |
| T18 | Ten-class masked output | Absent, model ML | #31 supported range chosen during grilling |
| T18 | Prior-derived bucket borders, saved borders, inverse scaling | Absent, model ML | Candidate distributional head #35; #31 basic regression units; use training-derived borders if selected |
| T18 | ECOC above ten classes | Absent, model ML | Candidate many-class strategy #35 |
| T19 | Fixed OpenML dev tasks and TabArena exclusion | Absent, Files | #25 selects datasets and checks overlap; historical task counts remain proposals |
| T19 | Offline cache and TabPFN version/weights | Absent, evaluation ML | #26 adapter; #25 pins version and asset terms |
| T19 | TabPFN baseline-only import isolation | Binding AGENTS rule; TabPFN implementation absent | #26 must demonstrate isolation; no teacher signal |
| T20 | Timing across T4/L4/A10/H100 and R1/R3 sizes | Absent, Files | Hardware shortlist #37 after #33 delivery evidence; old grid deferred pending budget |
| T20 | Steps/sec, peak memory, generation wait and ledger | Absent, Files | #33 delivery measurements, #34 ledger, #37 actual pilot evidence |
| T20 | Replace estimated cost with measured throughput | Absent, Files | #37 forecast with observed input and explicit uncertainty |

## Additional acceptance details retained

| Original | Promise | Observed state and evidence | Disposition and owner |
| --- | --- | --- | --- |
| T6 | Changing one query leaves other outputs unchanged; under-five-second check | Absent, model ML | #31 demonstrates query independence for the reference model; #35 declares alternate assumptions; measured CPU bound set there |
| T7 | Width 32, two layers, batch eight; 10+10 resume equals 20; reject over-cap configs | Absent, ML/Files | Numeric preset deferred to #32; retain resumed comparison and enforce caps in #32/#34 |
| T9 | Network-disabled checks and cache excluded from commits | pytest socket configuration and .gitignore are policy/configuration evidence; loader absent | #41 retains offline walkthrough and cache exclusion; real provisioning uses guarded lane |
| T10 | Probabilities bounded in [0,1] | Absent, evaluation ML | #26 retains valid probability output evidence |
| T12 | Smoke loss and evaluation table in PR | Required in AGENTS once capabilities land | #32 integrates evidence with #26; #38 release walkthrough |
| T13 | Ratchet integrated into make check | Make comment only | #34 owns human-reviewed regression policy and integration |
| T15 | Fixture reference under one minute | Absent, ML | #41 selects a bounded fixture workload and measured time limit |
| T17 | 1,000-task AUROC/R², balance and missingness report; prior-report command | Absent, priors ML/CLI | #23/#30 retain diagnostic distributions; count and CLI chosen with measurable cost before execution |
| T17 | Discretization and MCAR missingness | Absent, priors ML | #28/#30 include these candidates alongside other observation mechanisms |
| T18 | ECOC equals direct oracle on separable 30-class case | Absent, model ML | Conditional acceptance experiment in #35 if ECOC selected; class count chosen there |
| T19 | Package/weight versions per run and protocol hash | Absent, Files | #26/#34 retain both identities and #25 freezes protocol |
| T11 / vocabulary | Historical 64/256/512 context-shot grid | Proposed in pinned CONTEXT.md; no suite | #25/#41 select and freeze shot counts; historical grid remains a candidate |
| T18 | Equal-probability borders on fixed prior sample and inverse z-scoring | Absent, model ML | #35 defines border construction if selected, persists sample identity/borders and demonstrates decoding back to original units |
