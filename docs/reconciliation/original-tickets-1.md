# Original roadmap excerpts

Source supplied by the project owner. These historical proposals are reproduced for auditability; dispositions live in the reconciliation.

## T1

T1 Walking skeleton: r0 --version , subcommand stubs ( train , eval , data , leaderboard , ledger , summary ), TOML config → frozen dataclasses, seed_everything AFK — CLI prints the version; configs round-trip; unknown keys are an error make check ; CLI test via subprocess

## T2

T2 reldb : Table , RelDB , TaskRows ; validate() AFK 1 Duplicate PKs, dangling FKs and untyped time columns are rejected with clear messages; a 3-table fixture validates Unit + Hypothesis tests

## T3

T3 Synthetic relational prior: sample_task(rng, spec) AFK 2 Output validates; caps respected (entities, event tables, columns); labels depend only on events in (t, t+w]; same seed → identical task; every class present after resampling Property tests over seeds and specs; for the activity and value rules, editing events outside (t, t+w] leaves every label unchanged

## T4

T4 Featurizer DFS-lite v0: featurize(db, rows, spec) AFK 2 Future-rows invariance : appending random events dated at or after each row's seed time leaves every feature byte-identical; ≤ 40 features; NaN only for "no history"; vectorised (1M event rows in under 2 minutes on a hosted runner) Hypothesis property test; a timing test marked slow run in the train lane Verify the budget once

## T5

T5 Batching: make_batch : context from earlier seed times, queries from the latest; pad to 40 + mask; context-only z-scoring AFK 3, 4 Context and query sets disjoint; every context seed time < every query seed time; normalisation stats equal those of the context rows alone Unit + property tests

## T6

T6 Model RPFN v0: rows as tokens, label embedding on context, "attend to context only" mask AFK 1 Two heads on one backbone (v4, guide 2.19): classification logits [batch, query, 10] with absent classes masked, and regression bar logits [batch, query, n_buckets]; the head is chosen by task type. Context-permutation invariance (tol 1e-5); changing one query row changes no other query's output; query labels are not an input Invariance tests on CPU, under 5 s

## T7

T7 Training loop: AdamW, one-cycle, clipping, JSONL loss log, max_steps / max_wall_minutes , checkpoint + resume, wall-clock minutes in the run record; configs/ci/smoke.toml ; make smoke in checks.yml AFK 5, 6 Smoke (width 32, 2 layers, 150 steps, batch 8) runs under 60 s on a hosted runner with loss falling ≥ 10%; 10+10 steps with resume equals 20 steps; configs over the caps in configs/ci/ are rejected make smoke ; resume test. Verify the 10% and 60 s once on your runner

## T8

T8 Freeze the evaluation protocol: eval/suite_v0.toml , ADR 0003 with its hash, hash test, CODEOWNERS HITL 1 Merged through your local-lane PR; hash test green You review the protocol line by line

## T9

T9 relbench_adapter : r0 data fetches the suite into .cache/ , converts to RelDB + task rows, writes eval/suite_v0.lock.json (checksums only); offline load() ; a tiny committed fixture database in RelBench's layout for tests AFK 2, 8 Loader fails clearly when the cache is missing or a checksum differs; conversion round-trips the fixture; no network in tests; nothing under .cache/ is committed Unit tests with sockets disabled. make data itself runs only in the train lane

## T10

T10 Baselines behind one Classifier protocol ( fit , predict_proba ): base rate, logistic regression, LightGBM-k, LightGBM-full, with ADR 0003's hyperparameters AFK 4 Probabilities in [0, 1]; deterministic per seed; hyperparameters read from the suite file, not hard-coded Tests on fixture features
