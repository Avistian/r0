# Original roadmap excerpts

Source supplied by the project owner. These historical proposals are reproduced for auditability; dispositions live in the reconciliation.

## T11

T11 Evaluation harness: suite × k × seeds × classifiers → run record JSON (with protocol hash, git SHA, minutes) AFK 9, 10 Two runs on the fixture give byte-identical JSON (timestamps excluded); the spy classifier proves fit sees only context rows; the AUROC equals relbench 's on fixture predictions; the dev suite runs offline Determinism, spy and metric-equality tests

## T12

T12 RPFN classifier wrapper around a checkpoint; queries in chunks (exact, since queries are independent); make eval-smoke on the dev fixture in checks.yml AFK 7, 11 A make smoke checkpoint runs through the harness; results identical across chunk sizes Chunking test; make eval-smoke table in the PR body

## T13

T13 Leaderboard, ledger and ratchet: r0 leaderboard record , r0 ledger append , r0 summary , scripts/check_ratchet.py ; ratchet added to make check AFK 11, 12 best.json equals the best leaderboard row for its hash; a PR lowering it fails; rows are append-only; ledger rows have every column of factory-setup 5.8 Tests on fixture CSVs; make ratchet

## T14

T14 First real run via the train lane: Config: configs/runs/v0-cpu.toml , Run: v0-001 in the body agent:train 13 Results PR from the workflow with run record, leaderboard row, ledger row and best.json ; checkpoint as an artifact (a Release once you tag v0.1) The workflow's CPU smoke, then the run; you review the ledger and metrics, then merge

## T15

T15 RDL reference baseline: RelBench's example GraphSAGE (PyG + PyTorch Frame as an optional gnn extra) on suite_v0's tasks, full train table, run through the train lane AFK then agent:train 11 The code path runs on the fixture in under a minute on CPU; the real run's results land as "reference" rows Fixture test; real numbers only via the train lane

## T16

T16 GP sanity prior and exact-posterior check (stage R0): a 1-D GP prior, a tiny model trained by the same loop, and the exact GP posterior predictive as reference AFK 7 On fixed held-out GP tasks, the model's mean log-likelihood is within a threshold of the exact posterior's Verify ; a deliberately broken attention mask makes the test fail CPU test under 2 minutes; the mask-mutation test

## T17

T17 Single-table prior core: sample_table_task(rng, spec) with SCM graph, MLP / tree-ensemble / discretisation / GP / linear edge functions, categoricals, MCAR missingness, a filter, and weights per component (ADR 0012) AFK 3 Same seed → identical task; caps respected; every item on the specs/prior.md checklist has a test that the prior can produce it; a 1,000-task report (LightGBM AUROC/R² histogram, class balance, missingness) is written to reports/ Property tests; r0 prior-report --n 1000 on CPU

## T18

T18 Heads for three task types: classification (10 masked logits), bar-distribution regression (borders from prior targets, saved with the checkpoint), and an ECOC wrapper above 10 classes AFK 6 Bucket borders are equal-probability on a fixed prior sample; regression predictions invert z-scoring correctly (round-trip test); ECOC decodes a 30-class toy problem with the same accuracy as a direct 30-way oracle on separable data Unit tests on CPU

## T19

T19 Tabular dev track and TabPFN v2 reference: eval/tabular_dev_v0.toml (4.2), an OpenML adapter with an offline cache, and baselines/tabpfn.py pinned to v2 weights; an import-linter contract keeps TabPFN out of train/ and priors/ (ADR 0011) HITL 8, 10 You pick the task ids; none is in TabArena's 51 (name, shape and hash check); the protocol hash is pinned like suite_v0's; the TabPFN wrapper records package and weight versions on every run record You review line by line. GPU TabPFN runs only on Modal, started by you

## T20

T20 Modal GPU timing smoke: one short run per GPU type (T4, L4, A10, H100) at R1 and R3 model sizes, logging steps per second, peak memory and prior-generation time vs GPU step time agent:train 7, 13 A ledger row per GPU type and a table in reports/throughput.md that turns 2.18's cost estimates into measured ones The train lane's caps; you check the ledger before approving R1
