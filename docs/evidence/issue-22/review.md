# Issue 22 implementation and review record

Baseline: merged #21, commit 8d21b4cd82b082bfe1f9f4c97b44c0e8348032ec. Implementation used the existing design branch and all 27 accepted decisions in ../../specs/0022-oracles.md. No training, dataset generation, cloud calls or live W&B integration occurred.

## TDD at agreed public seams

- Oracle seam: coin test failed at missing module, then passed with the 4/6 posterior. GP test failed at missing public API, then passed against the scalar linear-GP calculation.
- Replay/configuration/scoring seam: initial test failed at missing replay API, then passed with query-only Brier/AUC and label-isolation checks. A prior-only configuration accepting irrelevant jitter was reproduced as a failing test and corrected.
- CLI seam: oracle command initially failed argument parsing, then created a report and preserved an existing different run record. Deliberately incorrect expected answers retain a FAIL report and nonzero exit.
- Independent-comparison seam: a missing comparison API was demonstrated before implementing numeric difference records.

## Standards axis

Initial review found one P2 numerical reporting defect: finite expected value 1e-320 versus observed 1.0 overflowed relative-error reporting to infinity. JSON serialization then failed and no FAIL report was retained. The reviewer independently reproduced this through write_oracle_report with gp-prior.toml.

Correction uses exact rational arithmetic on supplied numeric values to determine differences, thresholds and verdicts. Unrepresentable displayed differences become null with a range reason. Regression tests reproduced the original failure, then passed after correction, including absolute-difference and allowed-error overflow cases.

The standards reviewer independently repeated the original probe and confirmed valid JSON retained status FAIL, absolute_error 1.0, relative_error null, and reason "exceeds float64 range". Correction delta accepted. No other substantive documented violations or actionable judgment-only smells were reported. The historical finding remains recorded here.

## Specification axis

Initial review found one P2 discrepancy: the public gp_prior API accepted nonzero jitter and returned stabilized=True despite having no conditioning system. The configuration path already rejected the combination, but the direct public API did not.

The regression test first failed because no error was raised. Correction rejects nonzero jitter before prior calculation. The reviewer independently confirmed the original probe now raises the intended ValueError; the zero-jitter RBF prior remains valid with mean 0, latent/observation variance 1, empty conditioning covariance and stabilized=False. Correction delta accepted. The historical finding remains recorded here.

The specification reviewer independently computed:

| Quantity | Independent expected value | Observed absolute difference |
| --- | --- | --- |
| Coin next-head probability | (1+3)/(1+1+4) = 4/6 | 0 against float-rounded rational |
| Multiclass posterior | (1,2,3)+(2,0,0) = (3,2,3) | 0 |
| Multiclass probabilities | (3/8,1/4,3/8) | 0 |
| GP mean | A=2, b=2, y=2; by/A=2 | 2.220446049250313e-16 |
| GP latent variance | 4-b²/A=2 | 4.440892098500626e-16 |
| GP observation variance | 2+1=3 | 4.440892098500626e-16 |
| Tied ROC AUC | (1/2)/1=1/2 | 0 |

All 14 versioned cards also regenerated and passed the reviewer's in-memory comparisons. Later additions of the log-score expected values were calculated by the author from the formulas in README.md and passed the final retained walkthrough. Correction reviews cover their findings, not a fresh independent full-repository acceptance.

## Author verification

Final command: UV_OFFLINE=1 UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src make check.

Result: formatting and lint pass, zero Pyright errors/warnings, all three import contracts kept, module and file-size checks pass, 66 tests pass. Existing offline dependencies were reused with PYTHONPATH selecting this checkout. The sandbox initially prevented the type checker discovering dependency imports; running with access to the existing interpreter resolved that environment failure.

run-01/ retains every reference report with expected/observed values and numeric differences. Its walkthrough.json records the 14 passes, six expected errors, applicable label isolation, context/query reorder and extension checks, and exact repeated replay. Observed wall time was approximately 0.048 seconds, not a portable performance threshold.

Training smoke and evaluation smoke: NOT_RUN; their implementations remain future-ticket work. Hosted CI: NOT_RUN for this local implementation commit. No scientific model-performance or generator-quality acceptance is claimed.
