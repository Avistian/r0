# Generator coverage and remaining acceptance

This is the coverage map for the [accepted #27 specification](specs/0027-training-ready-generators.md).
An implemented range, a generated example and a repeated resource measurement are
different kinds of evidence. None alone establishes model quality or execution of
100 million tasks. The tables below preserve those distinctions.

## Mechanisms and supported inputs

All families return the same training-task contract through `generate_task`, with
regression, binary and multiclass target adapters. Hidden scores, probabilities and
mechanism parameters stay outside predictor inputs. Sampling distributions and
reference adaptations are defined in [sampling profiles](sampling-profiles.md).

| Family | Mechanism controls | Input distributions | Important exclusions |
| --- | --- | --- | --- |
| Linear | Dense or sparse normal coefficients; exact unit expected signal variance | Standard Gaussian | Other inputs use an explicitly named grouped or polynomial recipe |
| Grouped linear | Equal non-overlapping groups, sizes 1–256; dense or group-sparse | Gaussian, uniform, Student-t, two-Gaussian mixture | Signal and active counts must be divisible by group size |
| Polynomial | Additive integer powers, degree 1–16, sampled coefficients | All four | Unbounded Student-t needs `df > 2 × degree` |
| Quadratic | 1–256 factors, separate linear and off-diagonal interaction strengths | All four | Unbounded Student-t needs `df > 4`; factor count is not guaranteed matrix rank |
| RFF | 1–4,096 finite Fourier terms, length scale, sampled frequencies/phases | Standard Gaussian | Explicit finite approximation; not an exact GP draw |
| Periodic | 1–64 terms, feature frequencies and harmonics | Standard Gaussian | At most 64 active features; at least one term per active feature |
| Radial | Additive or weighted-feature ridge functions; 1–4,096 terms | All four | Additive projection needs at least one term per active feature |
| Sinusoidal | Additive or weighted-feature ridge functions; 1–4,096 terms | All four | Same additive term-count constraint |
| Exact GP | RBF, three Matérn kernels, rational quadratic, spectral and Gibbs; 1–64 kernel components; sums/products, subsets and feature scales | All four | At most 1,024 task rows and 512 calibration rows; CPU factorization remains CPU work even with CUDA task output |
| Composition | 1–64 analytic components, sums/products, signed strengths, subsets, repeated leaves; depth at most 8 | All four, subject to child compatibility | Exact GP children excluded; at most 64 leaf occurrences; parent and children share input distribution and score type |

Grouped linear requires `df > 2` for unbounded Student-t inputs. Composition moment
requirements follow the expression's growth degree; sums take the maximum and
products add degrees. Explicit bounded input distributions use rejection sampling,
not clipping. Bounds, location, scale, degrees of freedom and mixture gap remain in
the recipe. Finite required moments do not guarantee accurate finite calibration.
Plain linear has analytic normalization; other families use independent calibration
draws, never context/query targets. Valid weak signals and missing context classes
are retained; nonfinite or numerically unresolved functions fail visibly.

## Diversity axes and evidence boundaries

| Axis | Supported controls | Evidence and limits |
| --- | --- | --- |
| Shape | 16–20,000 rows, 1–256 features; explicit context/query sizes | Fixed matrix measures three shapes; separate family walkthroughs include lower boundaries. Exact GP uses its smaller limit |
| Sparsity | 1 through the candidate signal count; independent nuisance columns | Dense/sparse/group-sparse examples, inactive-feature checks and controlled-pair evidence. Group/axis compatibility is validated |
| Signal strength | Unit expected variance for Gaussian linear; independent-calibration unit-variance estimate otherwise; relative component and interaction strengths | Global signal variance is normalized, not an independently varied absolute-scale axis. Sampled composition strengths and interaction orders are reported separately from depth and component count |
| Regression noise | Finite nonnegative noise-to-signal standard-deviation ratio; independent Gaussian noise | Fixed resource matrix uses 0.5; sampled matrix uses 0, 0.5 and 2. Controlled pairs preserve the underlying function and noise stream |
| Classification difficulty | Nonnegative separation; finite class offsets; 2–100 classes | Fixed matrix includes 100 classes; sampled matrix includes weak separation and one-row contexts. Counts are observed, never forced |
| Input location/scale | Configurable for the four-distribution families | One-off family walkthroughs cover all four; broad-profile selected and completed histograms must be distinguished |
| Function complexity | Degrees, factors, terms, component counts, depth, kernel geometry | One-off maximum cases and profile-settings histograms exist. They do not validate all maxima simultaneously |
| Composed functions | Common inputs, mapped component axes, sum/product expression and separate parent calibration | Standalone family results are separate from the five composition walkthrough cases and sampled compositions |
| Device | Explicit CPU or CUDA; no silent fallback | Current retained Modal agreement is the earlier small linear smoke only. Broader family agreement remains unverified |

The [fixed CPU matrix](generator-resource-results.md), measured at
`3dd2d3b42ee9d3012e6dafa1d0c3a78764efc364`, passed 270 tasks across 90 cases.
It uses fixed Gaussian-input settings and three independent task identities per
case. Its separate author-run artifact audit verified all saved tasks and summary
calculations. This does not replace independent mathematical checks.

The [sampled matrix protocol](generator-diversity-qualification.md) declares 270
broad-profile and 108 adapted-reference draws. Measurement results and completed
distribution counts are pending; settings selected before generation are not yet
successful-task evidence. These control strata vary several properties together,
so they are not controlled difficulty comparisons.

After a suite finishes, audit its selections and saved tasks without generating:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/report_diversity.py \
  docs/evidence/issue-27-broad-qualification \
  --output docs/evidence/issue-27-broad-qualification/diversity-audit.json
```

Use the reference-suite directory for its separate audit. The output must be new.
This auditor requires a fully finished suite, including terminal failed/timed-out
cases. For an interrupted suite, use `qualify_generators.py --summarize` to retain
INCOMPLETE, LAUNCH_UNKNOWN and NOT_RUN states; the artifact auditor does not treat
missing records as verified failures or successful tasks. An audit PASS means its
checks succeeded, not that all generation attempts passed. Outcome counts remain
visible. This is an author-run artifact audit, not an independent formula oracle.

Histogram bins distinguish `SELECTED`, `GENERATION_SAVED` and the worker's terminal
outcome. A saved task can still have a verification timeout: generation publishes
its complete save result before replay/graph checks. `confirmed_saved_tasks` counts
those recorded completed saves; `verified_source_tasks` counts only tasks whose
worker checks passed and whose archive this auditor then checked. Missing save
confirmation is not proof that no file was written. Saved-but-unverified cases
remain in the failure list and never become PASS merely because an archive exists.

Earlier sampling evidence at `a9490e120e5f0066222bf9a2d46a26f440e985a7`
contains 1,792 settings selections and 42 generated examples. Settings include every
polynomial degree, every listed RFF term count, 1/64 components and actual expression
depths 0–8. Only the 42 examples were generated in that walkthrough. Depth zero
denotes a single component; it is not a nested composition.

## Worked mathematical checks and extreme cases

The programs below retain requests, sampled functions, selected-row calculations,
replay/save/resume outcomes and resource measurements. Most family examples are
single runs rather than repeated throughput estimates. Raw evidence stays in the
ignored local `docs/evidence/` directory; the committed programs permit reproduction.

| Evidence program | Distinct coverage |
| --- | --- |
| `scripts/linear_walkthrough.py` | Scalar dot product, noise pair, dense/sparse cases and maximum analytic shape |
| `scripts/grouped_linear_walkthrough.py` | Whole-group activation, scalar sum, four input distributions and one 256-feature group |
| `scripts/polynomial_walkthrough.py` | Scalar powers, four input distributions, bounded degree-16 example and maximum rows/features |
| `scripts/quadratic_walkthrough.py` | Independent pairwise matrix expansion, interaction control, four inputs and 256-factor maximum-shape example |
| `scripts/fourier_walkthrough.py` | Scalar cosine/frequency calculation, RFF and periodic examples, repeated local timings |
| `scripts/smooth_walkthrough.py` | Radial/sinusoidal × additive/ridge, four inputs, 4,096-term and maximum-shape cases |
| `scripts/exact_gp_walkthrough.py` | Independent covariance and square-root calculation; base, `--extended` and `--nonstationary` cases; declared numerical failures |
| `scripts/composition_walkthrough.py` | Sum of all eight analytic representatives, subset product, depth-8 repeated leaf, 64 components and maximum-shape composition |
| `scripts/controlled_pair_walkthrough.py` | Checked fixed properties, measured signal/noise changes and context-only least-squares fit with query-only scoring |
| `scripts/generation_graph_walkthrough.py` | 35 graph reconstruction cases across mechanisms; intermediate values reconstructed from actual provenance |
| `scripts/consume_generated_batches.py` and `scripts/verify_generated_batches.py` | Common episode consumption, bounded batches, saved-group input and independent NumPy/finite-difference optimizer checks |

Composition walkthrough source `844c3fce8fa09bfc8fe0449e72d5802670221688`
retains five successful cases and one intentional underflow failure. Its independent
scalar calculations use the saved normalizers; they do not independently prove
population variance. GP covariance calculations similarly establish their stated
small-case arithmetic and tolerance, not all kernels at every numerical extreme.

## Bulk and reference eligibility

Exact GP is a **reference generator**, including its kernel sums/products. RFF and
analytic smooth functions keep their own names; no runner replaces a GP with them.
The nine analytic family entries are **bulk candidates**, not blanket approval of
every supported setting. The fixed resource table supplies measured local CPU
workloads. Broad/reference sampling profiles remain exploration distributions;
unsupported or failing settings stay visible instead of being silently filtered.

A bulk recipe must explicitly select its family, concrete settings or versioned
profile, task geometry and resource limits. It must cite numerical/resource evidence
for that selection and its target device. No default production mixture or whole
profile is certified here; mixture weights and training-stream selection remain #30.
The present GPU and sampled-profile gaps prevent a final cross-device eligibility
claim. The 100-million-task figures in the resource report are extrapolations,
not observed execution or an estimate for a chosen production mix.

## Delivery and outstanding acceptance

| Requirement | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Shared generation, cards and replay | Public recipes, common task/episode contract, saved selected recipes and deterministic identities | Final integrated-head audit |
| Bounded training consumption | 30-task and 90-task optimizer walkthroughs; saved-group consumption with generation disabled | Full foundation-model training remains separate |
| Local create/save/explore | Actual browser checks, full-task summaries, sampled chart rows, individual-row inspection, graphs, curves, paired views and ZIP/SVG/PNG exports | Final integrated-head verification |
| Colab notebook and embedded explorer | Pinned install, local execution of generation/estimate cells, tested callback adapter and shared browser UI | Actual hosted install, interaction/download, Drive persistence and runtime restart are NOT_RUN |
| Modal generation and durable restart | Earlier CPU/T4 smoke with separate-container resume; both generated eight tasks and resumed with zero new generations | Broader family/device coverage and current sampled request path on Modal |
| Resource and diversity evidence | Fixed 270-task matrix, one-off extremes, profile selection histograms and mathematical walkthroughs | Sampled matrix results, explicit final eligibility and broader GPU checks |
| Published-capability comparison | Separate source-linked comparison PR #117 | Integration with the remaining PRs |
| Local evidence policy | `docs/evidence/` ignored; raw files removed from current Git tree | History was intentionally not rewritten |

Matching LimiX-2, TabPFN-3 or TabICLv2 is an eventual trained-model goal. Generator
correctness, varied tables and successful execution cannot establish that result.
Issue #27 remains open until its generator, UI, runtime and evidence requirements
are actually verified; this document does not redefine completion around available
local hardware.
