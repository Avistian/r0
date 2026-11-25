# Linear generation: first implementation slice

This slice implements Gaussian dense/sparse linear tasks for regression, binary
classification and multiclass classification. It does not complete #27. Other
families, run persistence, Colab/Modal and the explorer follow in dependent PRs.

The public interface is `LinearRecipe`, `generate_linear(recipe, seed=..., task_id=...,
device="cpu"|"cuda")` and its common `TrainingTask` tensor result. `task.episode()`
materializes the existing immutable episode contract for current consumers.
Training uses tensors without first converting every cell to a Python object.
Scores/probabilities are diagnostic truth; `episode().predictor_view()` excludes them.

`signal_features` defines the candidate signal columns; `active_features` chooses
how many are active. Columns beyond `signal_features` are independent nuisance
features. Dense tasks select all candidates. Each output's weights are normalized
so the expected signal variance is one under independent standard-normal inputs.
`noise_ratio` is the ratio of noise standard deviation to signal standard deviation.
Classification uses `separation` and `class_offsets`, without resampling missing
classes. Binary has one score/offset; multiclass has one per declared class.

## Reproduce

From this checkout with an existing environment (no dependency sync):

```sh
export UV_PROJECT_ENVIRONMENT=/home/avist/Projects/r0/.venv
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 scripts/run_check.py pytest tests/priors/test_linear_generation.py
python3 scripts/run_check.py python scripts/linear_walkthrough.py --out /tmp/linear-cpu.json
```

On a compatible CUDA environment, repeat the test and walkthrough with `--device cuda`.
The device test requires identical sampled inputs and score agreement within
`rtol=1e-10, atol=1e-12` in float64. CPU/GPU agreement is NOT_RUN in the retained
CPU-only environment; a skipped test is not GPU evidence.

The retained JSON contains a row and sampled coefficients. Its dot product is
recomputed with scalar `math.fsum`, separately from the tensor implementation.
The noise pair checks doubling the same realized perturbations. Timings contain
one untimed warm-up and three measured tasks per shape, including 20,000 × 256.
They exclude episode materialization and export. Peak memory is NOT_MEASURED;
these timings do not establish 100-million-task throughput or bulk eligibility.

The first test run failed to import `LinearRecipe`/`generate_linear`. After the
regression implementation, the classification tests failed because the recipe did
not accept `task_type`. Both failures were observed before their implementations.
