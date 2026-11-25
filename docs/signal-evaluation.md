# Evaluate a saved analytic signal

`evaluate_signal(task, inputs)` evaluates the sampled function on a new grid without
sampling a new mechanism, drawing observation noise or labels, or refitting any
normalization statistics. It returns normalized latent scores, one column per score
output, on the input tensor's device. The input is a finite float64 CPU/CUDA tensor
with 1–20,000 rows and the same columns, in the same order, as the generated task.

```python
import torch
from r0.modules.priors import PolynomialRecipe, evaluate_signal, generate_task

task = generate_task(PolynomialRecipe(), seed=27, task_id=3)
grid = task.inputs[:1].repeat(101, 1)
grid[:, 0] = torch.linspace(-3, 3, 101, dtype=torch.float64)
curve = evaluate_signal(task, grid)
```

Supported mechanisms are dense/sparse linear, grouped linear, additive polynomial,
quadratic, random Fourier features, periodic functions, and additive/ridge sinusoidal
or radial functions. Generation and inspection share their task-score calculation.
Calibration retains its existing numerical backend and independent input stream.
Chunking bounds large Fourier/smooth intermediate matrices to 256 rows.

Scores are diagnostic truth, not model inputs. The evaluator never reads observed
targets. It does not apply classification separation/offsets or convert scores to
probabilities; those remain the target adapter's responsibility. Changing input rows
or their order preserves the sampled parameters and calibration statistics.

Direct GP inspection is deliberately explicit: this analytic evaluator rejects a GP
task. Values at new inputs require distinguishing a conditional mean from additional
conditional samples; neither is an already observed truth value. That inspection
interface remains pending in the broader issue #27 explorer work.

Invalid grids and nonfinite results raise a visible error. No values are silently
clipped, no family is substituted and no task is retried.

## Compatibility procedure

`scripts/signal_compatibility.py` constructs 32 fixed tasks: eight analytic families
across all three target types, plus 513-row tasks crossing chunk boundaries. It hashes
serialized recipe/mechanism records and raw bytes of inputs, scores, targets,
probabilities and split indices, and retains source-task and episode identifiers.

Before centralizing evaluation, running this program against source
`8337bdeb873bb5dee92ab148825686ed96540524` produced
`/tmp/r0-signal-before.json`. The same cases after the change must agree exactly:

```sh
UV_PROJECT_ENVIRONMENT=/absolute/path/to/existing/.venv \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 scripts/run_check.py python scripts/signal_compatibility.py > /tmp/r0-signal-after.json
```

Compare the complete `cases` arrays, excluding only the outer source revision. To
reproduce the baseline, copy this script outside the checkout and run it through the
baseline checkout's `scripts/run_check.py`, which selects that checkout's source.
No environment installation or editable-metadata changes are needed. These temporary
JSON files are execution artifacts, not durable published outputs. This compatibility
check does not establish GPU acceptance, explorer interaction or full #27 completion.
