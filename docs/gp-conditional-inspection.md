# Conditional views of a saved Gaussian process

`priors.condition_gp(task, inputs, draws=0, seed=None)` inspects 1–256 float64 CPU
input rows from an exact GP task. It returns normalized latent means, standard
deviations, the raw conditional covariance and optional additional draws. This
is an inspection operation; it neither changes the task nor adds training rows.
The local/Colab UI integration is a separate pending slice of #27.

```python
from r0.modules.priors import condition_gp

mean = condition_gp(task, new_inputs)
samples = condition_gp(task, new_inputs, draws=3, seed=42)
```

The result separates original values from new ones. `original_support_indices`
identifies requests matching the saved support in all active coordinates. Those
values are returned from the original joint draw, with zero uncertainty. Other
rows are conditional means or additional draws, not evaluations of an analytic
function saved at task creation. Irrelevant columns do not affect the GP.

## What is conditioned on

The full saved support includes the independent calibration points and the task
points. Reconstruct its raw noiseless values `f` from the recorded standard normals
and the original symmetric covariance square root. With support covariance `K`,
cross covariance `C` and query covariance `Q`, the conditional raw mean is
`C K^-1 f`, and its covariance is `Q - C K^-1 C.T`. This is the noiseless case of
[GPML chapter 2, equations 2.23–2.24](https://gaussianprocess.org/gpml/chapters/RW2.pdf).
The implementation solves in the support eigenbasis, without forming an inverse.

The saved calibration center and scale remain fixed. Mean values subtract that
center and divide by the scale. Standard deviations divide by the scale; output
channel `c` has normalized covariance `raw_covariance / scale[c]**2`. Latent output
channels remain independent conditional on their saved draws. Observation noise,
class separation, offsets, probabilities and sampled labels are excluded. This is
mechanism inspection using hidden truth, not a prediction from context labels.

## Numerical limits and explicit approximation

Strict inspection rejects support eigenvalues at or below
`64 * float64_epsilon * support_count * max(1, largest_eigenvalue)`. It also rejects
indefinite covariance beyond roundoff tolerance. No jitter, new observation noise,
resampling or mechanism substitution is added to make a solve work.

Smooth kernels on dense support can have numerically unresolved directions. The
caller can explicitly request `allow_rank_truncation=True`. At new points this
conditions only on support eigenmodes above that threshold. It is an **approximate
conditional view**, not the exact posterior given every original support value.
The result records `approximate`, retained rank, threshold and the largest
normalized difference between the original support draw and its retained
projection. At exact saved coordinates the original values still take precedence;
this exception does not remove the approximation between those coordinates.
The default remains strict so approximation cannot happen silently.

Conditional covariance roundoff uses
`64 * float64_epsilon * (support_count + unique_query_count) * max(1, largest_joint_diagonal)`.
Negative conditional eigenvalues within this bound are set to zero, and the largest
correction is recorded. A larger negative eigenvalue fails visibly. Known support
coordinates have exactly zero covariance. Duplicate active query coordinates use
one sampled value, including when their irrelevant columns differ.

## Replay of additional values

Request 1–8 draws with an explicit integer seed in `[0, 2**63)`. Mean-only inspection
accepts no draw seed. A separate `gp-conditional-v1` random namespace includes the
source task identity, sorted unique active query coordinates, inspection seed and
draw index. The result retains the query fingerprint, inputs, seed and method.
Reordering or repeating query coordinates preserves their additional values.
Changing the set of query coordinates changes the additional draw: this is a joint
sample on the requested set, not a persistent sampled function on all inputs.

Inspection runs on CPU using NumPy's symmetric eigensolver. Replay on other
software/hardware can differ within numerical precision. Original task generation
and its existing CPU/GPU contract are unchanged. This bounded reference operation
is not bulk generation or evidence of predictive model performance.
