import json
import subprocess

import numpy as np
import torch

from r0.modules.priors import (
    GaussianProcessRecipe,
    LinearRecipe,
    StationaryKernel,
    condition_gp,
    generate_task,
)

recipe = GaussianProcessRecipe(
    task=LinearRecipe(
        rows=16,
        context_rows=8,
        features=3,
        signal_features=2,
        active_features=2,
        task_type="multiclass",
        classes=3,
    ),
    calibration_rows=32,
    kernels=(
        StationaryKernel(kind="matern12", length_scale=0.7, weight=1),
        StationaryKernel(length_scale=1.3, weight=3),
    ),
)
task = generate_task(recipe, seed=918, task_id=2)
card = json.loads(task.mechanism_json)
support = np.asarray(card["support"])
query = torch.tensor(
    [[0.17, -0.82, 11.0], [-0.3, 0.6, 0.0], [0.17, -0.82, -91.0]], dtype=torch.float64
)
points = query.numpy()[:, card["active"]]


def kernel(a, b):
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return 0.25 * np.exp(-np.sqrt(d2) / 0.7) + 0.75 * np.exp(-d2 / (2 * 1.3**2))


k = kernel(support, support)
w, v = np.linalg.eigh(k)
raw = ((v * np.sqrt(np.maximum(w, 0))) @ v.T) @ np.asarray(card["standard_normal"])
cross = kernel(points, support)
expected_mean = (cross @ np.linalg.solve(k, raw) - card["center"]) / card["scale"]
expected_cov = kernel(points, points) - cross @ np.linalg.solve(k, cross.T)
result = condition_gp(task, query, draws=4, seed=41)
mean_error = np.max(np.abs(np.asarray(result["mean"]) - expected_mean))
cov_error = np.max(np.abs(np.asarray(result["raw_covariance"]) - expected_cov))
std_error = np.max(
    np.abs(
        np.asarray(result["standard_deviation"])
        - np.sqrt(np.maximum(np.diag(expected_cov), 0))[:, None] / card["scale"]
    )
)
assert max(mean_error, cov_error, std_error) < 1e-10
assert result == condition_gp(task, query, draws=4, seed=41)
assert np.array_equal(np.asarray(result["draws"])[:, 0], np.asarray(result["draws"])[:, 2])
reordered = condition_gp(task, query[[2, 0, 1]], draws=4, seed=41)
assert np.array_equal(np.asarray(reordered["draws"])[:, [1, 2, 0]], np.asarray(result["draws"]))
saved = condition_gp(task, task.inputs[:3], draws=4, seed=41)
saved_error = np.max(np.abs(np.asarray(saved["mean"]) - task.scores[:3].numpy()))
assert saved_error < 1e-10
assert np.max(np.asarray(saved["raw_covariance"])) == 0
print(
    json.dumps(
        {
            "recipe": "weighted matern12(0.7,1)+rbf(1.3,3), 48 support, 3 channels",
            "seed": 918,
            "task_id": 2,
            "query": query.tolist(),
            "inspection_seed": 41,
            "mean_max_error": float(mean_error),
            "covariance_max_error": float(cov_error),
            "std_max_error": float(std_error),
            "saved_max_error": float(saved_error),
            "replay_and_reorder": "PASS",
            "duplicates": "PASS",
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        indent=2,
    )
)
