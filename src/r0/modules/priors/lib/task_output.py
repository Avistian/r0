"""Complete a mechanism's latent scores with shared targets, split and provenance."""

import platform
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import torch

from r0.modules.priors.lib.linear import LinearRecipe
from r0.modules.priors.lib.representatives import encode, record_fingerprint
from r0.modules.priors.lib.targets import adapt_targets
from r0.modules.priors.lib.training_task import TrainingTask


def complete_task(
    task: LinearRecipe,
    config: dict[str, object],
    mechanism: object,
    inputs: torch.Tensor,
    scores: torch.Tensor,
    *,
    seed: int,
    task_id: int,
    rng: Callable[..., np.random.Generator],
    source_config: dict[str, object] | None = None,
) -> TrainingTask:
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite latent function; task was not retried")
    device = inputs.device.type
    targets, probabilities, classes = adapt_targets(
        task, scores, noise_rng=rng("noise"), label_rng=rng("labels")
    )
    permutation = torch.tensor(rng("split", task.split).permutation(task.rows), device=device)
    source_config = {
        **(config if source_config is None else source_config),
        "task": {k: v for k, v in asdict(task).items() if k not in ("context_rows", "split")},
    }
    source_id = record_fingerprint((source_config, seed, task_id))
    return TrainingTask(
        source_id,
        record_fingerprint((source_id, task.split, task.context_rows)),
        task.task_type,
        encode(
            {
                "recipe": config,
                "seed": seed,
                "task_id": task_id,
                "execution": {
                    "implementation": config["version"],
                    "device": device,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": str(torch.__version__),
                    "cpu_threads": torch.get_num_threads(),
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                },
            }
        ),
        encode(mechanism),
        inputs,
        targets,
        scores,
        probabilities,
        permutation[: task.context_rows],
        permutation[task.context_rows :],
        classes,
    )
