"""Original task-adapter target stages for retained-row generation traces."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch


def target_steps(
    controls: dict[str, Any],
    signal: torch.Tensor,
    indices: Sequence[int],
    rng: Callable[[str], np.random.Generator],
    add: Callable[..., None],
) -> torch.Tensor:
    if controls["task_type"] == "regression":
        noise = torch.tensor(rng("noise").normal(size=controls["rows"])[list(indices)])
        add(
            "noise",
            "Replayed observation noise",
            [],
            "noise_ratio * original standard normal draw",
            noise * controls["noise_ratio"],
            {"noise_ratio": controls["noise_ratio"], "stream": "noise"},
        )
        targets = signal[:, 0] + controls["noise_ratio"] * noise
        add(
            "target",
            "Regression target",
            ["signal", "noise"],
            "signal + observation noise",
            targets,
            {},
        )
        return targets
    logits = controls["separation"] * signal + torch.tensor(
        controls["class_offsets"] or [0.0] * signal.shape[1], dtype=torch.float64
    )
    add(
        "logits",
        "Class separation and offsets",
        ["signal"],
        "separation * signal + class_offsets",
        logits,
        {"separation": controls["separation"], "class_offsets": controls["class_offsets"]},
    )
    if controls["task_type"] == "binary":
        positive = torch.sigmoid(logits[:, 0])
        probabilities = torch.stack((1 - positive, positive), dim=1)
        formula = "[1 - sigmoid(logit), sigmoid(logit)]"
    else:
        probabilities = torch.softmax(logits, dim=1)
        formula = "softmax(logits)"
    add("probabilities", "Class probabilities", ["logits"], formula, probabilities, {})
    uniforms = torch.tensor(rng("labels").uniform(size=controls["rows"])[list(indices)])
    add(
        "labels",
        "Replayed label draws",
        [],
        "original uniform(0,1) draws",
        uniforms,
        {"stream": "labels"},
    )
    targets = torch.sum(uniforms[:, None] >= probabilities.cumsum(dim=1)[:, :-1], dim=1)
    add(
        "target",
        "Sampled class",
        ["probabilities", "labels"],
        "number of cumulative class probabilities <= uniform draw",
        targets,
        {},
    )
    return targets
