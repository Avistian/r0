"""Valid nearconstant finite task summaries must remain inspectable without retries."""

import json
from pathlib import Path

from r0.modules.explorer import ExplorerSession


def test_guarded_multiclass_construct_with_nearconstant_entropy(tmp_path: Path):
    session = ExplorerSession(tmp_path)
    controls = session.request("construction_examples", {})["examples"][
        "Topology hyperprior: authored linear"
    ]
    controls["request"]["topology"]["nodes"] = 5
    controls["request"]["task"].update(
        task_type="multiclass",
        classes=3,
        noise_ratio=0,
        separation=2.1,
        class_offsets=[-0.4, 0.2, 0.9],
    )
    response = session.request(
        "construct",
        dict(
            controls=controls,
            construction_seed=79,
            seed=36028797018964015,
            task_id=36028797018964017,
            causal_limits=controls["request"]["limits"],
            max_seconds=60,
        ),
    )
    assert session.task is not None
    json.dumps(response, allow_nan=False)
    view = response["view"]
    entropy = view["classification"]["entropy"]
    assert sum(entropy["histogram"]["counts"]) == 16


def test_literal_finite_histogram_counts_without_changing_values():
    from dataclasses import replace

    import numpy as np
    import torch

    from r0.modules.explorer import inspect_task
    from r0.modules.priors import LinearRecipe, generate_task

    one_up = float(np.nextafter(1.0, 2.0))
    two_up = float(np.nextafter(one_up, 2.0))
    tiny = float(np.nextafter(0.0, 1.0))
    largest = float(np.finfo(np.float64).max)
    fixtures = [
        ([3.0] * 4, [4]),
        ([0.0] * 4, [4]),
        ([1.0, 1.0, one_up, one_up], [4]),
        ([1.0, 1.0, two_up, two_up], [2, 2]),
        ([-one_up, -one_up, -1.0, -1.0], [4]),
        ([0.0, 0.0, tiny, tiny], [4]),
        ([-largest, -largest, largest, largest], [2] + [0] * 18 + [2]),
    ]
    original = generate_task(
        LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1),
        seed=11,
        task_id=13,
    )
    for values, expected in fixtures:
        values = values * 4
        expected = [count * 4 for count in expected]
        inputs = torch.tensor(values, dtype=torch.float64)[:, None]
        task = replace(
            original,
            inputs=inputs,
            scores=torch.zeros_like(inputs),
            targets=torch.zeros(16, dtype=torch.float64),
        )
        view = inspect_task(task)
        summary = view["features"][0]["summary"]
        assert summary["histogram"]["counts"] == expected
        assert summary["min"] == min(values)
        assert summary["max"] == max(values)
        assert summary["histogram"]["edges"][0] == min(values)
        assert summary["histogram"]["edges"][-1] == max(values)
        assert sum(summary["histogram"]["counts"]) == 16
        json.dumps(view, allow_nan=False)
        assert torch.equal(task.inputs[:, 0], torch.tensor(values, dtype=torch.float64))
