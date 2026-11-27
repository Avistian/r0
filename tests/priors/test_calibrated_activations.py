"""Independent calibration freezes a function shared by generation and inspection."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from r0.modules import priors


def test_frozen_standard_activation_replays_without_refitting_on_task_rows() -> None:
    activation = priors.calibrate_source_activation(
        priors.CausalActivation(),
        np.array([[-1.0], [1.0]]),
        rng=np.random.default_rng(0),
        scale=2.0,
        calibration_id="independent-worked-two-point-sample",
    )
    controls = priors.LinearRecipe(
        rows=16, context_rows=8, features=1, signal_features=1, active_features=1
    )
    recipe = priors.CausalRecipe(
        task=controls,
        nodes=(
            priors.ActivatedCausalNode(
                bias=(3.0,), noise_scale=(0.0,), output_activation=activation
            ),
            priors.CausalNode(),
        ),
        observed=((1, 0),),
        target=((0, 0),),
        version="causal-v2",
    )
    # For identity on {-1,+1}, centering cancels the sampled shift. The
    # two population-standardization factors reduce to 2/(2+epsilon*(1+epsilon)).
    expected = 6.0 / 2.00010001
    restored = priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe))))
    assert restored == recipe
    for rows in (16, 32):
        current = replace(restored, task=replace(controls, rows=rows, context_rows=rows // 2))
        task = priors.generate_task(current, seed=28, task_id=1)
        torch.testing.assert_close(
            task.targets,
            torch.full((rows,), expected, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        assert priors.explain_generation(task, [0, rows - 1])["target_max_error"] < 1e-12
    assert activation.calibration_rows == 2
    assert activation.parameter_count > len(activation.parameters)
    with pytest.raises(ValueError, match="width"):
        priors.ActivatedCausalNode(
            bias=(0.0, 0.0), noise_scale=(0.0, 0.0), output_activation=activation
        )
