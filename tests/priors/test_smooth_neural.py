import json
import math
from dataclasses import replace
from typing import cast

import pytest

from r0.modules.priors import Proposal, sample_proposal


def test_gp_joint_draw_and_fixed_function_across_observations() -> None:
    task = sample_proposal(Proposal("gp"))
    assert (
        sample_proposal(replace(task.proposal, observation=1)).mechanism_json == task.mechanism_json
    )
    parameters = json.loads(task.mechanism_json)["parameters"]
    knots = parameters["knots"]
    z = parameters["standard_normal"]
    factor = parameters["cholesky"]
    # Independently derive first two rows of the two-point leading covariance block.
    a = 1.0001
    b = math.exp(-0.5 * ((knots[0] - knots[1]) / 0.6) ** 2)
    assert factor[0][0] == pytest.approx(math.sqrt(a))
    assert factor[1][0] == pytest.approx(b / math.sqrt(a))
    assert factor[1][1] == pytest.approx(math.sqrt(a - b * b / a))
    assert parameters["values"][1] == pytest.approx(factor[1][0] * z[0] + factor[1][1] * z[1])


def test_mlp_scalar_forward_matches_recorded_latent() -> None:
    task = sample_proposal(Proposal("mlp"))
    parameters = json.loads(task.mechanism_json)["parameters"]
    value = [float(cast(float, v)) for v in task.source_rows[0].values]
    weights = parameters["weights"]
    for i, matrix in enumerate(weights):
        value = [
            sum(x * w for x, w in zip(value, column, strict=True))
            for column in zip(*matrix, strict=True)
        ]
        if i != len(weights) - 1:
            value = [
                math.tanh(v) if parameters["activation"] == "tanh" else max(0.0, v) for v in value
            ]
    assert json.loads(task.trace_json)["latent"][0] == pytest.approx(value[0])
