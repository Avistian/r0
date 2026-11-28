"""Check dimension-changing authored compositions with distinct fitted stages."""

import json
import math
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from custody import record_custody
from integration import l2_expected
from numerical_oracle import expected_cdf

from r0.modules import priors

OUT = Path(os.environ["EVIDENCE_DIR"])


def oracle(transform, row, sample):
    if transform.kind == "l2_normalize":
        return [x * l2_expected(sample) for x in row]
    return [
        expected_cdf(x, float(sample[:, j].min()), float(sample[:, j].max()), 0.7, 1.3)
        for j, x in enumerate(row)
    ]


def main():
    record_custody(OUT / "raw" / "composition-custody.json")
    source = np.array([[-2.0, -3.0, 1.0], [3.0, 2.0, 4.0], [0.0, 0.0, 2.0]])
    destination = np.array([[-7.0, -2.0], [5.0, 8.0], [0.0, 1.0]])
    weights = ((1.0, -0.5), (-2.0, 1.0), (0.25, 2.0))
    for input_kind in ("l2", "kumaraswamy"):

        def fit(sample, kind):
            if kind == "l2":
                return priors.calibrate_l2_normalization(sample, calibration_id="composition")
            return priors.calibrate_kumaraswamy(sample, a=0.7, b=1.3, calibration_id="composition")

        first = fit(source, input_kind)
        second = fit(destination, "l2" if input_kind != "l2" else "kumaraswamy")
        node = priors.ActivatedCausalNode(
            parents=(0,),
            weights=weights,
            bias=(0.2, -0.1),
            noise_scale=(0.0, 0.0),
            input_activation=first,
            output_activation=second,
        )
        recipe = priors.CausalRecipe(
            task=priors.LinearRecipe(
                rows=16, context_rows=8, features=3, signal_features=3, active_features=3
            ),
            nodes=(priors.CausalNode(bias=(0.0, 0.0, 0.0), noise_scale=(1.0, 1.0, 1.0)), node),
            observed=((0, 0), (0, 1), (0, 2)),
            target=((1, 0),),
            version="causal-v2",
        )
        generated = priors.generate_task(recipe, seed=3928, task_id=8)
        expected = []
        for row in generated.inputs.tolist():
            hidden = oracle(first, row, source)
            projected = [
                math.fsum(hidden[i] * weights[i][j] for i in range(3)) + node.bias[j]
                for j in range(2)
            ]
            expected.append(oracle(second, projected, destination))
        torch.testing.assert_close(
            generated.scores[:, 0],
            torch.tensor([row[0] for row in expected], dtype=torch.float64),
            rtol=2e-12,
            atol=1e-14,
        )
        trace = priors.explain_generation(generated, [0, 15])
        node_trace = next(item for item in trace["nodes"] if item["id"] == "causal-1")
        torch.testing.assert_close(
            torch.tensor(node_trace["values"], dtype=torch.float64),
            torch.tensor([expected[0], expected[15]], dtype=torch.float64),
            rtol=2e-12,
            atol=1e-14,
        )
        assert node.activation_parameter_count == first.parameter_count + second.parameter_count
        assert node.working_extra_channels == max(
            first.working_extra_channels, second.working_extra_channels
        )
        assert priors.recipe_from_dict(json.loads(json.dumps(asdict(recipe)))) == recipe
        with (OUT / "raw" / "composition.jsonl").open("a") as output:
            output.write(
                json.dumps(
                    dict(
                        status="PASS",
                        input_kind=input_kind,
                        recipe=asdict(recipe),
                        expected=expected,
                        trace=trace,
                    )
                )
                + "\n"
            )
        try:
            replace(node, output_activation=first)
        except ValueError:
            pass
        else:
            raise AssertionError("dimension mismatch accepted")
    print(json.dumps(dict(status="PASS", dimension_changing_compositions=2)))


if __name__ == "__main__":
    main()
