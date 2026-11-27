"""Independent endpoint-neighborhood and wider-channel numerical checks."""

import json
import math
import os
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch
from custody import record_custody
from integration import l2_expected
from numerical_oracle import close, expected_cdf

from r0.modules import priors

OUT = Path(os.environ["EVIDENCE_DIR"])


def main():
    record_custody(OUT / "raw" / "endpoint-custody.json")
    count = 0
    tiny = math.ulp(0.0)
    eps = 1e-30
    shapes = [
        (tiny, 1e-308),
        (1e-308, tiny),
        (tiny, tiny),
        (tiny, 1e308),
        (1e308, tiny),
        (0.2, 0.2),
        (1.0, 1.0),
        (5.0, 5.0),
        (1e-200, 1e-200),
    ]
    ranges = [
        (0.0, tiny),
        (-tiny, 0.0),
        (-tiny, tiny),
        (tiny, tiny),
        (-tiny, -tiny),
        (0.0, 1e-100),
        (-1e-100, 1e-100),
        (-eps, 0.0),
        (0.0, eps),
        (eps, eps),
        (-2 * eps, -eps),
    ]
    for low, high in ranges:
        boundary = high + eps
        values = sorted(
            set(
                [
                    low,
                    high,
                    eps,
                    boundary,
                    math.nextafter(boundary, -math.inf),
                    math.nextafter(boundary, math.inf),
                    math.nextafter(low, math.inf),
                ]
            )
        )
        for a, b in shapes:
            transform = priors.calibrate_kumaraswamy(
                np.array([[low], [high]]), a=a, b=b, calibration_id="endpoint-neighborhood"
            )
            actual = transform.apply(torch.tensor(values, dtype=torch.float64).reshape(-1, 1))
            diagnostic = transform.boundary_diagnostic(
                torch.tensor(values, dtype=torch.float64).reshape(-1, 1)
            )
            with localcontext() as context:
                context.prec = 1000
                expected_clamp = sum(
                    x < low
                    or Decimal.from_float(x) - Decimal.from_float(high) > Decimal.from_float(eps)
                    for x in values
                ) / len(values)
            expected_outside = sum(x < low or x > high for x in values) / len(values)
            diagnostic_ok = diagnostic["domain_clamp_fraction"] == [expected_clamp] and diagnostic[
                "outside_calibration_fraction"
            ] == [expected_outside]
            for x, observed in zip(values, actual[:, 0].tolist(), strict=True):
                expected = expected_cdf(x, low, high, a, b)
                row = dict(
                    low=low,
                    high=high,
                    x=x,
                    a=a,
                    b=b,
                    actual=observed,
                    expected=expected,
                    passed=close(observed, expected) and diagnostic_ok,
                    expected_clamp=expected_clamp,
                    expected_outside=expected_outside,
                    diagnostic=diagnostic,
                )
                with (OUT / "raw" / "endpoint.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                assert row["passed"], row
                count += 1
    for width in (1, 2, 7, 64, 1024):
        for magnitude in (0.0, tiny, 1e-200, 1.0, 1e200, float.fromhex("0x1.fffffffffffffp+1023")):
            sample = np.full((2, width), magnitude)
            sample[1] *= -1
            transform = priors.calibrate_l2_normalization(sample, calibration_id="width-extremes")
            expected_factor = l2_expected(sample)
            actual = transform.apply(torch.tensor(sample, dtype=torch.float64))
            expected = torch.tensor(sample * expected_factor, dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=3e-11, atol=math.ulp(0.0) * 8)
            row = dict(
                kind="l2",
                width=width,
                magnitude=magnitude,
                record=asdict(transform),
                expected_factor=expected_factor,
                actual=actual[:, :3].tolist(),
                passed=close(transform.factor, expected_factor),
            )
            with (OUT / "raw" / "endpoint.jsonl").open("a") as output:
                output.write(json.dumps(row) + "\n")
            assert row["passed"], row
            count += 1
    sample = np.array([[0.0, -1e308, 0.0, 7.0], [1.0, 1e308, tiny, 7.0]])
    transform = priors.calibrate_kumaraswamy(sample, a=0.2, b=0.2, calibration_id="mixed-width")
    values = torch.tensor(
        [[0.25, 0.0, eps, 7.0], [1.0, 1e308, tiny, 8.0], [0.0, -1e308, 0.0, 6.0]],
        dtype=torch.float64,
    )
    actual = transform.apply(values)
    expected = torch.tensor(
        [
            [
                expected_cdf(x, float(sample[0, j]), float(sample[1, j]), 0.2, 0.2)
                for j, x in enumerate(row)
            ]
            for row in values.tolist()
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-11, atol=math.ulp(0.0) * 8)
    torch.testing.assert_close(
        torch.cat([transform.apply(row.reshape(1, -1)) for row in values]), actual, rtol=0, atol=0
    )
    sample[:] = -17.0
    torch.testing.assert_close(transform.apply(values), actual, rtol=0, atol=0)
    count += 1
    summary = dict(status="PASS", cases=count, precision=1000)
    (OUT / "endpoint-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
