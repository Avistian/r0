"""Independent Decimal equation checks; run against an explicit PYTHONPATH."""

import hashlib
import json
import math
import os
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import torch
from custody import record_custody

from r0.modules import priors

OUT = Path(os.environ["EVIDENCE_DIR"])
D = Decimal.from_float


def expected_cdf(x, low, high, a, b):
    """Direct defining equation at 1000 digits, without implementation branches."""
    with localcontext() as context:
        context.prec = 1000
        u = max(Decimal(0), min(Decimal(1), (D(x) - D(low)) / (D(high) - D(low) + D(1e-30))))
        if u == 0 or u == 1:
            return float(u)
        exponent = D(a) * u.ln()
        if exponent < -10000:
            return 0.0
        p = exponent.exp()
        return float(1 - ((1 - p).ln() * D(b)).exp())


def close(actual, expected):
    return abs(actual - expected) <= max(abs(expected) * 3e-11, math.ulp(expected) * 8)


def main():
    record_custody(OUT / "raw" / "numerical-custody.json")
    smallest = math.ulp(0.0)
    shapes = [
        (1.0, 1.0),
        (0.2, 0.2),
        (5.0, 5.0),
        (smallest, 1e-308),
        (1e-308, smallest),
        (smallest, 1e308),
        (1e308, smallest),
        (1e308, 1e308),
        (1e-200, 1e200),
        (1e200, 1e-200),
        (36.0, 1e308),
        (1000.0, 1e308),
        (1.0, 0.2),
        (0.2, 1.0),
    ]
    ranges = [
        (0.0, 1.0),
        (-1e308, 1e308),
        (0.0, 1e308),
        (0.0, 1e-30),
        (0.0, smallest),
        (0.0, 0.0),
        (7.0, 7.0),
        (1e308, math.nextafter(1e308, math.inf)),
    ]
    records = []
    for low, high in ranges:
        midpoint = low / 2 + high / 2
        values = sorted(
            set(
                [
                    low,
                    high,
                    midpoint,
                    math.nextafter(low, -math.inf),
                    math.nextafter(low, math.inf),
                    math.nextafter(high, -math.inf),
                    math.nextafter(high, math.inf),
                    smallest,
                    0.0,
                    1e-30,
                    2e-30,
                ]
            )
        )
        values = [v for v in values if math.isfinite(v)]
        for a, b in shapes:
            transform = priors.calibrate_kumaraswamy(
                np.array([[low], [high]]), a=a, b=b, calibration_id="oracle-independent"
            )
            actual = transform.apply(torch.tensor(values, dtype=torch.float64).reshape(-1, 1))
            for x, observed in zip(values, actual[:, 0].tolist(), strict=True):
                expected = expected_cdf(x, low, high, a, b)
                row = dict(
                    kind="kumaraswamy",
                    low=low,
                    high=high,
                    a=a,
                    b=b,
                    x=x,
                    actual=observed,
                    expected=expected,
                    passed=close(observed, expected),
                )
                records.append(row)
                with (OUT / "raw" / "numerical.jsonl").open("a") as output:
                    output.write(json.dumps(row) + "\n")
                if not row["passed"]:
                    raise AssertionError(row)
    for sample in [
        np.array([[3.0, 4.0], [0.0, 0.0]]),
        np.zeros((3, 4)),
        np.full((2, 2), 1e308),
        np.full((3, 7), smallest),
        np.array([[1e308, -1e308], [smallest, 1.0]]),
        np.arange(1.0, 22.0).reshape(3, 7),
    ]:
        transform = priors.calibrate_l2_normalization(sample, calibration_id="l2-oracle")
        with localcontext() as context:
            context.prec = 1000
            energy = sum((D(float(x)) ** 2 for x in sample.flat), Decimal(0)) / len(sample)
            expected = float(1 / (energy.sqrt() + D(1e-8)))
        observed = transform.factor
        row = dict(
            kind="l2",
            sample=sample.tolist(),
            record=asdict(transform),
            expected=expected,
            actual=observed,
            passed=close(observed, expected),
        )
        records.append(row)
        with (OUT / "raw" / "numerical.jsonl").open("a") as output:
            output.write(json.dumps(row) + "\n")
        if not row["passed"]:
            raise AssertionError(row)
    result = dict(
        status="PASS",
        cases=len(records),
        precision=1000,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (OUT / "numerical-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
