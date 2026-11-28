"""Independent Decimal oracle and public fit/apply validation."""

import hashlib
from dataclasses import FrozenInstanceError, asdict, replace
from decimal import Decimal, localcontext

import numpy as np
import torch
from common import check, close, custody, rejects, save

from r0.modules import priors as p


def oracle(sample):
    means, inverse = [], []
    with localcontext() as ctx:
        ctx.prec = 1000
        for channel in sample.T:
            values = [Decimal.from_float(float(x)) for x in channel]
            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)).sqrt()
            means.append(float(mean))
            inverse.append(float(1 / (std + Decimal("0.0001"))))
    return means, inverse


cases = {
    "population": np.array([[2.0, 9.0, -4.0], [6.0, 9.0, 8.0], [11.0, 9.0, 2.0]]),
    "one": np.array([[17.0, -6.0]]),
    "offset": np.array([[1e15, -1e16], [1e15 + 0.25, -1e16 + 2], [1e15 + 0.5, -1e16 + 4]]),
    "opposite": np.array([[-1.7e308], [1.7e308]]),
    "rescue": np.array([[0.0], [1.6e308]]),
    "subnormal": np.array([[0.0], [5e-324]]),
}
for name, original in cases.items():
    sample = original.copy()
    stage = p.calibrate_standardization(sample, calibration_id=name)
    mean, inverse = oracle(sample)
    save(
        name + "-fit.json",
        dict(sample=sample, fitted=stage, expected_mean=mean, expected_inverse=inverse),
    )
    close(name + "-mean", stage.mean, mean)
    close(name + "-inverse", stage.inverse_std, inverse, atol=0)
    digest = hashlib.sha256(
        f"{sample.shape[0]},{sample.shape[1]}:".encode() + sample.astype("<f8").tobytes()
    ).hexdigest()
    check(name + "-digest", stage.calibration_digest, digest)
    values = sample.copy() if name != "rescue" else np.array([[-1.6e308], [1.6e308]])
    if name == "one":
        values = np.array([[18.0, -5.0], [17.0, -6.0]])
    expected = []
    with localcontext() as ctx:
        ctx.prec = 1000
        for row in values:
            expected.append(
                [
                    float(
                        (Decimal.from_float(float(v)) - Decimal.from_float(m))
                        * Decimal.from_float(s)
                    )
                    for v, m, s in zip(row, mean, inverse, strict=True)
                ]
            )
    actual = stage.apply(torch.tensor(values, dtype=torch.float64)).numpy()
    save(name + "-apply.json", dict(input=values, expected=expected, actual=actual))
    close(name + "-apply", actual, expected)
    frozen = asdict(stage)
    sample[:] = 0
    stage.apply(torch.tensor(values[:1], dtype=torch.float64))
    check(name + "-immutability", asdict(stage), frozen)
    try:
        stage.mean = ()
    except FrozenInstanceError:
        check(name + "-frozen", True)
    else:
        check(name + "-frozen", False)

stage = p.calibrate_standardization(np.array([[4.0], [4.0]]), calibration_id="malformed")
for name, change in {
    "kind": {"kind": "tanh"},
    "version": {"calibration_version": "bad"},
    "shape_parameters": {"parameters": (("a", 1.0),)},
    "width": {"inverse_std": (1.0, 2.0)},
    "nan": {"mean": (float("nan"),)},
    "inf": {"inverse_std": (float("inf"),)},
    "negative": {"inverse_std": (-1.0,)},
    "zero": {"inverse_std": (0.0,)},
    "overmax": {"inverse_std": (10001.0,)},
    "empty": {"mean": (), "inverse_std": ()},
    "list": {"mean": [4.0]},
    "rows": {"calibration_rows": 0},
    "boolrows": {"calibration_rows": True},
    "digest": {"calibration_digest": "bad"},
    "identity": {"calibration_id": ""},
}.items():
    rejects("record-" + name, lambda change=change: replace(stage, **change))
for name, values in {
    "empty": np.empty((0, 1)),
    "integer": np.ones((2, 1), dtype=int),
    "rank": np.ones(2),
    "nan": np.array([[np.nan]]),
    "inf": np.array([[np.inf]]),
}.items():
    rejects(
        "fit-" + name,
        lambda values=values, name=name: p.calibrate_standardization(values, calibration_id=name),
    )
for name, values in {
    "rank": torch.ones(2),
    "width": torch.ones(2, 2),
    "integer": torch.ones((2, 1), dtype=torch.int64),
    "nonfinite": torch.tensor([[np.inf]]),
    "overflow": torch.tensor([[1e308]], dtype=torch.float64),
}.items():
    rejects("apply-" + name, lambda values=values: stage.apply(values))
rejects("converter-forbidden", lambda: p.NumericalConverter(channel=0, transform=stage))
# Independent arithmetic: 2w generic slots + 2w means/scales + row count;
# row buffers 6w+2w+4w+4+10w = 22w+4, each 8 bytes.
for rows, width in [(2, 2), (7, 3), (1, 1)]:
    parameters = 4 * width + 1
    working = 8 * rows * (22 * width + 4) + 128 * parameters
    limits = p.CausalLimits(parameters=parameters, working_bytes=working)
    sample = np.arange(rows * width, dtype=float).reshape(rows, width)
    fitted = p.calibrate_standardization(sample, calibration_id="limits", limits=limits)
    check(f"limit-{rows}-{width}-count", fitted.parameter_count, 2 * width + 1)
    save(
        f"limits-{rows}-{width}.json",
        dict(rows=rows, width=width, parameters=parameters, working=working),
    )
    for field, value in [
        ("parameters", parameters - 1),
        ("working_bytes", working - 1),
        ("node_channels", width - 1),
        ("total_channels", width - 1),
    ]:
        if value > 0:
            rejects(
                f"limit-{rows}-{width}-{field}",
                lambda field=field, value=value, sample=sample, limits=limits: (
                    p.calibrate_standardization(
                        sample, calibration_id="limits", limits=replace(limits, **{field: value})
                    )
                ),
            )
custody()
