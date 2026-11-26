"""Independent decimal arithmetic for structural-scale repair boundaries."""

import itertools
import json
import math
import sys
from dataclasses import asdict, replace
from decimal import Decimal, localcontext

import numpy as np

from r0.modules.priors import (
    CausalActivation,
    CausalRecipe,
    ExogenousDistribution,
    InputDistribution,
    NoiseResponse,
    StochasticCausalNode,
    recipe_from_dict,
)


def repair_boundaries(root, gen, trace, controls, vals, must_reject, record):
    # Independent high-precision real equation, not the implementation's log rescue.
    cases = [
        ("mixed zero/rescued/ordinary", (1e200, 1e200, 2.0), (0.0, 1e-200, 0.5), 1.0, 1.0, 2.0),
        (
            "coefficient intermediate overflow",
            (1e100, 0.0, -1e100),
            (1e-200, 1.0, 1e-200),
            0.0,
            1e200,
            2.0,
        ),
        (
            "nonzero offset and fractional power",
            (1e250, 0.0, -1e250),
            (1e-150, 1.0, 1e-150),
            3.0,
            0.125,
            1.5,
        ),
        ("zero strength skips response", (1e200, 0.0, -1e200), (0.0, 2.0, 0.5), 2.0, 0.0, 2.0),
    ]
    for name, bias, base, offset, strength, power in cases:
        n = StochasticCausalNode(
            bias=bias,
            noise_scale=base,
            noise_response=NoiseResponse(offset, strength, power),
            output_activation=CausalActivation("tanh"),
        )
        r = CausalRecipe(
            task=controls(64, 2),
            nodes=(n,),
            observed=((0, 0), (0, 1)),
            target=((0, 2),),
            version="causal-v2",
        )
        t = gen(r)
        ids = [0, 7, 31, 63]
        tr = trace(t, ids)
        with localcontext() as ctx:
            ctx.prec = 100
            expected = [
                float(
                    Decimal.from_float(b)
                    * (
                        Decimal.from_float(offset)
                        + Decimal.from_float(strength)
                        * abs(Decimal.from_float(x)) ** Decimal.from_float(power)
                    )
                )
                for b, x in zip(base, bias, strict=True)
            ]
        for row in range(len(ids)):
            scales = tr["noise-scale-0"]["values"][row]
            raw = tr["exogenous-0"]["values"][row]
            values = tr["causal-0"]["values"][row]
            for j in range(3):
                assert math.isclose(scales[j], expected[j], rel_tol=3e-13, abs_tol=0.0), (
                    name,
                    j,
                    scales[j],
                    expected[j],
                )
                assert math.isclose(
                    tr["innovation-0"]["values"][row][j],
                    scales[j] * raw[j],
                    rel_tol=1e-14,
                    abs_tol=0.0,
                )
                assert math.isclose(
                    values[j],
                    math.tanh(bias[j] + expected[j] * raw[j]),
                    rel_tol=1e-13,
                    abs_tol=1e-13,
                )
        restored = recipe_from_dict(json.loads(json.dumps(asdict(r))))
        assert gen(restored).episode() == t.episode()
        assert np.array_equal(vals(t), vals(gen(replace(r, task=controls(97, 2))))[:64])
        record(
            "repair decimal equation, mixed vectors, replay and prefix",
            case=name,
            expected_scales=expected,
        )
    # A true nonrepresentable final scale must fail even with a bounded activation.
    r = root(noise_response=NoiseResponse(0.0, 1.0, 2.0))
    r = replace(
        r,
        nodes=(
            replace(r.nodes[0], bias=(1e200, 1e200), output_activation=CausalActivation("tanh")),
        ),
    )
    must_reject("genuine final scale overflow remains explicit", lambda: gen(r))
    # Underflow is the symmetric representable-result boundary of overflow rescue.
    r = root(noise_response=NoiseResponse(0.0, 1.0, 2.0))
    r = replace(r, nodes=(replace(r.nodes[0], bias=(1e-200, 1e-200), noise_scale=(1e200, 1e200)),))
    t = gen(r)
    actual = trace(t, [0])["noise-scale-0"]["values"][0][0]
    with localcontext() as ctx:
        ctx.prec = 100
        expected = float(Decimal.from_float(1e200) * Decimal.from_float(1e-200) ** 2)
    assert math.isclose(actual, expected, rel_tol=3e-13, abs_tol=0.0), (
        "representable scale lost to intermediate underflow",
        actual,
        expected,
    )
    record(
        "representable tiny scale survives intermediate underflow", expected=expected, actual=actual
    )


def decimal_scale_grid(root, gen, trace, record):
    """64 log-spaced cases plus explicit subnormal and offset boundaries."""
    cases = [
        (f"grid-{i}", 10.0**x, 10.0**b, 0.0, 10.0**s, 2.0)
        for i, (x, b, s) in enumerate(
            itertools.product((-200, -155, -100, 100), (-200, 0, 200, 300), (-200, 0, 200, 300))
        )
    ]
    cases += [
        ("positive offset hides power underflow", 1e-200, 1e200, 1e-101, 1e300, 2.0),
        ("positive offset hides coefficient underflow", 1e-100, 1e300, 1e-301, 1e-200, 2.0),
        ("subnormal power intermediate", 1e-160, 1e200, 0.0, 1.0, 2.0),
        ("subnormal coefficient intermediate", 1e-100, 1e200, 0.0, 1e-120, 2.0),
        ("noninteger power intermediate", 1e-250, 1e200, 0.0, 1.0, 1.5),
        ("final subnormal", 1e-200, 1e100, 0.0, 1e-23, 2.0),
        ("final smallest subnormal", 1e-200, 1e100, 0.0, 5e-24, 2.0),
        ("final rounds to zero", 1e-200, 1e100, 0.0, 1e-25, 2.0),
        ("zero base", 1e200, 0.0, 1e200, 1e200, 2.0),
    ]
    finite = overflow = rounded_zero = 0
    for name, x, base, offset, strength, power in cases:
        with localcontext() as context:
            context.prec = 100
            exact = Decimal.from_float(base) * (
                Decimal.from_float(offset)
                + Decimal.from_float(strength)
                * abs(Decimal.from_float(x)) ** Decimal.from_float(power)
            )
            expected = float(exact)
        r = root(
            ExogenousDistribution(inputs=InputDistribution(distribution="uniform", scale=0.25)),
            rows=16,
            noise_response=NoiseResponse(offset, strength, power),
            output_activation=CausalActivation("tanh"),
        )
        r = replace(r, nodes=(replace(r.nodes[0], bias=(x, -x), noise_scale=(base, base)),))
        if math.isinf(expected):
            try:
                gen(r)
            except ValueError as error:
                if "nonfinite signal-dependent noise scale" not in str(error):
                    raise AssertionError((name, "unexpected rejection", str(error))) from error
                overflow += 1
                record("decimal grid genuine final overflow", case=name, exact_decimal=str(exact))
                continue
            raise AssertionError((name, "accepted genuinely overflowing scale", str(exact)))
        t = gen(r)
        tr = trace(t, [0, 15])
        for row in range(2):
            for channel in range(2):
                actual = tr["noise-scale-0"]["values"][row][channel]
                # Normal floats: relative accuracy. Subnormal floats: at most one ULP.
                allowed = max(4e-13 * expected, math.ulp(0.0)) if expected else 0.0
                assert abs(actual - expected) <= allowed, (name, actual, expected, allowed)
                raw = tr["exogenous-0"]["values"][row][channel]
                innovation = tr["innovation-0"]["values"][row][channel]
                assert math.isclose(innovation, actual * raw, rel_tol=1e-14, abs_tol=math.ulp(0.0))
                assert math.isfinite(tr["causal-0"]["values"][row][channel])
        finite += 1
        rounded_zero += int(expected == 0 and exact > 0)
        record(
            "decimal grid finite scale",
            case=name,
            exact_decimal=str(exact),
            expected=expected,
            actual=tr["noise-scale-0"]["values"][0][0],
            rounds_to_zero=expected == 0 and exact > 0,
        )
    record(
        "bounded independent Decimal grid complete",
        cases=len(cases),
        finite=finite,
        explicit_overflow=overflow,
        positive_rounds_to_zero=rounded_zero,
        normal_min=sys.float_info.min,
        minimum_subnormal=math.ulp(0.0),
    )
