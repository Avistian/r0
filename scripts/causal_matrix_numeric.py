"""Independent exact-input matrix oracles; no implementation arithmetic helpers."""

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from dataclasses import asdict, replace
from decimal import Decimal as D
from decimal import localcontext
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch

from r0.modules.priors import (
    CausalActivation,
    CausalLimits,
    CausalNode,
    CausalRecipe,
    GaussianWeights,
    ImportanceFactorizedWeights,
    KernelPointCloud,
    LaplaceKernelWeights,
    LinearRecipe,
    MatrixCausalNode,
    NormalizedWeights,
    PolynomialWeightVector,
    TransformedWeights,
    UniformWeightVector,
    recipe_from_dict,
)


def dec(x):
    return D.from_float(float(x))


def vector(v):
    if isinstance(v, UniformWeightVector):
        return v.draws
    with localcontext() as c:
        c.prec = 3000
        scores = [
            dec(v.log_noise_scale) * dec(z) - dec(v.decay) * dec(math.log(i + 1))
            for i, z in enumerate(v.draws)
        ]
        scores = [scores[i] for i in (v.permutation or range(len(scores)))]
        high = max(scores)
        weights = [(s - high).exp() if s - high > -10000 else D(0) for s in scores]
        norm = (sum(w * w for w in weights) / len(weights)).sqrt()
        # The contract rounds vectors before multiplying by factors or innovations.
        return tuple(float(w / norm) for w in weights)


def product(left, weights, right):
    return tuple(
        tuple(
            float(
                sum(
                    (
                        F(a) * F(w) * F(right[k][j])
                        for k, (a, w) in enumerate(zip(row, weights, strict=True))
                    ),
                    F(),
                )
            )
            for j in range(len(right[0]))
        )
        for row in left
    )


def norm(base, noise, scale, eps):
    with localcontext() as c:
        c.prec = 3000
        result = []
        for row, jitter in zip(base, noise, strict=True):
            exact = [dec(x) + dec(scale) * dec(z) for x, z in zip(row, jitter, strict=True)]
            den = dec(eps) + sum(x * x for x in exact).sqrt()
            result.append(tuple(float(x / den) if den else 0.0 for x in exact))
        return tuple(result)


def points(cloud):
    with localcontext() as c:
        c.prec = 3000
        rows = []
        for k, innovation in zip(cloud.assignments, cloud.innovations, strict=True):
            values = []
            log = dec(cloud.scale_mean) + dec(cloud.scale_coefficient) * dec(cloud.scale_draws[k])
            for factor, center in zip(cloud.factors[k], cloud.centers[k], strict=True):
                core = sum(
                    (
                        F(a) * F(w) * F(z)
                        for a, w, z in zip(
                            factor, vector(cloud.factor_weights[k]), innovation, strict=True
                        )
                    ),
                    F(),
                )
                exact = D(core.numerator) / D(core.denominator)
                values.append(float(dec(center) + exact * log.exp()))
            rows.append(tuple(values))
        return tuple(rows)


def run(out):
    out.mkdir(parents=True)
    events = []

    def emit(row):
        events.append(row)
        with (out / "numeric.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        assert row["status"] == "PASS", row["name"]

    def check(name, actual, expected, record=None):
        def flat(xs):
            return [x for row in xs for x in row]

        a, b = flat(actual), flat(expected)
        ok = len(a) == len(b) and all(
            math.isfinite(x) and math.isclose(x, y, rel_tol=5e-12, abs_tol=5e-324)
            for x, y in zip(a, b, strict=False)
        )
        emit(
            dict(
                name=name,
                status="PASS" if ok else "REJECT",
                actual=actual,
                expected=expected,
                record=asdict(record) if record else None,
            )
        )

    def reject(name, fn):
        try:
            fn()
        except ValueError as e:
            emit(dict(name=name, status="PASS", rejection=str(e)))
            return
        emit(dict(name=name, status="REJECT", reason="invalid construction accepted"))

    g = GaussianWeights(((3.0, -4.0), (0.0, 5.0)))
    check("raw Gaussian retained unchanged", g.materialize(), g.draws, g)
    for bad in ((), ((1.0,), (2.0, 3.0)), ((math.inf,),), ((True,),)):
        reject("Gaussian geometry " + str(bad), lambda bad=bad: GaussianWeights(bad))
    rng = random.Random(28071)
    vectors = [
        UniformWeightVector((0.25, 0.75)),
        PolynomialWeightVector((0.0, 0.0), decay=0, log_noise_scale=0),
        PolynomialWeightVector((0.0, -1000.0), decay=0, log_noise_scale=1),
        PolynomialWeightVector((1e308, 1e308), decay=2, log_noise_scale=1e308),
    ]
    for i in range(36):
        v = vectors[i % len(vectors)]
        left = tuple(tuple(rng.uniform(-4, 4) for _ in range(2)) for _ in range(3))
        right = tuple(tuple(rng.uniform(-4, 4) for _ in range(4)) for _ in range(2))
        obj = ImportanceFactorizedWeights(left, right, v)
        check("factor " + str(i), obj.materialize(), product(left, vector(v), right), obj)
    obj = ImportanceFactorizedWeights(((1e308, 1e308),), ((1e308,), (-1e308,)), vectors[1])
    check("factor overflowing opposite terms cancel", obj.materialize(), ((0.0,),), obj)
    obj = ImportanceFactorizedWeights(((1e308, 1e308),), ((0.0,), (1e308,)), vectors[2])
    check("factor underflowed vector stays zero", obj.materialize(), ((0.0,),), obj)
    reject(
        "factor exact budget",
        lambda: ImportanceFactorizedWeights(
            ((1e308, 1e308),), ((1e308,), (-1e308,)), vectors[1], exact_terms_limit=1
        ).materialize(),
    )
    invalid = ImportanceFactorizedWeights(((1e308,),), ((1e308,),))
    reject("raw factor nonfinite", invalid.materialize)
    reject(
        "normalization cannot rescue raw factor", lambda: NormalizedWeights(invalid).materialize()
    )
    for dimension in (1, 2, 3, 7):
        left = tuple(tuple(rng.uniform(-4, 4) for _ in range(dimension)) for _ in range(2))
        right = tuple(tuple(rng.uniform(-4, 4) for _ in range(dimension)) for _ in range(3))
        signs = ((1.0, -1.0, 0.0), (-1.0, 0.0, 1.0))
        for scale in (0.0, 0.1, 1.0, 10.0, 1e308):
            obj = LaplaceKernelWeights(left, right, signs, scale)
            with localcontext() as c:
                c.prec = 100
                expected = tuple(
                    tuple(
                        float(
                            dec(s)
                            * (
                                -(
                                    sum(
                                        (dec(a) - dec(b)) ** 2
                                        for a, b in zip(left_point, r, strict=True)
                                    )
                                ).sqrt()
                                * dec(scale)
                            ).exp()
                        )
                        if scale < 1e308
                        else (float(s) if left_point == r else 0.0)
                        for r, s in zip(right, sg, strict=True)
                    )
                    for left_point, sg in zip(left, signs, strict=True)
                )
            check(f"kernel d{dimension} scale{scale}", obj.materialize(), expected, obj)
    obj = LaplaceKernelWeights(((1e308, 1e308),), ((-1e308, -1e308),), ((-1.0,),), 1e-308)
    with localcontext() as c:
        c.prec = 150
        distance = (2 * (2 * dec(1e308) * dec(1e-308)) ** 2).sqrt()
        expected = ((-float((-distance).exp()),),)
    check("kernel overflowing difference recovered", obj.materialize(), expected, obj)
    reject(
        "kernel full geometry fallback budget",
        lambda: replace(obj, exact_distance_terms_limit=1).materialize(),
    )
    for kwargs in ({"signs": ((0.5,),)}, {"distance_scale": -1}, {"right_points": ((0.0, 0.0),)}):
        reject(
            "kernel malformed " + str(kwargs), lambda kwargs=kwargs: LaplaceKernelWeights(**kwargs)
        )
    for base in (((3.0, 4.0),), ((1e308, -1e308),), ((5e-324, -1e-323),), ((0.0, 0.0),)):
        for jitter in (((0.0, 0.0),), ((-1.0, 1.0),), ((1e308, -1e308),)):
            for scale in (0.0, 1e-6, 1e308):
                for eps in (0.0, 1e-6, 1e308):
                    obj = NormalizedWeights(GaussianWeights(base), jitter, scale, eps)
                    check(
                        "normalization " + str(len(events)),
                        obj.materialize(),
                        norm(base, jitter, scale, eps),
                        obj,
                    )
    reject(
        "jitter exact budget",
        lambda: NormalizedWeights(g, ((0.0, 0.0), (0.0, 0.0)), exact_terms_limit=3).materialize(),
    )
    reject("normalization direct nesting", lambda: NormalizedWeights(NormalizedWeights()))
    child = NormalizedWeights(
        GaussianWeights(((3.0, 4.0), (-4.0, 3.0))), ((0.0, 0.0), (0.0, 0.0)), 0.0, 0.0
    )
    for kind, expected in [
        ("rank", ((1.0, 3.0), (0.0, 2.0))),
        ("argmax", ((0.0, 1.0), (0.0, 0.0))),
        ("argsort", ((2.0, 0.0), (3.0, 1.0))),
    ]:
        obj = TransformedWeights(child, CausalActivation(kind), ((0.0, 0.0), (0.0, 0.0)), 0.0)
        check("whole flattened " + kind, obj.materialize(), expected, obj)
    values = (0.6, 0.8, -0.8, 0.6)
    exp = [math.exp(v) for v in values]
    expected = tuple(tuple(v / math.fsum(exp) for v in exp[i : i + 2]) for i in (0, 2))
    obj = TransformedWeights(child, CausalActivation("softmax"), ((0.0, 0.0), (0.0, 0.0)), 0.0)
    check("whole flattened softmax", obj.materialize(), expected, obj)
    reject(
        "recursive transformed child",
        lambda: TransformedWeights(NormalizedWeights(obj, ((0.0, 0.0), (0.0, 0.0)))),
    )
    bad = TransformedWeights(activation=CausalActivation("exp", (("input_shift", 1000.0),)))
    reject("raw activation nonfinite", bad.materialize)
    reject(
        "outer normalization cannot rescue activation", lambda: NormalizedWeights(bad).materialize()
    )
    for i in range(20):
        law = "random-covariance" if i % 2 == 0 else "gaussian-mixture"
        cloud = KernelPointCloud(
            centers=((0.0, 0.0),) if law == "random-covariance" else ((1.0, -2.0), (3.0, 4.0)),
            factors=(((2.0, 3.0), (-4.0, 5.0)),) * (1 if law == "random-covariance" else 2),
            factor_weights=(vectors[i % 4],) * (1 if law == "random-covariance" else 2),
            component_weights=UniformWeightVector(
                (0.5,) if law == "random-covariance" else (0.25, 0.75)
            ),
            assignments=(0, 0) if law == "random-covariance" else (0, 1),
            innovations=((0.3, -0.8), (0.7, 0.2)),
            scale_mean=0.0 if law == "random-covariance" else 0.3,
            scale_coefficient=0.0 if law == "random-covariance" else -0.7,
            scale_draws=(0.0,) if law == "random-covariance" else (0.2, -0.4),
            law=law,
            base_law="uniform" if law == "random-covariance" else "gaussian",
        )
        check("point formula " + str(i), cloud.materialize(), points(cloud), cloud)
    cloud = KernelPointCloud(
        centers=((-1.0,),), factors=(((2.0,),),), innovations=((1.0,),), scale_mean=1e-100
    )
    check("point near center cancellation", cloud.materialize(), points(cloud), cloud)
    cloud = KernelPointCloud(factors=(((1e308,),),), innovations=((1e308,),), scale_mean=-1000.0)
    check("point overflowing product recovered by scale", cloud.materialize(), points(cloud), cloud)
    reject("nonrepresentable raw points", lambda: replace(cloud, scale_mean=0.0).materialize())
    reject(
        "point term budget",
        lambda: replace(
            KernelPointCloud(),
            innovations=((1.0,), (2.0,)),
            assignments=(0, 0),
            exact_terms_limit=1,
        ).materialize(),
    )
    reject(
        "point zero component mass",
        lambda: KernelPointCloud(component_weights=UniformWeightVector((0.0,))).materialize(),
    )
    cloud = KernelPointCloud(
        centers=((0.0,), (1.0,)),
        factors=(((1.0,),),) * 2,
        factor_weights=(UniformWeightVector(),) * 2,
        component_weights=UniformWeightVector((0.0, 0.5)),
        assignments=(0,),
        scale_draws=(0.0, 0.0),
    )
    reject("assignment to zero mass", cloud.materialize)
    for kwargs in ({"base_law": "uniform"}, {"assignments": (True,)}, {"scale_draws": (math.inf,)}):
        reject("point malformed " + str(kwargs), lambda kwargs=kwargs: KernelPointCloud(**kwargs))
    reject(
        "uniform endpoint outside support",
        lambda: KernelPointCloud(
            law="random-covariance", base_law="uniform", innovations=((1.0,),)
        ),
    )
    reject(
        "covariance nonzero center",
        lambda: KernelPointCloud(law="random-covariance", centers=((1.0,),)),
    )
    cloud = KernelPointCloud(innovations=((0.0,), (2.0,)), assignments=(0, 0))
    kernel = LaplaceKernelWeights(((0.0,),), ((1.0,),), point_construction=cloud)
    check("stored point geometry agrees", kernel.materialize(), ((math.exp(-1),),), kernel)
    reject(
        "stored point geometry mismatch",
        lambda: replace(kernel, right_points=((2.0,),)).materialize(),
    )
    task = LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    node = MatrixCausalNode(parents=(0,), weights=kernel.materialize(), weight_construction=kernel)
    kwargs = dict(
        task=task,
        nodes=(CausalNode(), node),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    recipe = CausalRecipe(**kwargs)
    data = asdict(recipe)
    data["nodes"][1]["weights"] = ((0.5,),)
    reject("stored dense projection mismatch", lambda: recipe_from_dict(data))
    with patch.object(
        LaplaceKernelWeights, "materialize", side_effect=AssertionError("early materialization")
    ):
        reject(
            "parameter admission before materialization",
            lambda: CausalRecipe(**kwargs, limits=CausalLimits(parameters=1)),
        )
    data = asdict(recipe)
    stage = {"version": "gaussian-weights-v1"}
    for _ in range(100):
        stage = {"version": "normalized-weights-v1", "source": stage}
    data["nodes"][1]["weight_construction"] = stage
    reject("decoder bounds deep stage recursion", lambda: recipe_from_dict(data))
    source = Path(inspect.getfile(GaussianWeights))
    result = dict(
        status="PASS",
        checks=len(events),
        python=sys.executable,
        public_import=str(source),
        module_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    (out / "numeric-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    run(parser.parse_args().output_dir)
