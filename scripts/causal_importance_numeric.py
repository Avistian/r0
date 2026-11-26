"""Independent public-contract oracle; PYTHONPATH selects an immutable source tree."""

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from dataclasses import asdict
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from unittest.mock import patch

from r0.modules.priors import (
    CausalLimits,
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    MatrixCausalNode,
    recipe_from_dict,
    sample_recipe,
)
from r0.modules.priors import ImportanceWeights as I
from r0.modules.priors import PolynomialWeightVector as P
from r0.modules.priors import UniformWeightVector as U


def dec(x):
    return D.from_float(float(x))


def oracle(v, g=None, epsilon=0):
    # Exact binary inputs and products; transcendental evaluation separate from production.
    with localcontext() as c:
        c.prec = 3000
        c.Emin = -999999
        c.Emax = 999999
        if isinstance(v, U):
            w = [dec(x) for x in v.draws]
        else:
            scores = [
                dec(v.log_noise_scale) * dec(z) - dec(v.decay) * dec(math.log(i + 1))
                for i, z in enumerate(v.draws)
            ]
            scores = [scores[i] for i in (v.permutation or tuple(range(len(scores))))]
            top = max(scores)
            centered = [s - top for s in scores]
            # Avoid exponentiating numbers whose exp cannot influence binary64 even
            # after multiplication by max float64. Epsilon-zero direction recenters
            # over active coordinates, preserving arbitrarily small common factors.
            if g is not None and epsilon == 0:
                active = [s for s, x in zip(scores, g, strict=False) if x]
                if not active:
                    return [0.0] * len(scores)
                top = max(active)
                centered = [s - top for s in scores]
            with localcontext() as e:
                e.prec = 100
                w = [
                    D(0) if s < -10000 or (g is not None and epsilon == 0 and not g[i]) else s.exp()
                    for i, s in enumerate(centered)
                ]
                if not (g is not None and epsilon == 0):
                    norm = (sum(x * x for x in w) / D(len(w))).sqrt()
                    w = [x / norm for x in w]
        if g is None:
            return [float(x) for x in w]
        numerator = [dec(x) * v for x, v in zip(g, w, strict=False)]
        norm = sum(x * x for x in numerator).sqrt()
        den = dec(epsilon) + norm
        return [float(x / den) if den else 0.0 for x in numerator]


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    events = []

    def record(name, actual, expected, inputs=None):
        ok = len(actual) == len(expected) and all(
            math.isfinite(a) and math.isclose(a, b, rel_tol=3e-12, abs_tol=5e-324)
            for a, b in zip(actual, expected, strict=False)
        )
        row = dict(
            name=name,
            status="PASS" if ok else "REJECT",
            actual=actual,
            expected=expected,
            inputs=inputs,
        )
        events.append(row)
        with (out / "numeric.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        if not ok:
            raise AssertionError(name)

    def reject(name, fn):
        try:
            fn()
        except ValueError as e:
            events.append(dict(name=name, status="PASS", rejection=str(e)))
            return
        raise AssertionError("accepted malformed input: " + name)

    vectors = [
        U((0.0, 0.5, math.nextafter(1.0, 0.0))),
        U((0.0, 0.0, 0.0)),
        U((5e-324, 1e-300, 0.5)),
    ]
    vectors += [
        P((1.0, 1.0, 1.0), decay=d, log_noise_scale=s, permutation=p)
        for d in (0.0, 1.0, 6.0)
        for s in (0.0, 1e200, 1e308)
        for p in ((), (2, 0, 1))
    ]
    vectors += [
        P((1e308, 1e308, 1e308), decay=1.0, log_noise_scale=1e308),
        P((0.0, -1000.0, -1001.0), decay=0.0, log_noise_scale=1.0),
        P((-1e308, 1e308, 0.0), decay=1e308, log_noise_scale=1e308),
    ]
    rng = random.Random(28)
    vectors += [
        P(
            tuple(rng.gauss(0, 1) for _ in range(3)),
            rng.uniform(0, 6),
            10 ** rng.uniform(-4, 1),
            tuple(rng.sample(range(3), 3)),
        )
        for _ in range(60)
    ]
    gs = [
        (3.0, -4.0, 5.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, -1.0),
        (1e308, -1e308, 1e308),
        (5e-324, -1e-323, 2e-323),
        (1e-300, 1e-200, -1e300),
    ]
    for index, v in enumerate(vectors):
        record(f"vector-{index}", list(v.values()), oracle(v), asdict(v))
        for gi, g in enumerate(gs):
            for eps in (0.0, 5e-324, 1e-6, 1e308):
                obj = I((g,), (v,), eps)
                record(
                    f"matrix-{index}-{gi}-{eps}",
                    list(obj.materialize()[0]),
                    oracle(v, g, eps),
                    dict(vector=asdict(v), g=g, epsilon=eps),
                )
    for value in (float("nan"), float("inf"), -float("inf"), True, "x"):
        reject("uniform draw " + str(value), lambda value=value: U((value,)))
        reject("polynomial draw " + str(value), lambda value=value: P((value,)))
        reject("epsilon " + str(value), lambda value=value: I(epsilon=value))
        reject("gaussian " + str(value), lambda value=value: I(gaussian=((value,),)))
    for draws in ((), (1.0,), (-0.1,), [0.5]):
        reject("uniform malformed " + str(draws), lambda draws=draws: U(draws))
    for kwargs in (
        {"decay": -1},
        {"log_noise_scale": -1},
        {"permutation": (0, 0)},
        {"permutation": (True,)},
        {"exact_scores_limit": 0},
    ):
        reject("polynomial malformed " + str(kwargs), lambda kwargs=kwargs: P(**kwargs))
    reject("exact resource limit", lambda: P((0.0, 0.0), exact_scores_limit=1).values())
    wc = I(((3.0, 4.0),), (U((0.5, 0.5)),), 0)
    child = MatrixCausalNode(
        parents=(0,),
        weights=((0.6, 0.8),),
        bias=(0.0, 0.0),
        noise_scale=(0.0, 0.0),
        weight_construction=wc,
    )
    task = LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
    kw = dict(
        task=task,
        nodes=(CausalNode(), child),
        observed=((0, 0),),
        target=((1, 0),),
        version="causal-v2",
    )
    with patch.object(
        I, "materialize", side_effect=AssertionError("materialized before admission")
    ):
        reject(
            "recipe resource admission precedes materialization",
            lambda: CausalRecipe(**kw, limits=CausalLimits(parameters=1)),
        )
    good = CausalRecipe(**kw)
    data = asdict(good)
    data["nodes"][1]["weights"] = ((0.7, 0.8),)
    reject("disagreeing saved projection", lambda data=data: recipe_from_dict(data))
    for profile, wrong in [
        ("causal-importance-v1", "activation-factorized-v1"),
        ("causal-factorized-v1", "activation-importance-v1"),
    ]:
        recipe = sample_recipe(
            "redirection",
            task=LinearRecipe(rows=16, context_rows=8),
            seed=28,
            task_id=10,
            profile=profile,
        )
        data = asdict(recipe)
        data["construction"] = wrong
        reject("family provenance " + profile, lambda data=data: recipe_from_dict(data))
    receipt = {
        "status": "PASS",
        "checks": len(events),
        "python": sys.executable,
        "public_import": inspect.getfile(I),
        "module_sha256": hashlib.sha256(Path(inspect.getfile(I)).read_bytes()).hexdigest(),
        "events": events[-40:],
    }
    (out / "numeric-summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "events"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path)
    a = p.parse_args()
    run(a.output_dir)
