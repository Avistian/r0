"""Independent public-interface oracle for the scoped exogenous checkpoint."""

import argparse
import hashlib
import json
import math
import runpy
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import r0.modules.priors as public_priors
from r0.modules.priors import (
    CausalActivation,
    CausalNode,
    CausalRecipe,
    InputDistribution,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
    sample_recipe,
)

if "--legacy-only" not in sys.argv:
    from r0.modules.priors import ExogenousDistribution, NoiseResponse, StochasticCausalNode

checks = []


def record(name, **evidence):
    checks.append({"name": name, "status": "PASS", **evidence})


def controls(rows=64, features=1):
    return LinearRecipe(
        rows=rows,
        context_rows=8,
        features=features,
        signal_features=features,
        active_features=features,
    )


def root(spec=None, width=2, rows=64, normalize=False, **kw):
    if spec is None:
        spec = ExogenousDistribution()
    n = StochasticCausalNode(bias=(0.0,) * width, noise_scale=(1.0,) * width, exogenous=spec, **kw)
    return CausalRecipe(
        task=controls(rows, width - 1),
        nodes=(n,),
        observed=tuple((0, j) for j in range(width - 1)),
        target=((0, width - 1),),
        version="causal-v2",
        normalize=normalize,
    )


def gen(r):
    return generate_task(r, seed=413, task_id=17)


def vals(t):
    return torch.cat((t.inputs, t.scores), dim=1).numpy()


def trace(t, indices):
    return {n["id"]: n for n in explain_generation(t, indices)["nodes"]}


def must_reject(name, fn):
    try:
        fn()
    except (ValueError, TypeError) as e:
        record(name, error=type(e).__name__, message=str(e))
        return
    raise AssertionError(name + " unexpectedly accepted")


def legacy():
    out = {}
    for profile in ("causal-affine-v1", "causal-activated-v1", "causal-preferences-v1"):
        for family in ("forward", "layered", "redirection"):
            for seed in (5, 19):
                r = sample_recipe(family, task=controls(), seed=seed, task_id=6, profile=profile)
                t = generate_task(r, seed=seed, task_id=6)
                # Canonical public recipe/card/tensor bytes; no source implementation oracle.
                h = hashlib.sha256(json.dumps(asdict(r), sort_keys=True).encode())
                h.update(t.mechanism_json.encode())
                for a in (t.inputs, t.scores):
                    h.update(a.numpy().tobytes())
                out[f"{profile}/{family}/{seed}"] = h.hexdigest()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--legacy-baseline")
    a = parser.parse_args()
    result = {
        "status": "RUNNING",
        "checks": checks,
        "python": sys.version,
        "public_import_path": public_priors.__file__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if a.legacy_only:
        result["legacy_signatures"] = legacy()
        result["status"] = "PASS"
        with Path(a.output).open("x") as out:
            json.dump(result, out, indent=2)
        return
    try:
        # Prefix/replay across every supported base law, truncation, geometry and factor.
        for family in ("gaussian", "uniform", "student_t", "mixture"):
            for bound in (None, 1.25):
                spec = ExogenousDistribution(
                    inputs=InputDistribution(
                        distribution=family,
                        df=1.5,
                        location=0.3,
                        scale=0.7,
                        mixture_gap=1.8,
                        bound=bound,
                    ),
                    factor=((1.0, 2.0), (-0.4, 0.5)),
                )
                r = root(spec, rows=257)
                t = gen(r)
                long = gen(replace(r, task=controls(541)))
                assert np.array_equal(vals(t), vals(long)[:257]), (family, bound, "prefix")
                restored = recipe_from_dict(json.loads(json.dumps(asdict(r))))
                assert gen(restored).episode() == t.episode()
                tr = trace(t, [256, 0, 19])
                assert np.array_equal(np.array(tr["causal-0"]["values"]), vals(t)[[256, 0, 19]])
                record("law prefix/replay/selected-original-row", family=family, bound=bound)
        for d in (2, 4):
            r = root(ExogenousDistribution(geometry="ball"), width=d, rows=10000)
            t = gen(r)
            v = vals(t)
            rad = np.linalg.norm(v, axis=1)
            assert rad.max() <= 1 + 1e-14
            u = rad**d
            assert abs(u.mean() - 0.5) < 0.015
            assert abs(np.mean(rad**2) - d / (d + 2)) < 0.015
            assert np.max(abs(v.mean(axis=0))) < 0.025
            long = gen(replace(r, task=controls(10037, d - 1)))
            assert np.array_equal(v, vals(long)[:10000])
            card = json.loads(t.mechanism_json)
            assert "disconnected_target" not in card["flags"]
            record(
                "ball uniform-volume law, dependence, prefix",
                dimension=d,
                mean_radius_power_d=float(u.mean()),
                mean_radius_squared=float(np.mean(rad**2)),
            )
        # Expected matrix orientation and analytic covariance, independent of implementation.
        F = np.array([[1.0, 2.0], [-0.4, 0.5]])
        for family in ("gaussian", "uniform", "mixture"):
            inp = InputDistribution(distribution=family, location=0.3, scale=0.7, mixture_gap=1.8)
            plain = gen(root(ExogenousDistribution(inputs=inp), rows=10000))
            transformed = gen(
                root(
                    ExogenousDistribution(
                        inputs=inp, factor=tuple(tuple(float(v) for v in row) for row in F)
                    ),
                    rows=10000,
                )
            )
            assert np.allclose(vals(transformed), vals(plain) @ F, rtol=1e-14, atol=1e-14)
            expected = (0.7**2) * (1 + (1.8**2 if family == "mixture" else 0)) * (F.T @ F)
            empirical = np.cov(vals(transformed), rowvar=False)
            assert np.max(abs(empirical - expected)) / np.max(abs(expected)) < 0.06
            record(
                "factor right multiplication and analytic covariance",
                family=family,
                expected=expected.tolist(),
                empirical=empirical.tolist(),
            )
        # Independent row arithmetic, both noise positions and nonlinear parent transform.
        for position in ("before_output", "after_output"):
            n0 = StochasticCausalNode(
                exogenous=ExogenousDistribution(inputs=InputDistribution(distribution="uniform"))
            )
            n1 = StochasticCausalNode(
                parents=(0,),
                weights=((2.0,),),
                bias=(1.0,),
                noise_scale=(0.5,),
                input_activation=CausalActivation("square"),
                output_activation=CausalActivation("tanh"),
                noise_position=position,
                noise_response=NoiseResponse(1.0, 0.25, 2.0),
            )
            r = CausalRecipe(
                task=controls(),
                nodes=(n0, n1),
                observed=((0, 0),),
                target=((1, 0),),
                version="causal-v2",
            )
            t = gen(r)
            indices = [0, 7, 31, 63]
            tr = trace(t, indices)
            for row, i in enumerate(indices):
                x = float(t.inputs[i, 0])
                affine = 2 * x * x + 1
                scale = 0.5 * (1 + 0.25 * abs(affine) ** 2)
                z = tr["exogenous-1"]["values"][row][0]
                innovation = scale * z
                expected = (
                    math.tanh(affine + innovation)
                    if position == "before_output"
                    else math.tanh(affine) + innovation
                )
                assert math.isclose(tr["noise-scale-1"]["values"][row][0], scale, rel_tol=1e-13)
                assert math.isclose(tr["innovation-1"]["values"][row][0], innovation, rel_tol=1e-13)
                assert math.isclose(float(t.scores[i, 0]), expected, rel_tol=1e-13, abs_tol=1e-13)
            edited = gen(replace(r, nodes=(n0, replace(n1, noise_response=NoiseResponse()))))
            assert trace(edited, indices)["exogenous-1"]["values"] == tr["exogenous-1"]["values"]
            assert edited.source_task_id != t.source_task_id
            record(
                "independent nonlinear heteroscedastic equation and matched draws",
                position=position,
            )
        # Tail criterion E|T_df|^p finite iff p<df, then growth transforms.
        for kind, power in (
            ("identity", 1.0),
            ("square", 2.0),
            ("sqrt_abs", 0.5),
            ("signed_power", 3.0),
        ):
            activation = (
                CausalActivation(kind, (("exponent", power),))
                if kind == "signed_power"
                else CausalActivation(kind)
            )
            for df in (2 * power, 2 * power + 0.25):
                spec = ExogenousDistribution(
                    inputs=InputDistribution(distribution="student_t", df=df)
                )
                r = root(spec, output_activation=activation)
                card = json.loads(gen(r).mechanism_json)
                assert card["moment_contract"]["target_orders"] == [df / power]
                if df == 2 * power:
                    must_reject(
                        "infinite second-moment boundary " + kind,
                        lambda r=r: replace(r, normalize=True),
                    )
                else:
                    gen(replace(r, normalize=True))
                    record("finite second moment " + kind, df=df)
        for kind in ("tanh", "sin", "log_abs", "reciprocal_abs_power"):
            r = root(
                ExogenousDistribution(inputs=InputDistribution(distribution="student_t", df=0.75)),
                normalize=True,
                output_activation=CausalActivation(kind),
            )
            assert json.loads(gen(r).mechanism_json)["moment_contract"]["target_orders"] == ["all"]
            record("bounded or slow-growth descendant certified", kind=kind)
        for kind in ("exp", "log"):
            r = root(output_activation=CausalActivation(kind))
            must_reject("unknown integrability " + kind, lambda r=r: replace(r, normalize=True))
        # Heteroscedastic tail is min(df_noise,df_parent/power) when independent.
        for p, df in ((2.0, 4.0), (2.0, 4.5), (0.5, 1.0), (0.5, 1.25)):
            n0 = StochasticCausalNode(
                exogenous=ExogenousDistribution(
                    inputs=InputDistribution(distribution="student_t", df=df)
                )
            )
            n1 = StochasticCausalNode(
                parents=(0,),
                weights=((1.0,),),
                noise_response=NoiseResponse(0.0, 1.0, p),
                noise_position="after_output",
                output_activation=CausalActivation("tanh"),
            )
            r = CausalRecipe(
                task=controls(),
                nodes=(n0, n1),
                observed=((0, 0),),
                target=((1, 0),),
                version="causal-v2",
            )
            order = json.loads(gen(r).mechanism_json)["moment_contract"]["target_orders"][0]
            assert order == df / p
            if order <= 2:
                must_reject(
                    "heteroscedastic infinite second moment", lambda r=r: replace(r, normalize=True)
                )
            else:
                gen(replace(r, normalize=True))
                record("heteroscedastic finite second moment", order=order)
        # Singular/zero factors, zero responses and deterministic nonlinear targets.
        for factor, response in (
            (((1.0, 2.0), (0.0, 0.0)), NoiseResponse(0.0, 0.0, 2.0)),
            (((0.0, 0.0), (0.0, 0.0)), NoiseResponse()),
        ):
            r = root(ExogenousDistribution(factor=factor), normalize=True, noise_response=response)
            r = replace(
                r,
                nodes=(
                    replace(r.nodes[0], bias=(1.0, 2.0), output_activation=CausalActivation("exp")),
                ),
            )
            t = gen(r)
            assert np.allclose(vals(t), [[math.e, math.e**2]] * 64)
            c = json.loads(t.mechanism_json)
            assert c["moment_contract"]["target_orders"] == ["all"]
            assert "zero_target_dispersion" in c["flags"]
            assert "disconnected_target" in c["flags"]
            record(
                "constant exp target identity normalization",
                factor=factor,
                response=asdict(response),
            )
        for kwargs in (
            {"factor": [[1.0]]},
            {"factor": ((1.0, 2.0),)},
            {"factor": ((True,),)},
            {"factor": ((math.inf,),)},
            {"geometry": "sphere"},
            {"geometry": "ball", "inputs": InputDistribution(scale=2.0)},
            {"inputs": {}},
        ):
            must_reject(
                "invalid exogenous descriptor", lambda kw=kwargs: ExogenousDistribution(**kw)
            )
        for kwargs in (
            {"offset": -1},
            {"strength": -1},
            {"power": 0},
            {"offset": True},
            {"power": math.nan},
        ):
            must_reject("invalid noise response", lambda kw=kwargs: NoiseResponse(**kw))
        must_reject("factor width mismatch", lambda: root(ExogenousDistribution(factor=((1.0,),))))
        r = root(ExogenousDistribution(factor=((1.0, 2.0), (0.0, 0.0))), rows=16)
        c = json.loads(gen(r).mechanism_json)["resource_admission"]
        assert c["parameters"] == 15
        # Six total-width buffers, no parents, feature+target buffers, eight extra widths.
        expected = 8 * 16 * (6 * 2 + 2 * 1 + 4 * 1 + 4 + 8 * 2) + 128 * 15
        assert c["estimated_working_bytes"] == expected, c
        must_reject(
            "working-memory admission boundary",
            lambda: replace(r, limits=replace(r.limits, working_bytes=expected - 1)),
        )
        record("independent resource count", parameters=15, working_bytes=expected)
        for family in ("forward", "layered", "redirection"):
            r = sample_recipe(
                family, task=controls(), seed=7, task_id=2, profile="causal-exogenous-v1"
            )
            saved = json.loads(json.dumps(asdict(r)))
            restored = recipe_from_dict(saved)
            assert asdict(restored) == asdict(r)
            t = gen(r)
            assert gen(restored).episode() == t.episode()
            assert all(type(n) is StochasticCausalNode for n in r.nodes)
            record("sampled frozen profile serialized/replayed", family=family, nodes=len(r.nodes))
        current = legacy()
        result["legacy_signatures"] = current
        if a.legacy_baseline:
            assert current == json.loads(Path(a.legacy_baseline).read_text())["legacy_signatures"]
            record(
                "legacy exact canonical recipes/cards/tensors across all three profiles",
                cases=len(current),
            )
        # Author-attributed hint, independently reproduced through public constructors.
        # This equation is deterministic: scale=0 implies innovation=0 and value=1e200.
        result["author_hint_probe"] = {
            "attribution": "author code-reading hint relayed by parent",
            "expected": "finite deterministic target 1e200; exact zero structural innovation",
        }
        n0 = StochasticCausalNode(
            bias=(1e200,), noise_scale=(0.0,), noise_response=NoiseResponse(1.0, 1.0, 2.0)
        )
        r = CausalRecipe(
            task=controls(),
            nodes=(CausalNode(), n0),
            observed=((0, 0),),
            target=((1, 0),),
            version="causal-v2",
        )
        t = gen(r)
        assert torch.all(t.scores == 1e200)
        record("zero structural scale removes irrelevant overflowing response")
        repair = runpy.run_path(str(Path(__file__).with_name("verify_causal_exogenous_repair.py")))
        repair["repair_boundaries"](root, gen, trace, controls, vals, must_reject, record)
        repair["decimal_scale_grid"](root, gen, trace, record)
        result["status"] = "PASS"
    except Exception as e:
        result["status"] = "REJECT"
        result["failure"] = {"type": type(e).__name__, "message": str(e)}
        import traceback

        result["failure"]["traceback"] = traceback.format_exc()
    with Path(a.output).open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": len(checks),
                "failure": result.get("failure"),
                "output": a.output,
            },
            indent=2,
        )
    )
    if result["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
