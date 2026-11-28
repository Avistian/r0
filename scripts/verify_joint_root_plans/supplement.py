"""Additional source-contract, boundary, policy and fault-cutoff probes."""

import ast
import copy
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from verification import INPUTS, prepare

from r0.modules import priors as p

OUT = prepare(p)
RESULTS = []
FIXTURES = Path(os.environ["R0_REVIEW_FIXTURES"])


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def reject(label, call):
    try:
        call()
    except (ValueError, TypeError, KeyError) as exc:
        RESULTS.append({"case": label, "result": "REJECTED", "reason": str(exc)})
    else:
        raise AssertionError("invalid case accepted: " + label)


def root_prior(function, scalarization="scalarized-weight-batches-v1", **kwargs):
    return p.JointRootPrior(
        families=("covariance",),
        function=p.JointFunctionPrior(
            functions=function,
            matrix=p.JointMatrixPrior(families=("gaussian",), scalarization=scalarization),
            **kwargs,
        ),
    )


def sample(ctx, prior, name="root", width=2, rows=4):
    return p.sample_joint_root(ctx, name, width=width, calibration_rows=rows, prior=prior)


def source_requests():
    scope = {
        "self": SimpleNamespace(
            out_features=2, config=SimpleNamespace(max_discretization_cardinality=256)
        ),
        "x": SimpleNamespace(shape=(4, 2)),
        "min": min,
        "max": max,
    }
    contracts = {}
    tree = ast.parse((INPUTS / "I2func.py").read_text())
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "sampler"
            and node.func.attr in ("randint", "numerical", "boolean")
        ):
            continue
        name = ast.literal_eval(node.args[0])
        if node.func.attr == "boolean":
            contracts[name] = {"choices": ["false", "true"], "mode": "meta"}
            continue
        # Evaluate only retained source's scalar bound expressions in a tiny inert namespace.
        bounds = [
            eval(compile(ast.Expression(arg), "source-bound", "eval"), {"__builtins__": {}}, scope)
            for arg in node.args[1:3]
        ]
        keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
        contracts[name] = {
            "low": bounds[0],
            "high": bounds[1],
            "mode": "meta",
            "shape": [],
            "scale": "log" if keywords.get("use_log") else "linear",
            "integer": "source-clip-floor-v1"
            if node.func.attr == "randint"
            else ("round" if keywords.get("use_int") else "none"),
            "boundary_mass": False,
        }
    observed = set()
    files = sorted(FIXTURES.glob("fixture-*.json"))
    assert len(files) == 320
    for path in files:
        data = json.loads(path.read_text())
        for kind in ("numerical", "categorical"):
            for draw in data["decisions"][kind]["draws"]:
                req = draw["request"]
                if req["name"] in contracts:
                    observed.add(req["name"])
                    for field, value in contracts[req["name"]].items():
                        assert req[field] == value, (path.name, req, field, value)
    assert observed == set(contracts)
    save("source-contracts.json", contracts)
    RESULTS.append(
        {
            "case": "AST-derived source request laws",
            "result": "PASS",
            "requests": len(contracts),
            "fixtures": len(files),
        }
    )


def policy_cases():
    for mode in ("local", "meta", "global"):
        for batch in ("source-weight-batches-v1", "scalarized-weight-batches-v1"):
            for policy in ("strict-contract-v1", "scalar-quantile-v1"):
                rng = np.random.default_rng(5)
                ctx = p.JointSamplingContext(rng, mode=mode, global_policy=policy)
                before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
                prior = root_prior("tree", batch)
                label = f"{mode}/{batch}/{policy}"
                if mode == "global" and batch.startswith("source") and policy.startswith("scalar"):
                    reject(label, lambda ctx=ctx, prior=prior: sample(ctx, prior))
                    assert ctx.freeze() == before
                    assert rng.bit_generator.state == state
                    continue
                first = sample(ctx, prior)
                second = sample(ctx, prior, "second")
                assert first.weights.scalarization == batch
                for d in ctx.freeze().numerical.draws[:2]:
                    assert d.request.shape == ((1,) if batch.startswith("source") else ())
                first.validate(ctx.freeze())
                second.validate(ctx.freeze())
                save(
                    "policy-" + label.replace("/", "_") + ".json",
                    {"plans": [asdict(first), asdict(second)], "decisions": asdict(ctx.freeze())},
                )
                if mode == "global" and policy == "strict-contract-v1":
                    before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
                    reject(
                        label + " changed-width strict law",
                        lambda ctx=ctx, prior=prior: sample(ctx, prior, "changed", width=3),
                    )
                    assert ctx.freeze() == before
                    assert rng.bit_generator.state == state
                RESULTS.append({"case": label, "result": "PASS"})


def fault_cutoffs():
    good = root_prior("mlp")
    reference = p.JointSamplingContext(np.random.default_rng(70), mode="local")
    sample(reference, good)
    total = len(reference.freeze().events)
    for cap in range(1, total):
        rng = np.random.default_rng(70)
        ctx = p.JointSamplingContext(rng, mode="local", limits=p.PreferenceLimits(max_draws=cap))
        before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
        reject("named draw cutoff " + str(cap), lambda ctx=ctx: sample(ctx, good))
        assert ctx.freeze() == before
        assert rng.bit_generator.state == state
        # A recovered same-name attempt must be indistinguishable from a fresh control.
        if cap >= 6:
            simpler = root_prior("lin")
            recovered = sample(ctx, simpler)
            control = p.JointSamplingContext(
                np.random.default_rng(70), mode="local", limits=p.PreferenceLimits(max_draws=cap)
            )
            expected = sample(control, simpler)
            assert recovered == expected
            assert ctx.freeze() == control.freeze()
    RESULTS.append(
        {
            "case": "every named-event cutoff",
            "result": "PASS",
            "successful_reference_events": total,
            "failure_cutoffs": total - 1,
        }
    )


def corners():
    for rows, explicit, expected in ((1, None, 1), (2, None, 1), (4, 9, 4), (4, 2, 2)):
        ctx = p.JointSamplingContext(np.random.default_rng(99), mode="local")
        root = sample(ctx, root_prior("disc", discretization_centers=explicit), rows=rows)
        assert root.function.center_count == expected
        names = [d.request.name for d in ctx.freeze().numerical.draws]
        assert ("rand_discr_n_centers" in names) == (rows > 1 and explicit is None)
        save(
            f"disc-{rows}-{explicit}.json",
            {"plan": asdict(root), "decisions": asdict(ctx.freeze())},
        )
    for width in (1, 20, 21):
        ctx = p.JointSamplingContext(np.random.default_rng(2), mode="local")
        root = sample(ctx, root_prior("quad"), width=width)
        matrix = root.function.matrices[0]
        assert (matrix.rows, matrix.columns, matrix.batch) == (min(width, 20) + 1,) * 2 + (width,)
    for bad in (" lin", "lin ", "lin,,gp", "unknown"):
        reject(
            "untrimmed/unknown catalog " + bad, lambda bad=bad: p.JointFunctionPrior(functions=bad)
        )
    for value, expected in ((2.5, 2), (3.5, 4)):
        request = p.NumericalParameter("tie", 0, 8, integer="round")
        d = p.NumericalDraw(request, 0, (value / 8,), (expected,))
        assert d.values == (expected,)
    request = p.NumericalParameter("depth", 1, 8, integer="source-clip-floor-v1")
    assert p.NumericalDraw(request, 0, (1.0,), (7,)).values == (7,)
    # Cover the only three-layer MLP flag combination absent from the initial fixed grid.
    matched = None
    for seed in range(200):
        ctx = p.JointSamplingContext(np.random.default_rng(seed), mode="local")
        root = sample(ctx, root_prior("mlp"))
        v = dict(root.function.parameters)
        if v["layers"] == 3 and v["start_activation"] and v["end_activation"]:
            assert len(root.function.matrices) == 3
            assert len(root.function.activations) == 4
            matched = seed
            save(
                "mlp-three-both-flags.json",
                {"seed": seed, "plan": asdict(root), "decisions": asdict(ctx.freeze())},
            )
            break
    assert matched is not None
    RESULTS.append(
        {
            "case": "discretization/quadratic/catalog/integer/MLP corners",
            "result": "PASS",
            "mlp_three_layer_both_flags_seed": matched,
        }
    )


try:
    source_requests()
    policy_cases()
    fault_cutoffs()
    corners()
    save("report.json", {"verdict": "PASS_COMPONENT", "results": RESULTS})
    print("PASS_COMPONENT supplement", len(RESULTS))
except Exception:
    save(
        "report.json",
        {
            "verdict": "FAIL_REQUIRES_CLASSIFICATION",
            "results": RESULTS,
            "traceback": traceback.format_exc(),
        },
    )
    raise
