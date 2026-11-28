"""Retained legacy serialization/output comparison through the public priors API."""

import hashlib
import itertools
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r0.modules import priors as p

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
BASE = Path(sys.argv[2]) if len(sys.argv) > 2 else None
SOURCE = (
    "662fb89e098126990c2b6d5a86a4e6c189fc0321"
    if BASE
    else "32c24ec0fde27ec2ae096272f66e584ad3760e4c"
)
results = []


def encode(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.tolist(),
    )


def case(name, inputs, function):
    try:
        result = {"status": "completed", "output": function()}
    except (ValueError, ArithmeticError) as exc:
        result = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
    record = {"case": name, "input": inputs, **result}
    text = encode(record)
    (OUT / (name + ".json")).write_text(text)
    if BASE:
        same = (BASE / (name + ".json")).read_text() == text
        print(encode({"case": name, "byte_equal": same, "status": result["status"]}), flush=True)
        assert same, "Legacy changed: " + name
    else:
        print(encode({"case": name, "status": result["status"]}), flush=True)
    results.append(
        {
            "case": name,
            "status": result["status"],
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    )


def numerical(mode, policy):
    rng = np.random.default_rng(911)
    sampler = p.ParameterSampler(rng, default_mode=mode, global_policy=policy)
    for iteration in range(3):
        for request in (
            p.NumericalParameter("linear", -3.0, 7.0, shape=(2,)),
            p.NumericalParameter("log", 0.01, 10.0, scale="log", boundary_mass=True),
            p.NumericalParameter("integer", -5, 8, integer="floor"),
        ):
            sampler.numerical(request)
        if mode == "global" and policy == "quantile":
            sampler.numerical(p.NumericalParameter("linear", -10.0 - iteration, 17.0, shape=(2,)))
    record = sampler.freeze()
    decoded = p.parameter_preferences_from_dict(json.loads(encode(asdict(record))))
    assert decoded == record
    return {"record": asdict(record), "next_random": rng.random(8).tolist()}


def categorical(mode, policy):
    rng = np.random.default_rng(912)
    sampler = p.CategoricalSampler(rng, default_mode=mode, policy=policy)
    for _ in range(3):
        for name, choices in (("act", ("sin", "relu", "tanh")), ("matrix", ("gaussian", "kernel"))):
            sampler.categorical(p.CategoricalParameter(name, choices))
    record = sampler.freeze()
    assert p.categorical_preferences_from_dict(json.loads(encode(asdict(record)))) == record
    return {"record": asdict(record), "next_random": rng.random(8).tolist()}


def activation(mode, policy, selector):
    rng = np.random.default_rng(913)
    sampler = p.SourceActivationSampler(rng, mode=mode, policy=policy)
    acts = [asdict(sampler.sample(selector)) for _ in range(8)]
    record = sampler.freeze()
    assert p.source_activation_selections_from_dict(json.loads(encode(asdict(record)))) == record
    return {"activations": acts, "record": asdict(record), "next_random": rng.random(8).tolist()}


def matrices(mode, policy, selector, family):
    base = p.sample_causal_recipe(
        p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
        task=p.LinearRecipe(
            rows=16, context_rows=8, features=2, signal_features=2, active_features=2
        ),
        seed=914,
        task_id=0,
        widths=(2, 1, 2, 1),
    )
    recipe = p.with_source_matrices(
        base,
        seed=914,
        task_id=0,
        prior=p.SourceActivationPrior(
            selector=selector, mode=mode, policy=policy, placement="both", calibration_rows=32
        ),
        matrix_prior=p.CausalMatrixPrior(
            mode=mode, categorical_policy=policy, families=(family,), simple_families=("gaussian",)
        ),
    )
    serialized = asdict(recipe)
    assert p.recipe_from_dict(json.loads(encode(serialized))) == recipe
    task = p.generate_task(recipe, seed=914, task_id=0)
    return {"recipe": serialized, "generated_task": asdict(task)}


for mode, policy in itertools.product(("meta", "local", "global"), ("strict", "quantile")):
    case(
        "numerical-" + mode + "-" + policy,
        {"seed": 911, "mode": mode, "policy": policy},
        lambda mode=mode, policy=policy: numerical(mode, policy),
    )
for mode, policy in itertools.product(("meta", "local", "global"), ("shared", "refresh")):
    case(
        "categorical-" + mode + "-" + policy,
        {"seed": 912, "mode": mode, "policy": policy},
        lambda mode=mode, policy=policy: categorical(mode, policy),
    )
    for selector in ("plain", "standard"):
        case(
            "activation-" + mode + "-" + policy + "-" + selector,
            {"seed": 913, "mode": mode, "policy": policy, "selector": selector},
            lambda mode=mode, policy=policy, selector=selector: activation(mode, policy, selector),
        )
        for family in ("gaussian", "importance", "factorized", "kernel", "transformed"):
            case(
                "matrix-" + mode + "-" + policy + "-" + selector + "-" + family,
                {
                    "seed": 914,
                    "mode": mode,
                    "policy": policy,
                    "selector": selector,
                    "family": family,
                },
                lambda mode=mode, policy=policy, selector=selector, family=family: matrices(
                    mode, policy, selector, family
                ),
            )
for family, noise, task_type, seed in itertools.product(
    p.FAMILIES, ("none", "added"), ("regression", "binary", "multiclass"), (11, 28)
):
    proposal = p.Proposal(
        family=family,
        seed=seed,
        noise=noise,
        task_type=task_type,
        rows=16,
        context_rows=8,
        features=3,
    )
    case(
        f"proposal-{family}-{noise}-{task_type}-{seed}",
        asdict(proposal),
        lambda proposal=proposal: asdict(p.sample_proposal(proposal)),
    )

imports = {}
for name, module in sorted(sys.modules.items()):
    filename = getattr(module, "__file__", None)
    if name.startswith("r0.") and filename and Path(filename).is_file():
        imports[name] = {
            "path": filename,
            "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
        }
(OUT / "imports.json").write_text(json.dumps(imports, indent=2))
(OUT / "environment.json").write_text(
    json.dumps(
        {
            "source": SOURCE,
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "priors_file": p.__file__,
        },
        indent=2,
    )
)
totals = {
    "cases": len(results),
    "completed": sum(r["status"] == "completed" for r in results),
    "failures_retained": sum(r["status"] == "failed" for r in results),
    "byte_equal_cases": len(results) if BASE else None,
    "source": SOURCE,
}
(OUT / "totals.json").write_text(json.dumps(totals, indent=2))
(OUT / "case-manifest.json").write_text(json.dumps(results, indent=2))
print(encode(totals))
