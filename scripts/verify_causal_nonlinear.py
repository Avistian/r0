#!/usr/bin/env python3
"""Independent public-API audit; writes a fresh temporary evidence directory.

Run with PYTHONPATH pointing to the reviewed checkout src directory.
The default repository is the current directory; --repo selects another checkout.
Requires assertions enabled. No full-issue or environment acceptance is implied.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from pathlib import Path

import torch

import r0.modules.priors as priors_module
from r0.modules.priors import (
    ActivatedCausalNode,
    CausalActivation,
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    explain_generation,
    generate_task,
    recipe_from_dict,
)

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("--base", default="23f030212dc6d90afe6041d78191585ac7a2b2f2")
args = parser.parse_args()
repo = Path(args.repo).resolve()
if not __debug__:
    raise RuntimeError("Run without -O; this verifier requires assertions")
if not Path(priors_module.__file__).resolve().is_relative_to(repo / "src"):
    raise RuntimeError("Set PYTHONPATH to the requested repository src directory")
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
out = Path(tempfile.mkdtemp(prefix=f"verify-causal-nonlinear-{head[:12]}-", dir="/tmp"))

torch.set_num_threads(1)
records = []


def check(name, fn):
    try:
        detail = fn()
        records.append({"name": name, "status": "PASS", "detail": detail})
    except Exception as exc:
        records.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})


controls = LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)


def root_recipe(kind, values, params=()):
    return CausalRecipe(
        task=controls,
        nodes=(
            ActivatedCausalNode(
                bias=tuple(values),
                noise_scale=(0.0,) * len(values),
                output_activation=CausalActivation(kind, params),
            ),
            CausalNode(),
        ),
        observed=((1, 0),),
        target=((0, 0),),
        version="causal-v2",
    )


def evaluate(kind, values, params=()):
    recipe = root_recipe(kind, values, params)
    wire = json.loads(json.dumps(asdict(recipe)))
    original = copy.deepcopy(wire)
    restored = recipe_from_dict(wire)
    assert wire == original, "decoder mutated caller record"
    task = generate_task(restored, seed=719, task_id=7)
    assert task.targets.dtype == torch.float64
    trace = explain_generation(task, [0, 7, 15])
    assert trace["target_max_error"] < 1e-12
    vals = next(n["values"][0] for n in trace["nodes"] if n["id"] == "causal-0")
    return vals


def close(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected, strict=True):
        assert math.isclose(a, e, rel_tol=2e-13, abs_tol=1e-13), f"actual={a}, expected={e}"


scalar = {
    "identity": ([-2, 0, 2], [-2, 0, 2]),
    "relu": ([-2, 0, 2], [0, 0, 2]),
    "tanh": ([-2, 0, 2], [math.tanh(-2), 0, math.tanh(2)]),
    "sigmoid": ([-2, 0, 2], [1 / (1 + math.exp(2)), 0.5, 1 / (1 + math.exp(-2))]),
    "leaky_relu": ([-2, 0, 2], [-0.02, 0, 2]),
    "elu": ([-2, 0, 2], [math.expm1(-2), 0, 2]),
    "selu": (
        [-2, 0, 2],
        [1.0507009873554805 * 1.6732632423543772 * math.expm1(-2), 0, 2 * 1.0507009873554805],
    ),
    "silu": ([-2, 0, 2], [-2 / (1 + math.exp(2)), 0, 2 / (1 + math.exp(-2))]),
    "softplus": ([-2, 0, 2], [math.log1p(math.exp(x)) for x in [-2, 0, 2]]),
    "smooth_relu": ([-2, 0, 2], [math.log1p(math.exp(x)) for x in [-2, 0, 2]]),
    "relu6": ([-1, 0, 6, 7], [0, 0, 6, 6]),
    "hardtanh": ([-2, -1, 0, 1, 2], [-1, -1, 0, 1, 1]),
    "sign": ([-1, 0, 1], [-1, 0, 1]),
    "sign_positive": ([-1, 0, 1], [-1, 1, 1]),
    "gaussian_bump": ([-2, 0, 2], [math.exp(-4), 1, math.exp(-4)]),
    "exp": ([-2, 0, 2], [math.exp(-2), 1, math.exp(2)]),
    "sqrt_abs": ([-4, 0, 4], [2, 0, 2]),
    "interval_abs_le_1": ([-2, -1, 0, 1, 2], [0, 1, 1, 1, 0]),
    "interval_0_1": ([-1, 0, 0.5, 1, 2], [0, 1, 1, 1, 0]),
    "sin": ([-math.pi / 2, 0, math.pi / 2], [-1, 0, 1]),
    "square": ([-2, 0, 2], [4, 0, 4]),
    "abs": ([-2, 0, 2], [2, 0, 2]),
    "log": ([0.5, 1, 2], [math.log(0.5), 0, math.log(2)]),
    "log_abs": ([-2, 0, 2], [math.log(2.000001), math.log(0.000001), math.log(2.000001)]),
    "log_sigmoid": (
        [-2, 0, 2],
        [-math.log1p(math.exp(2)), -math.log(2), -math.log1p(math.exp(-2))],
    ),
    "round": ([-2.5, -1.5, 0.5, 1.5, 2.5], [-2, -2, 0, 2, 2]),
    "step": ([-1, 0, 1], [0, 0, 1]),
    "heaviside": ([-1, 0, 1], [0, 0, 1]),
    "heaviside_positive": ([-1, 0, 1], [0, 1, 1]),
    "modulo": ([-1.25, 0, 1.25], [0.75, 0, 0.25]),
    "gelu": ([-2, 0, 2], [x * 0.5 * (1 + math.erf(x / math.sqrt(2))) for x in [-2, 0, 2]]),
    "power": ([-2, 0, 2], [4, 0, 4]),
    "integer_power": ([-2, 0, 2], [4, 0, 4]),
    "relu_power": ([-2, 0, 2], [0, 0, 4]),
    "signed_power": ([-2, 0, 2], [-4, 0, 4]),
    "reciprocal_abs_power": ([-2, 0, 2], [2.001**-2, 0.001**-2, 2.001**-2]),
    "log_clip_abs": ([-2, 0, 2], [math.log(2), math.log(1e-6), math.log(2)]),
    "softmax": ([0, math.log(3)], [0.25, 0.75]),
    "tied_max": ([2, 2, -1, 0], [1, 1, 0, 0]),
    "argmax": ([2, 2, -1, 0], [1, 0, 0, 0]),
    "rank": ([2, 2, -1, 0], [2, 3, 0, 1]),
    "argsort": ([2, 2, -1, 0], [2, 3, 0, 1]),
}
for kind, (values, expected) in scalar.items():
    check(
        "equation_and_boundary_" + kind,
        lambda k=kind, v=values, e=expected: close(evaluate(k, v), e),
    )

parameter_cases = [
    (
        "identity",
        [2],
        (("input_scale", -2), ("input_shift", 1), ("output_scale", 3), ("output_shift", -4)),
        [-13],
    ),
    ("leaky_relu", [-2], (("negative_slope", 0.3),), [-0.6]),
    ("elu", [-2], (("alpha", 2),), [2 * math.expm1(-2)]),
    ("softplus", [-2, 0, 2], (("beta", 2),), [math.log1p(math.exp(2 * x)) / 2 for x in [-2, 0, 2]]),
    (
        "smooth_relu",
        [-2, 0, 2],
        (("beta", 2),),
        [math.log1p(math.exp(2 * x)) / 2 for x in [-2, 0, 2]],
    ),
    ("power", [4], (("exponent", 0.5),), [2]),
    ("integer_power", [-2], (("exponent", -3),), [-0.125]),
    ("relu_power", [-4, 4], (("exponent", 0.5),), [0, 2]),
    ("signed_power", [-4, 4], (("exponent", 0.5),), [-2, 2]),
    ("log_abs", [-2, 0], (("epsilon", 2),), [math.log(4), math.log(2)]),
    ("log_clip_abs", [-0.5, 2], (("epsilon", 1),), [0, math.log(2)]),
    ("reciprocal_abs_power", [-2, 0], (("epsilon", 1), ("exponent", 0.5)), [1 / math.sqrt(3), 1]),
    ("modulo", [-3.25, 4], (("period", 2),), [0.75, 0]),
]
for kind, values, params, expected in parameter_cases:
    check(
        "parameters_" + kind,
        lambda k=kind, v=values, p=params, e=expected: close(evaluate(k, v, p), e),
    )


def reject(fn):
    try:
        fn()
    except ValueError:
        return "ValueError"
    raise AssertionError("invalid request accepted")


invalid = [
    ("identity", (("x", 1),)),
    ("identity", (("beta", 1),)),
    ("identity", (("input_scale", 1), ("input_scale", 2))),
    ("identity", (("output_shift", float("nan")),)),
    ("identity", (("output_scale", True),)),
    ("softplus", (("beta", 0),)),
    ("log_abs", (("epsilon", 0),)),
    ("modulo", (("period", -1),)),
    ("integer_power", (("exponent", 0.5),)),
    ("signed_power", (("exponent", 0),)),
    ("relu_power", (("exponent", -1),)),
    ("reciprocal_abs_power", (("exponent", 0),)),
    ("identity", [("input_scale", 1)]),
    ("identity", (("input_scale",),)),
]
for i, (kind, params) in enumerate(invalid):
    check(
        f"parameter_rejection_{i}", lambda k=kind, p=params: reject(lambda: CausalActivation(k, p))
    )
for kind, value, params in [
    ("log", 0, ()),
    ("log", -1, ()),
    ("power", -1, (("exponent", 0.5),)),
    ("power", 0, (("exponent", -1),)),
    ("exp", 1000, ()),
    ("identity", 1e308, (("input_scale", 2),)),
    ("identity", 1e308, (("output_scale", 2),)),
]:
    check(
        f"domain_or_overflow_{kind}_{value}_{params}",
        lambda k=kind, v=value, p=params: reject(lambda: evaluate(k, [v], p)),
    )


def noise_and_input_order():
    details = []
    for position in ["before_output", "after_output"]:
        nodes = (
            CausalNode(bias=(-2.0, 3.0), noise_scale=(0.0, 0.0)),
            ActivatedCausalNode(
                parents=(0,),
                weights=((2.0,), (-1.0,)),
                bias=(1.0,),
                noise_scale=(0.8,),
                input_activation=CausalActivation("square"),
                output_activation=CausalActivation("tanh"),
                noise_position=position,
            ),
        )
        r = CausalRecipe(
            task=controls, nodes=nodes, observed=((0, 0),), target=((1, 0),), version="causal-v2"
        )
        t = generate_task(r, seed=113, task_id=13)
        trace = {n["id"]: n for n in explain_generation(t, list(range(16)))["nodes"]}
        errors = []
        for i in range(16):
            e = trace["innovation-1"]["values"][i][0]
            expected = math.tanh(e) if position == "before_output" else e
            close([t.targets[i].item()], [expected])
            errors.append(abs(t.targets[i].item() - expected))
        details.append({"position": position, "max_error": max(errors)})
    return details


check("input_projection_output_noise_order", noise_and_input_order)


def ancestry():
    details = []
    for kind in ["softmax", "tied_max", "argmax", "rank", "argsort"]:
        for stage in ["input", "output"]:
            for position in ["before_output", "after_output"]:
                node = ActivatedCausalNode(
                    parents=(0,),
                    weights=((1.0, 0.0), (0.0, 1.0)),
                    bias=(0.0, 0.0),
                    noise_scale=(0.2, 0.2),
                    input_activation=CausalActivation(kind if stage == "input" else "identity"),
                    output_activation=CausalActivation(kind if stage == "output" else "identity"),
                    noise_position=position,
                )
                r = CausalRecipe(
                    task=controls,
                    nodes=(CausalNode(bias=(0.0, 0.0), noise_scale=(1.0, 1.0)), node),
                    observed=((0, 1),),
                    target=((1, 0),),
                    version="causal-v2",
                )
                assert json.loads(generate_task(r, seed=1, task_id=1).mechanism_json)["active"] == [
                    0
                ]
                details.append([kind, stage, position])
    for position in ["before_output", "after_output"]:
        n = ActivatedCausalNode(
            bias=(0.0, 0.0),
            noise_scale=(1.0, 1.0),
            output_activation=CausalActivation("softmax"),
            noise_position=position,
        )
        r = CausalRecipe(
            task=controls, nodes=(n,), observed=((0, 0),), target=((0, 1),), version="causal-v2"
        )
        t = generate_task(r, seed=3, task_id=4)
        expected = [0] if position == "before_output" else []
        assert json.loads(t.mechanism_json)["active"] == expected
        if position == "before_output":
            assert torch.allclose(t.inputs[:, 0] + t.targets, torch.ones(16, dtype=torch.float64))
    for activation in [
        CausalActivation("identity", (("input_scale", 0),)),
        CausalActivation("tanh", (("output_scale", 0),)),
        CausalActivation("power", (("exponent", 0),)),
    ]:
        for stage in ["input", "output"]:
            node = ActivatedCausalNode(
                parents=(0,),
                weights=((1.0,),),
                bias=(0.0,),
                noise_scale=(1.0,),
                input_activation=activation if stage == "input" else CausalActivation(),
                output_activation=activation if stage == "output" else CausalActivation(),
            )
            r = CausalRecipe(
                task=controls,
                nodes=(CausalNode(), node),
                observed=((0, 0),),
                target=((1, 0),),
                version="causal-v2",
            )
            assert json.loads(generate_task(r, seed=5, task_id=2).mechanism_json)["active"] == []
    return {"channel_cases": details, "constant_and_post_noise_checks": "PASS"}


check("channel_ancestry_and_post_activation_noise", ancestry)


def replay_calibration():
    n = ActivatedCausalNode(
        parents=(0,),
        weights=((2.0,),),
        bias=(0.4,),
        noise_scale=(0.3,),
        output_activation=CausalActivation("tanh"),
    )
    r = CausalRecipe(
        task=controls,
        nodes=(CausalNode(), n),
        observed=((0, 0),),
        target=((1, 0),),
        normalize=True,
        version="causal-v2",
    )
    t = generate_task(r, seed=113, task_id=18)
    replay = generate_task(
        recipe_from_dict(json.loads(json.dumps(asdict(r)))), seed=113, task_id=18
    )
    assert t.episode() == replay.episode()
    assert t.mechanism_json == replay.mechanism_json
    calib = json.loads(t.mechanism_json)["calibration"]
    for rows, ctx, noise in [(16, 4, 0), (32, 8, 0), (32, 16, 0.5)]:
        other = generate_task(
            replace(r, task=replace(controls, rows=rows, context_rows=ctx, noise_ratio=noise)),
            seed=113,
            task_id=18,
        )
        assert json.loads(other.mechanism_json)["calibration"] == calib
        assert torch.equal(other.scores[:16], t.scores)
    return calib


check("replay_query_independent_calibration", replay_calibration)


def admission():
    r = root_recipe("tanh", [1], (("input_scale", 2), ("output_shift", 1)))
    t = generate_task(r, seed=1, task_id=1)
    card = json.loads(t.mechanism_json)
    assert card["resource_admission"]["parameters"] == 6
    reject(lambda: replace(r, limits=replace(r.limits, parameters=5)))
    reject(lambda: replace(r, version="causal-v1"))
    reject(lambda: ActivatedCausalNode(input_activation=CausalActivation("tanh")))
    return card["resource_admission"]


check("parameter_admission_and_version_gate", admission)


def finite_reciprocal_oracle():
    with localcontext() as ctx:
        ctx.prec = 80
        expected = float(((Decimal("1e308") + Decimal("1e308")).ln() * Decimal("-.001")).exp())
    actual = evaluate("reciprocal_abs_power", [1e308], (("epsilon", 1e308), ("exponent", 0.001)))[0]
    assert math.isclose(actual, expected, rel_tol=1e-12), (
        f"actual={actual}; 80-digit Decimal oracle={expected}"
    )


check("extreme_finite_reciprocal_power", finite_reciprocal_oracle)


def finite_additive_log_oracle():
    with localcontext() as ctx:
        ctx.prec = 80
        expected = float((Decimal("1e308") + Decimal("1e308")).ln())
    try:
        actual = evaluate("log_abs", [1e308], (("epsilon", 1e308),))[0]
    except ValueError as exc:
        raise AssertionError(
            f"explicit rejection, not silent corruption: {exc}; "
            f"finite 80-digit Decimal oracle={expected}"
        ) from exc
    assert math.isclose(actual, expected, rel_tol=1e-12), f"actual={actual}; oracle={expected}"


check("extreme_finite_additive_log", finite_additive_log_oracle)


def large_integer_oracle():
    exponent = 2**53 + 1
    expected = -1 if exponent % 2 else 1
    actual = evaluate("integer_power", [-1.0], (("exponent", exponent),))[0]
    assert actual == expected, (
        f"exponent={exponent}; actual={actual}; integer parity oracle={expected}"
    )


check("exact_integer_power_parity", large_integer_oracle)

AFFINE = """import json,hashlib,torch
from dataclasses import asdict
from r0.modules.priors import LinearRecipe,sample_recipe,generate_task,recipe_from_dict
torch.set_num_threads(1)
results=[]
for family in ('forward','layered','redirection'):
 for kind,classes in [('regression',0),('binary',2),('multiclass',3)]:
  for seed in [3,19]:
   controls=LinearRecipe(rows=32,context_rows=12,task_type=kind,classes=classes)
   r=sample_recipe(family,profile='causal-affine-v1',task=controls,seed=seed,task_id=4)
   t=generate_task(r,seed=seed,task_id=4)
   entry={'family':family,'kind':kind,'seed':seed,'recipe':asdict(r),'source':t.source_task_id,'episode':t.episode_id,'recipe_json':t.recipe_json,'mechanism_json':t.mechanism_json}
   for field in ['inputs','targets','scores','probabilities','context_indices','query_indices']:
    x=getattr(t,field)
    entry[field]=None if x is None else hashlib.sha256(x.numpy().tobytes()).hexdigest()
   results.append(entry)
print(json.dumps(results,sort_keys=True))
"""


def affine_preservation():
    baseline = out / "baseline"
    baseline.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", args.base, "src"], capture_output=True, check=True
    )
    subprocess.run(["tar", "-x", "-C", str(baseline)], input=archive.stdout, check=True)
    outputs = []
    for label, src in [("base", baseline / "src"), ("head", repo / "src")]:
        env = dict(os.environ, PYTHONPATH=str(src), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [sys.executable, "-B", "-c", AFFINE],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        (out / f"affine-{label}.json").write_text(result.stdout)
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1], "base/head public outputs or serialized metadata differ"
    return {
        "cases": len(outputs[0]),
        "exact_fields": "recipe, recipe_json, mechanism_json, ids, all tensor bytes",
        "base": args.base,
    }


check("exact_affine_v1_base_head_preservation", affine_preservation)

report = {
    "head": head,
    "base": args.base,
    "repo": str(repo),
    "evidence_directory": str(out),
    "scope": "public-API verification of partial nonlinear slice; no full-issue closure",
    "python": sys.executable,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "records": records,
    "passed": sum(r["status"] == "PASS" for r in records),
    "failed": sum(r["status"] == "FAIL" for r in records),
    "not_run": [
        "GPU",
        "browser",
        "Colab",
        "cloud",
        "full source inventory acceptance",
        "production scale",
    ],
}
report["head_unchanged"] = (
    subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    == head
)
(out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
(out / "verifier.py").write_bytes(Path(__file__).read_bytes())
(out / "report.md").write_text(
    f"HEAD {head}\n\nPASS {report['passed']}; FAIL {report['failed']}\n\n"
    + "\n".join(f"- {r['name']}: {r['status']} {r.get('error', '')}" for r in records)
    + "\n"
)
print(json.dumps(report, indent=2))
sys.exit(1 if report["failed"] or not report["head_unchanged"] else 0)
