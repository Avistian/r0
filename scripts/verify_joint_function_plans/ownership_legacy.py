"""Additional ownership mutations and exact numerical compatibility at the base."""

import ast
import copy
import hashlib
import json
import math
from dataclasses import asdict
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from verification import INPUTS, prepare

from r0.modules import priors as p

OUT, source_identity = prepare(__file__, p)
source = (INPUTS / "base_parameter_preferences.txt").read_text()
(OUT / "base_parameter_preferences.txt").write_text(source)
node = next(
    n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_values"
)
namespace = {
    "NumericalParameter": p.NumericalParameter,
    "Decimal": Decimal,
    "localcontext": localcontext,
    "math": math,
}
exec(
    compile(ast.Module(body=[node], type_ignores=[]), "<retained-base-_values>", "exec"), namespace
)
units = (0.0, 0.125, 0.5, 0.875, 1.0, *map(float, np.random.default_rng(482).random(32)))
comparisons = 0
for integer in ("none", "floor", "round"):
    for scale in ("linear", "log"):
        for boundary in (False, True):
            request = p.NumericalParameter(
                "legacy", 2, 8, integer=integer, scale=scale, boundary_mass=boundary
            )
            for u in units:
                expected = namespace["_values"](request, (u,))
                # Public current constructor recomputes and rejects any mismatch.
                p.NumericalDraw(request, 0, (u,), expected)
                comparisons += 1
print("BASE_VALUE_COMPATIBILITY", comparisons, "exact scalar values including boundary cases")


def rejected(payload, record, label):
    try:
        p.joint_function_plan_from_dict(payload, decisions=record)
    except ValueError as exc:
        print("REJECTED", label, str(exc))
    else:
        raise AssertionError("invalid ownership accepted: " + label)


fixtures = []
for family in ("lin", "quad", "disc", "em", "mlp", "prod", "gp"):
    # Seed zero global gives radial GP; use local elsewhere for differing children.
    context = p.JointSamplingContext(
        np.random.default_rng(0), mode="global" if family == "gp" else "local"
    )
    plan = p.sample_joint_function(
        context,
        "f",
        input_width=3,
        output_width=2,
        calibration_rows=9,
        prior=p.JointFunctionPrior(
            functions=family, matrix=p.JointMatrixPrior(families=("gaussian",))
        ),
    )
    record = context.freeze()
    payload = json.loads(json.dumps(asdict(plan)))
    fixtures.append({"family": family, "plan": payload, "record": asdict(record)})
    if plan.matrices:
        bad = copy.deepcopy(payload)
        bad["matrices"][0]["operation"] = ["f", "orphan"]
        rejected(bad, record, family + " orphan matrix")
        bad = copy.deepcopy(payload)
        bad["matrices"][0]["columns"] += 1
        rejected(bad, record, family + " changed child geometry")
    if plan.activations:
        bad = copy.deepcopy(payload)
        bad["activations"][0]["operation"] = ["f", "stolen"]
        rejected(bad, record, "MLP activation owner")
    if plan.children:
        bad = copy.deepcopy(payload)
        bad["children"][1] = copy.deepcopy(bad["children"][0])
        rejected(bad, record, "product aliases left as right")
        bad = copy.deepcopy(payload)
        bad["children"].reverse()
        rejected(bad, record, "product reversed children")
        bad = copy.deepcopy(payload)
        bad["children"][0]["children"] = [copy.deepcopy(payload)]
        rejected(bad, record, "product recursive child")
    if plan.weights is not None:
        bad = copy.deepcopy(payload)
        bad["weights"]["operation"] = ["f", "stolen"]
        rejected(bad, record, "GP weight owner")
        bad = copy.deepcopy(payload)
        bad["weights"]["width"] += 1
        rejected(bad, record, "GP weight geometry")
(OUT / "ownership-fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")
for family in ("gaussian", "importance", "factorized", "kernel", "transformed"):
    context = p.JointSamplingContext(np.random.default_rng(92), mode="local")
    plan = p.sample_joint_function(
        context,
        "f",
        input_width=3,
        output_width=2,
        calibration_rows=9,
        prior=p.JointFunctionPrior(functions="lin", matrix=p.JointMatrixPrior(families=(family,))),
    )
    record = context.freeze()
    payload = json.loads(json.dumps(asdict(plan)))
    assert p.joint_function_plan_from_dict(payload, decisions=record) == plan
    print("MATRIX_FAMILY_ROUNDTRIP", family)
identity = {
    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__), OUT / "base_parameter_preferences.txt")
}
(OUT / "supplement-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
print("PASS_COMPONENT ownership/legacy supplement", identity)
