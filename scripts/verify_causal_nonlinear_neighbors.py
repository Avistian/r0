#!/usr/bin/env python3
"""Public-API Decimal oracle over neighboring extreme-value parameter cases."""

import json
import math
import subprocess
import tempfile
from decimal import Decimal, localcontext
from pathlib import Path

import r0.modules.priors as priors
from r0.modules.priors import (
    ActivatedCausalNode,
    CausalActivation,
    CausalNode,
    CausalRecipe,
    LinearRecipe,
    generate_task,
)

source = Path(priors.__file__).resolve().parent
head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
out = Path(tempfile.mkdtemp(prefix=f"nonlinear-neighbors-{head[:12]}-"))
controls = LinearRecipe(rows=16, context_rows=8, features=1, signal_features=1, active_features=1)
results = []


def value(kind, x, params):
    node = ActivatedCausalNode(
        bias=(x,), noise_scale=(0.0,), output_activation=CausalActivation(kind, params)
    )
    recipe = CausalRecipe(
        task=controls,
        nodes=(node, CausalNode()),
        observed=((1, 0),),
        target=((0, 0),),
        version="causal-v2",
    )
    return generate_task(recipe, seed=37, task_id=53).targets[0].item()


def check(kind, x, params, expected):
    row = {"kind": kind, "x": x, "parameters": params, "expected": expected}
    try:
        actual = value(kind, x, params)
        row["actual"] = actual
        row["pass"] = expected != "REJECT" and math.isclose(
            actual, expected, rel_tol=5e-13, abs_tol=0
        )
    except ValueError as exc:
        row["error"] = str(exc)
        row["pass"] = expected == "REJECT"
    results.append(row)


with localcontext() as ctx:
    ctx.prec = 400
    maximum = Decimal.from_float(float.fromhex("0x1.fffffffffffffp+1023")).ln()
    minimum = Decimal.from_float(float.fromhex("0x0.0000000000001p-1022")).ln() - Decimal(2).ln()
    for x in [0.0, 1e308, -1e308, 1e-308, -1e-308, 1.0, -1.0]:
        for epsilon in [1e308, 1e-308, 0.001, 1.0]:
            logsum = (Decimal.from_float(abs(x)) + Decimal.from_float(epsilon)).ln()
            check("log_abs", x, (("epsilon", epsilon),), float(logsum))
            for exponent in [1e-308, 0.001, 0.5, 2.0, 1e308]:
                exponentiated = -Decimal.from_float(exponent) * logsum
                expected = (
                    "REJECT"
                    if exponentiated > maximum
                    else 0.0
                    if exponentiated < minimum
                    else float(exponentiated.exp())
                )
                check(
                    "reciprocal_abs_power",
                    x,
                    (("epsilon", epsilon), ("exponent", exponent)),
                    expected,
                )

for kind in ["power", "integer_power"]:
    for exponent in [
        2**53 + 1,
        -(2**53 + 1),
        2**63 + 1,
        -(2**63 + 1),
        10**100 + 1,
        -(10**100 + 1),
        1e308,
    ]:
        for x in [-1.0, 1.0, 0.0]:
            expected = (
                "REJECT"
                if x == 0 and exponent < 0
                else 0.0
                if x == 0
                else -1.0
                if x < 0 and int(exponent) % 2
                else 1.0
            )
            check(kind, x, (("exponent", exponent),), expected)
    check(kind, 1.0, (("exponent", 10**1000),), "REJECT")

report = {
    "head": head,
    "checks": len(results),
    "passed": sum(row["pass"] for row in results),
    "failed": sum(not row["pass"] for row in results),
    "cases": results,
    "evidence_directory": str(out),
}
(out / "neighbors.json").write_text(json.dumps(report, indent=2) + "\n")
(out / "verifier.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(report, indent=2))
raise SystemExit(1 if report["failed"] else 0)
