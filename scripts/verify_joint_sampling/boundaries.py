"""Raw JSON boundary adversaries and independent scalar-law reconstruction."""

import copy
import hashlib
import json
import math
import sys
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from r0.modules import priors as p

OUT = Path(sys.argv[1])
OUT.mkdir(exist_ok=True)
checks = 0
rejections = 0


def ok(label, condition):
    global checks
    checks += 1
    print(json.dumps({"check": checks, "label": label, "pass": bool(condition)}), flush=True)
    assert condition, label


def rejected(label, data, expected=None):
    global rejections
    (OUT / (label + ".json")).write_text(json.dumps(data, indent=2))
    try:
        p.joint_sampling_from_dict(data)
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        rejections += 1
        print(
            json.dumps({"rejection": label, "exception": type(exc).__name__, "message": str(exc)}),
            flush=True,
        )
        ok(label, expected is None or expected in str(exc))
    else:
        ok(label, False)


c = p.JointSamplingContext(np.random.default_rng(128))
with c.transaction("graph") as s:
    s.forced_identity("target", reason="target-warp-disabled")
original = json.loads(json.dumps(asdict(c.freeze())))
(OUT / "original.json").write_text(json.dumps(original, indent=2))
for label, value in [
    ("string-path", "graph"),
    ("dict-path", {"graph": None}),
    ("null-path", None),
    ("integer-path", 7),
    ("boolean-path", True),
    ("empty-path", []),
    ("nested-path", [["graph"]]),
    ("numeric-component", [7]),
    ("null-component", [None]),
    ("untrimmed-component", [" graph"]),
    ("nul-component", ["g\0"]),
    ("unicode-overlength", ["é" * 65]),
]:
    data = copy.deepcopy(original)
    data["events"][0]["operation"] = value
    rejected(
        label,
        data,
        "array"
        if label in ("string-path", "dict-path", "null-path", "integer-path", "boolean-path")
        else None,
    )
for field in ("role", "reason", "kind"):
    for label, value in [
        ("bool", True),
        ("int", 7),
        ("null", None),
        ("list", ["x"]),
        ("dict", {"x": 1}),
    ]:
        data = copy.deepcopy(original)
        data["events"][0][field] = value
        rejected(field + "-" + label, data)
for label, value in [("null", None), ("string", "graph"), ("dict", {"graph": None}), ("int", 7)]:
    data = copy.deepcopy(original)
    data["events"] = value
    rejected("events-" + label, data)
for label, value in [("null", None), ("string", "graph"), ("list", [["operation", ["graph"]]])]:
    data = copy.deepcopy(original)
    data["events"] = [value]
    rejected("event-" + label, data)
for field in ("max_draws", "max_values", "max_parameters"):
    for label, value in [
        ("zero", 0),
        ("negative", -1),
        ("overflow", 2**63),
        ("boolean", True),
        ("float", 1.0),
    ]:
        data = copy.deepcopy(original)
        data["limits"][field] = value
        rejected(field + "-" + label, data)
for value in (["graph"], ("graph",), ["é" * 64]):
    data = copy.deepcopy(original)
    data["events"][0]["operation"] = value
    decoded = p.joint_sampling_from_dict(data)
    ok("valid exact container/name boundary", decoded.events[0].operation == tuple(value))

# Independently reconstruct log, rounded, floored and boundary-mass scalar laws.
for seed in (1, 3, 28, 128):
    c = p.JointSamplingContext(np.random.default_rng(seed), mode="global")
    with c.transaction("laws") as s:
        for name, scale, integer, boundary, low, high in (
            ("log", "log", "none", False, 0.125, 64.0),
            ("rounded", "linear", "round", False, -3.0, 7.0),
            ("floored", "linear", "floor", False, -3, 7),
            ("boundary", "linear", "none", True, -3.0, 7.0),
        ):
            d = s.numerical(
                name,
                p.NumericalParameter(
                    name, low, high, scale=scale, integer=integer, boundary_mass=boundary
                ),
            )
            u = d.units[0]
            if boundary:
                u = max(0.0, min(1.0, 1.2 * u - 0.1))
            if scale == "log":
                expected = math.exp(math.log(low) * (1 - u) + math.log(high) * u)
                matches = math.isclose(d.values[0], expected, rel_tol=1e-14)
            else:
                expected = float(Fraction(low) * (1 - Fraction(u)) + Fraction(high) * Fraction(u))
                if integer == "round":
                    expected = round(expected)
                elif integer == "floor":
                    expected = max(low, min(high - 1, math.floor(expected)))
                matches = d.values[0] == expected
            ok("independent scalar law " + name, matches)
    (OUT / f"laws-{seed}.json").write_text(json.dumps(asdict(c.freeze()), indent=2))

# Child scope raw values/shapes exceed aggregate cap before poisoned units are decoded.
c = p.JointSamplingContext(np.random.default_rng(55))
with c.transaction("node") as s:
    s.categorical("c", p.CategoricalParameter("c", ("a", "b")))
data = json.loads(json.dumps(asdict(c.freeze())))
data["limits"]["max_values"] = 1
nested = data["categorical"]["preferences"][0]["hyperparameters"]["draws"][0]
nested["units"] = ["poison"]
rejected("raw-nested-preflight", data, "limit")

imports = {}
for name, module in sorted(sys.modules.items()):
    path = getattr(module, "__file__", None)
    if name.startswith("r0.") and path and Path(path).is_file():
        imports[name] = {
            "path": path,
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
(OUT / "imports.json").write_text(json.dumps(imports, indent=2))
(OUT / "totals.json").write_text(
    json.dumps(
        {
            "checks": checks,
            "expected_rejections": rejections,
            "source": "662fb89e098126990c2b6d5a86a4e6c189fc0321",
            "verdict": "PASS_COMPONENT",
        },
        indent=2,
    )
)
print(
    json.dumps({"checks": checks, "expected_rejections": rejections, "verdict": "PASS_COMPONENT"})
)
