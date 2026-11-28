"""Bounded independent public-API verification; never imports private r0 helpers."""

import argparse
import copy
import hashlib
import importlib
import json
import math
import platform
import random
import subprocess
import sys
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

PIN = "593c924f5e2ea8da31d4c34d9da430aef4b38489"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--repo", type=Path, default=Path.cwd())
parser.add_argument("output_dir", type=Path, help="New directory for retained results")
args = parser.parse_args()
repo = args.repo.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=False)
sys.dont_write_bytecode = True


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=repo)


head = git("rev-parse", "HEAD").decode().strip()
if head != PIN:
    raise RuntimeError(f"Expected pinned HEAD {PIN}, got {head}")
source_manifest = {}
for entry in git("ls-tree", "-r", "-z", PIN, "--", "src").split(b"\0"):
    if not entry:
        continue
    metadata, raw_path = entry.split(b"\t", 1)
    mode, kind, expected_blob = metadata.decode().split()
    path = raw_path.decode()
    if kind != "blob" or mode not in {"100644", "100755"}:
        raise RuntimeError(f"Unsupported source entry: {path}")
    content = (repo / path).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if blob != expected_blob:
        raise RuntimeError(f"Source differs from pin: {path}")
    source_manifest[path] = {"git_blob": blob, "sha256": hashlib.sha256(content).hexdigest()}
actual_python = {p.relative_to(repo).as_posix() for p in (repo / "src").rglob("*.py")}
if actual_python - set(source_manifest):
    raise RuntimeError("Untracked Python files in source tree")
sys.path.insert(0, str(repo / "src"))
public = importlib.import_module("r0.modules.priors")
imported_r0 = Path(importlib.import_module("r0").__file__).resolve()
imported_public = Path(public.__file__).resolve()
if imported_public != repo / "src/r0/modules/priors/__init__.py":
    raise RuntimeError(f"Wrong imported public API: {imported_public}")
P, D, S = public.NumericalParameter, public.NumericalDraw, public.ParameterSampler
Limits, decode = public.PreferenceLimits, public.parameter_preferences_from_dict
(OUT / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
counts = {}
probe_index = 0


def retain(name, passed, evidence):
    global probe_index
    probe_index += 1
    row = {"index": probe_index, "check": name, "passed": bool(passed), "evidence": evidence}
    with (OUT / "probes.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def check(name, condition, evidence=None):
    retain(name, condition, evidence)
    if not condition:
        raise AssertionError(name)
    counts[name] = counts.get(name, 0) + 1


def reject(name, fn):
    try:
        fn()
    except (ValueError, TypeError, KeyError) as error:
        retain(name, True, {"exception": type(error).__name__, "message": str(error)})
        counts[name] = counts.get(name, 0) + 1
    else:
        retain(name, False, {"exception": None})
        raise AssertionError("accepted " + name)


def numeric_check(name, request, unit, expected):
    try:
        actual = D(request, 0, (unit,), (expected,)).values[0]
    except Exception as error:
        retain(
            name,
            False,
            {
                "request": asdict(request),
                "unit": unit,
                "expected": expected,
                "exception": repr(error),
            },
        )
        raise
    # Public reconstruction validates expected values against its numerical transform.
    evidence = {
        "request": asdict(request),
        "unit": unit,
        "unit_hex": unit.hex(),
        "expected": expected,
        "actual_validated_value": actual,
        "expected_hex": float(expected).hex(),
        "actual_hex": float(actual).hex(),
        "observation": "NumericalDraw accepted the independently computed value",
    }
    check(name, actual == expected, evidence)


def state(rng):
    return json.dumps(rng.bit_generator.state, sort_keys=True)


def oracle_linear(lo, hi, u):
    # Exact rational convex interpolation, with one correctly rounded float conversion.
    return float(
        Fraction.from_float(float(lo)) * (1 - Fraction.from_float(u))
        + Fraction.from_float(float(hi)) * Fraction.from_float(u)
    )


def oracle_log(lo, hi, u):
    # Independent ratio-and-power formulation at 180-digit precision.
    with localcontext() as ctx:
        ctx.prec = 180
        a, b, t = (
            Decimal.from_float(float(lo)),
            Decimal.from_float(float(hi)),
            Decimal.from_float(u),
        )
        return float(a * (b / a) ** t)


def mapped(req, u):
    t = min(1.0, max(0.0, 1.2 * u - 0.1)) if req.boundary_mass else u
    x = (
        oracle_linear(req.low, req.high, t)
        if req.scale == "linear"
        else oracle_log(req.low, req.high, t)
    )
    if req.integer == "round":
        return round(x)
    if req.integer == "floor":
        return min(req.high - 1, max(req.low, math.floor(x)))
    return x


rnd = random.Random(280593924)
units = [
    0.0,
    math.nextafter(0.0, 1.0),
    0.125,
    0.375,
    0.5,
    0.625,
    0.875,
    math.nextafter(1.0, 0.0),
    1.0,
]
linear_ranges = [
    (-sys.float_info.max, sys.float_info.max),
    (0.0, math.ulp(0.0)),
    (-1e308, 1e-300),
    (2**53, 2**53 + 4),
    (-7.0, 11.0),
]
for bounds in linear_ranges:
    for integer in ["none", "round"]:
        for boundary in [False, True]:
            req = P("oracle", *bounds, integer=integer, boundary_mass=boundary)
            for u in units + [rnd.random() for _ in range(12)]:
                expected = mapped(req, u)
                numeric_check("fraction_linear", req, u, expected)
for bounds in [
    (1e-300, 1e300),
    (math.ulp(0.0), sys.float_info.max),
    (0.125, 128.0),
    (1.0, math.nextafter(1.0, 2.0)),
]:
    for boundary in [False, True]:
        req = P("log", *bounds, scale="log", boundary_mass=boundary)
        for u in units + [rnd.random() for _ in range(12)]:
            expected = mapped(req, u)
            numeric_check("decimal_ratio_log", req, u, expected)
for bounds in [(-5, 7), (0, 5), (2**63, 2**63 + 8), (-(10**300), 10**300)]:
    for scale in ["linear"] + (["log"] if bounds[0] > 0 else []):
        req = P("floor", *bounds, integer="floor", scale=scale)
        for u in units:
            expected = mapped(req, u)
            numeric_check("floor", req, u, expected)
for mode in ["local", "meta", "global"]:
    rng = np.random.default_rng(28)
    sampler = S(rng, default_mode=mode)
    a = sampler.numerical(P("gain", -2.0, 3.0, shape=(2, 3)))
    b = sampler.numerical(P("gain", -2.0, 3.0, shape=(2, 3)))
    record = sampler.freeze()
    check("scope", a.preference == b.preference and len(record.preferences) == 1)
    check("reuse_or_refresh", (a.units == b.units) == (mode == "global"))
    check("roundtrip", decode(json.loads(json.dumps(asdict(record)))) == record)
    replica = S(np.random.default_rng(28), default_mode=mode)
    check("replay", replica.numerical(a.request) == a and replica.numerical(b.request) == b)
    if mode != "global":
        c = sampler.numerical(P("gain", 1.0, 5.0, shape=(3,)))
        check("changed_bounds_shape", c.preference == a.preference)
    local = sampler.numerical(P("gain", -2.0, 3.0, mode="local"))
    global_ = sampler.numerical(P("gain-global", -2.0, 3.0, mode="global"))
    check(
        "explicit_modes",
        sampler.freeze().preferences[local.preference].mode == "local"
        and sampler.freeze().preferences[global_.preference].mode == "global",
    )
for policy in ["strict", "quantile"]:
    rng = np.random.default_rng(82)
    sampler = S(rng, default_mode="global", global_policy=policy)
    req = P("cache", 1.0, 8.0, shape=(2,))
    a = sampler.numerical(req)
    before = state(rng)
    check(
        "global_no_rng",
        sampler.numerical(replace(req, mode="global")).units == a.units and state(rng) == before,
    )
    for patch in [
        dict(low=0.5),
        dict(high=9.0),
        dict(scale="log"),
        dict(integer="round"),
        dict(boundary_mass=True),
        dict(shape=(1, 2)),
    ]:
        changed = replace(req, **patch)
        frozen = sampler.freeze()
        if policy == "strict" or "shape" in patch:
            reject(
                "contract_reject",
                lambda sampler=sampler, changed=changed: sampler.numerical(changed),
            )
            check("rejection_atomic", state(rng) == before and sampler.freeze() == frozen)
        else:
            remap = sampler.numerical(changed)
            check(
                "quantile_oracle",
                remap.units == a.units
                and remap.values == tuple(mapped(changed, u) for u in a.units),
                {
                    "request": asdict(changed),
                    "units": a.units,
                    "expected": tuple(mapped(changed, u) for u in a.units),
                    "actual": remap.values,
                },
            )
    check("roundtrip", decode(json.loads(json.dumps(asdict(sampler.freeze())))) == sampler.freeze())
for limits in [Limits(max_draws=1), Limits(max_values=2), Limits(max_parameters=13)]:
    rng = np.random.default_rng(8)
    sampler = S(rng, default_mode="global", limits=limits)
    sampler.numerical(P("cost", 0.0, 1.0, shape=(2,)))
    check("leaf_cost", sampler.freeze().parameter_count == 13)
    before = state(rng)
    frozen = sampler.freeze()
    reject("limit_reject", lambda sampler=sampler: sampler.numerical(P("more", 0.0, 1.0)))
    check("rejection_atomic", state(rng) == before and sampler.freeze() == frozen)
for shape in [(0,), (-1,), (True,), (1,) * 9, (100001,), [2], (1.5,)]:
    reject("shape_reject", lambda shape=shape: P("bad", 0.0, 1.0, shape=shape))
for patch in [
    dict(low=float("nan")),
    dict(high=float("inf")),
    dict(low=2.0),
    dict(scale="log"),
    dict(integer="floor"),
    dict(boundary_mass=1),
]:
    reject(
        "domain_reject", lambda patch=patch: P(**dict(dict(name="bad", low=0.0, high=1.0), **patch))
    )
sampler = S(np.random.default_rng(63))
sampler.numerical(P("law", 0.0, 1.0, shape=(4096,)))
record = sampler.freeze()
pref = record.preferences[0]
values = np.array(record.draws[0].units)
var = pref.mean * (1 - pref.mean) / (1 + pref.concentration)
(OUT / "beta-draw.json").write_text(json.dumps(asdict(record), indent=2) + "\n")
check(
    "beta_mean",
    abs(float(values.mean()) - pref.mean) < 8 * math.sqrt(var / len(values)),
    {
        "mean": pref.mean,
        "concentration": pref.concentration,
        "expected": pref.mean,
        "actual": float(values.mean()),
        "tolerance": 8 * math.sqrt(var / len(values)),
        "samples": "beta-draw.json",
    },
)
check(
    "beta_second_moment",
    abs(float(np.mean(values**2)) - (var + pref.mean**2)) < 8 / math.sqrt(len(values)),
    {
        "expected": var + pref.mean**2,
        "actual": float(np.mean(values**2)),
        "tolerance": 8 / math.sqrt(len(values)),
        "samples": "beta-draw.json",
    },
)


# Independent recursive JSON numeric-leaf count, excluding configuration/Booleans.
def leaves(obj):
    if isinstance(obj, dict):
        return sum(leaves(v) for k, v in obj.items() if k != "limits")
    if isinstance(obj, (list, tuple)):
        return sum(map(leaves, obj))
    return int(type(obj) in (float, int))


check("leaf_cost", leaves(asdict(record)) == record.parameter_count)
base = json.loads(json.dumps(asdict(record)))
corruptions = [
    ("unit_support", lambda x: x["draws"][0]["units"].__setitem__(0, 1.1)),
    ("value", lambda x: x["draws"][0]["values"].__setitem__(0, -2)),
    ("reference", lambda x: x["draws"][0].__setitem__("preference", 2)),
    ("scope", lambda x: x["draws"][0]["request"].__setitem__("mode", "global")),
    ("first_request", lambda x: x["preferences"][0]["request"].__setitem__("high", 2.0)),
    ("beta", lambda x: x["preferences"][0].__setitem__("mean", 0)),
    ("cost", lambda x: x["limits"].__setitem__("max_parameters", 1)),
    ("shape", lambda x: x["draws"][0]["request"].__setitem__("shape", [1])),
    ("orphan", lambda x: x.__setitem__("draws", [])),
    ("version", lambda x: x.__setitem__("version", "unknown")),
]
for name, mutate in corruptions:
    data = copy.deepcopy(base)
    mutate(data)
    reject("json_" + name, lambda data=data: decode(data))
for relative, identity in source_manifest.items():
    if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != identity["sha256"]:
        raise RuntimeError(f"Source changed during verification: {relative}")
if git("rev-parse", "HEAD").decode().strip() != PIN:
    raise RuntimeError("HEAD changed during verification")
receipt = {
    "result": "PASS_BOUNDED",
    "counts": counts,
    "total_checks": sum(counts.values()),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "head": head,
    "repo": str(repo),
    "imported_r0": str(imported_r0),
    "imported_public_api": str(imported_public),
    "source_tree": git("rev-parse", PIN + ":src").decode().strip(),
    "verified_source_files": len(source_manifest),
    "source_manifest_sha256": hashlib.sha256(
        (OUT / "source-manifest.json").read_bytes()
    ).hexdigest(),
    "arithmetic_observation": "Public NumericalDraw validation of independent expected values",
    "sampler_observation": "Quantile and Beta results are actual ParameterSampler outputs",
    "argv": sys.argv,
    "output_dir": str(OUT),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "not_run": [
        "full tests",
        "categorical preferences",
        "generator integration",
        "paid/cloud/data",
    ],
    "scope": "public numerical API only; no overall issue acceptance",
}
(OUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))

with (OUT / "SHA256SUMS").open("w") as stream:
    for artifact in sorted(OUT.iterdir()):
        if artifact.is_file() and artifact.name != "SHA256SUMS":
            stream.write(
                hashlib.sha256(artifact.read_bytes()).hexdigest() + "  " + artifact.name + "\n"
            )
