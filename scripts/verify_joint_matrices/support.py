"""Independent evidence helpers; no implementation arithmetic is imported."""

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"assertions": 0, "rejections": 0}


def check(value, label):
    COUNTS["assertions"] += 1
    assert value, label


def close(actual, expected, label):
    COUNTS["assertions"] += 1
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12, err_msg=label)


def reject(function, label):
    try:
        function()
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        COUNTS["rejections"] += 1
        print(
            json.dumps({"rejected": label, "type": type(exc).__name__, "error": str(exc)}),
            flush=True,
        )
    else:
        raise AssertionError("ACCEPTED MALFORMED: " + label)


def raw(value):
    return json.loads(json.dumps(asdict(value)))


def finish(name):
    imports = {}
    for key, module in sorted(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if key.startswith("r0.") and filename and Path(filename).is_file():
            imports[key] = {
                "path": filename,
                "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            }
    result = {
        "counts": COUNTS,
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "imports": imports,
    }
    (ROOT / "results" / (name + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps({"PASS": name, **COUNTS}), flush=True)


def normalize(a):
    return a / (1e-6 + np.linalg.norm(a, axis=-1, keepdims=True))


def vectors(plan, rng):
    b, w = plan.batch, plan.width
    if plan.family == "uniform":
        return rng.random((b, w))
    innovations = rng.normal(size=(b, w))
    rows = []
    for z, decay, scale in zip(innovations, plan.decays, plan.scales, strict=True):
        logits = (-decay * np.log(np.arange(1, w + 1)) + scale * z)[rng.permutation(w)]
        masses = np.exp(logits - np.max(logits))
        rows.append(np.sqrt(w) * masses / np.linalg.norm(masses))
    return np.array(rows)


def points(plan, rng):
    c, d, q = plan.components, plan.dimension, plan.count
    mass = vectors(plan.mixture_weights, rng)[0]
    assignments = rng.choice(c, size=q, p=mass / mass.sum())
    centers = rng.normal(size=(c, d))
    factors = rng.normal(size=(c, d, d))
    weights = vectors(plan.factor_weights, rng)
    mean, coefficient = rng.normal(), rng.normal()
    scale_draws = rng.normal(size=c)
    innovations = rng.normal(size=(q, d))
    scaled = factors * weights[:, None, :] * np.exp(mean + coefficient * scale_draws)[:, None, None]
    values = centers[assignments] + np.einsum("qij,qj->qi", scaled[assignments], innovations)
    return values, {"mean": mean, "coefficient": coefficient, "assignments": assignments.tolist()}


def activation(a, descriptor):
    kind = descriptor.kind
    pars = dict(descriptor.parameters)
    exponent = pars.get("exponent", 2)
    if kind in ("integer_power", "power"):
        return a**exponent
    if kind == "relu_power":
        return np.maximum(a, 0) ** exponent
    if kind == "signed_power":
        return np.sign(a) * abs(a) ** exponent
    if kind == "reciprocal_abs_power":
        return (abs(a) + pars.get("epsilon", 1e-3)) ** -exponent
    if kind == "log_clip_abs":
        return np.log(np.maximum(abs(a), pars.get("epsilon", 1e-6)))
    if kind == "softmax":
        e = np.exp(a - a.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    if kind in ("rank", "argsort"):
        order = np.argsort(a, axis=1, stable=True)
        return np.argsort(order, axis=1, stable=True) if kind == "rank" else order
    if kind == "argmax":
        return np.eye(a.shape[1])[np.argmax(a, axis=1)]
    functions = {
        "identity": lambda: a,
        "relu": lambda: np.maximum(a, 0),
        "tanh": lambda: np.tanh(a),
        "sigmoid": lambda: 1 / (1 + np.exp(-a)),
        "sin": lambda: np.sin(a),
        "abs": lambda: abs(a),
        "square": lambda: a * a,
        "exp": lambda: np.exp(a),
        "sign": lambda: np.sign(a),
        "sign_positive": lambda: np.where(a >= 0, 1, -1),
        "round": lambda: np.round(a),
        "sqrt_abs": lambda: np.sqrt(abs(a)),
        "gaussian_bump": lambda: np.exp(-a * a),
        "interval_abs_le_1": lambda: abs(a) <= 1,
        "interval_0_1": lambda: (a >= 0) & (a <= 1),
        "heaviside": lambda: a > 0,
        "heaviside_positive": lambda: a >= 0,
        "step": lambda: a > 0,
        "softplus": lambda: np.logaddexp(0, a),
        "smooth_relu": lambda: np.logaddexp(0, a),
        "silu": lambda: a / (1 + np.exp(-a)),
        "log_sigmoid": lambda: -np.logaddexp(0, -a),
        "tied_max": lambda: a == a.max(axis=1, keepdims=True),
        "relu6": lambda: np.clip(a, 0, 6),
        "hardtanh": lambda: np.clip(a, -1, 1),
        "selu": lambda: 1.0507009873554805 * np.where(a > 0, a, 1.6732632423543772 * np.expm1(a)),
        "elu": lambda: np.where(a > 0, a, pars.get("alpha", 1) * np.expm1(a)),
        "leaky_relu": lambda: np.where(a > 0, a, pars.get("negative_slope", 0.01) * a),
    }
    return functions[kind]()


def matrix(plan, rng):
    b, n, m = plan.batch, plan.rows, plan.columns
    if plan.family == "gaussian":
        a = rng.normal(size=(b, n, m))
    elif plan.family == "importance":
        w = vectors(plan.weights, rng).reshape(b, n, m)
        a = normalize(w * rng.normal(size=(b, n, m)))
    elif plan.family == "factorized":
        k = min(n, m)
        u = rng.normal(size=(b, n, k))
        w = vectors(plan.weights, rng)
        v = rng.normal(size=(b, k, m))
        a = (u * w[:, None, :]) @ v
    elif plan.family == "kernel":
        cloud, _ = points(plan.points, rng)
        cloud = cloud.reshape(b, n + m, 3)
        distances = np.linalg.norm(cloud[:, :n, None, :] - cloud[:, None, n:, :], axis=-1)
        a = np.exp(-plan.distance * distances) * np.sign(rng.normal(size=(b, n, m)))
    else:
        child = matrix(plan.child, rng)
        a = activation(child.reshape(b, n * m), plan.activation.activation).reshape(b, n, m)
        a = a + 1e-3 * rng.normal(size=(b, n, m))
    return normalize(a + 1e-6 * rng.normal(size=(b, n, m)))


def vector_count(plan):
    return plan.batch * (plan.width if plan.family == "uniform" else 2 * plan.width + 2)


def point_count(plan):
    c, d, q = plan.components, plan.dimension, plan.count
    return (
        c * d
        + c * d * d
        + vector_count(plan.factor_weights)
        + vector_count(plan.mixture_weights)
        + q
        + q * d
        + 2
        + c
    )


def matrix_count(plan):
    b, n, m = plan.batch, plan.rows, plan.columns
    if plan.family == "gaussian":
        return b * (2 * n * m + 2)
    if plan.family == "importance":
        return b * (2 * n * m + 3) + vector_count(plan.weights)
    if plan.family == "factorized":
        k = min(n, m)
        return b * (n * k + k * m + n * m + 2) + vector_count(plan.weights)
    if plan.family == "kernel":
        return point_count(plan.points) + b * (3 * (n + m) + 2 * n * m + 3)
    return matrix_count(plan.child) + b * (
        len(plan.activation.activation.parameters) + 2 * n * m + 3
    )
