import argparse
import hashlib
import json
import pathlib
import sys
import traceback
from dataclasses import asdict, replace
from itertools import product

import numpy as np

from r0.modules import priors as p

ap = argparse.ArgumentParser()
ap.add_argument("outputdir")
args = ap.parse_args()
out = pathlib.Path(args.outputdir)
out.mkdir(parents=True, exist_ok=True)


def save(name, obj):
    (out / name).write_text(
        json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.tolist())
    )


def sha(b):
    return hashlib.sha256(b).hexdigest()


save(
    "execution.json",
    {
        "script_sha256": sha(pathlib.Path(__file__).read_bytes()),
        "python": sys.executable,
        "imports": {
            str(pathlib.Path(m.__file__).resolve()): sha(pathlib.Path(m.__file__).read_bytes())
            for n, m in list(sys.modules.items())
            if n.startswith("r0.") and getattr(m, "__file__", None)
        },
    },
)


def rng(seed, task, *keys):
    raw = json.dumps(
        ("causal-v1", seed, task, keys), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return np.random.default_rng(int(sha(raw), 16))


def primitive(x, a):
    k = a.kind
    v = dict(a.parameters)
    e = v.get("exponent", 2.0)
    if k == "identity":
        return x
    if k == "relu_power":
        return np.maximum(x, 0) ** e
    if k == "signed_power":
        return np.sign(x) * np.abs(x) ** e
    if k == "reciprocal_abs_power":
        return (np.abs(x) + v.get("epsilon", 1e-3)) ** (-e)
    if k == "integer_power":
        return x ** int(e)
    if k == "tanh":
        return np.tanh(x)
    if k == "sigmoid":
        return np.exp(-np.logaddexp(0, -x))
    if k == "leaky_relu":
        return np.where(x >= 0, x, 0.01 * x)
    if k == "elu":
        return np.where(x >= 0, x, np.expm1(x))
    if k == "selu":
        return 1.0507009873554805 * np.where(x >= 0, x, 1.6732632423543772 * np.expm1(x))
    if k == "silu":
        return x * np.exp(-np.logaddexp(0, -x))
    if k == "relu":
        return np.maximum(x, 0)
    if k == "softplus":
        return np.logaddexp(0, x)
    if k == "relu6":
        return np.clip(x, 0, 6)
    if k == "hardtanh":
        return np.clip(x, -1, 1)
    if k == "sign_positive":
        return np.where(x >= 0, 1.0, -1.0)
    if k == "heaviside_positive":
        return (x >= 0).astype(float)
    if k == "gaussian_bump":
        return np.exp(-x * x)
    if k == "exp":
        return np.exp(x)
    if k == "interval_abs_le_1":
        return (np.abs(x) <= 1).astype(float)
    if k == "sin":
        return np.sin(x)
    if k == "square":
        return x * x
    if k == "abs":
        return np.abs(x)
    if k == "log_clip_abs":
        return np.log(np.maximum(np.abs(x), 1e-6))
    if k == "rank":
        return np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1, kind="stable").astype(float)
    if k == "round":
        return np.round(x)
    if k == "modulo":
        return np.remainder(x, 1.0)
    if k == "softmax":
        z = np.exp(x - x.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
    if k == "tied_max":
        return (x == x.max(axis=1, keepdims=True)).astype(float)
    if k == "argsort":
        return np.argsort(x, axis=1, kind="stable").astype(float)
    if k == "log_sigmoid":
        return -np.logaddexp(0, -x)
    raise AssertionError(k)


checks = []
maxerr = 0.0
maxscaled = 0.0
worst = {}
raw = {}


def close(actual, expected, label):
    global maxerr, maxscaled, worst
    a, b = np.asarray(actual), np.asarray(expected)
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    maxerr = max(maxerr, err)
    scaled = float(np.max(np.abs(a - b) / (1 + np.abs(b)))) if a.size else 0.0
    if scaled > maxscaled:
        maxscaled = scaled
        worst = {
            "label": label,
            "absolute_error": err,
            "scale": float(np.max(np.abs(b))),
            "case": locals().get("name"),
        }
    np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-9, err_msg=label)


def stage(sample, a, seed, tid, i, s, calibrate):
    if not isinstance(a, p.CalibratedActivation):
        return primitive(sample, a)
    if calibrate:
        mean = sample.mean(axis=0)
        inv = 1 / (sample.std(axis=0) + 1e-4)
        idx = rng(seed, tid, "source-activation-calibration-v1", "shift", i, s).integers(
            len(sample), size=sample.shape[1]
        )
        shift = -((sample - mean) * inv)[idx, np.arange(sample.shape[1])]
        pre = a.scale * ((sample - mean) * inv + shift)
        y = primitive(pre, a)
        om = y.mean(axis=0)
        oi = 1 / (y.std(axis=0) + 1e-4)
        for key, expected in [
            ("input_mean", mean),
            ("input_inverse_std", inv),
            ("shift", shift),
            ("output_mean", om),
            ("output_inverse_std", oi),
        ]:
            close(getattr(a, key), expected, f"{i}/{s}/{key}")
        assert tuple(idx) == a.shift_rows
        assert a.calibration_id == f"source-activation-calibration-v1/{seed}/{tid}/{i}/{s}"
        raw[f"{i}-{s}-sample"] = sample
        # Independent calibration must propagate its own constants so its sampled
        # shift rows cancel exactly. Mixing NumPy samples with Torch fitted
        # constants is unstable for fractional powers at zero.
        return (y - om) * oi
    # Evaluate with retained constants independently, without implementation activation calls.
    return (
        primitive(a.scale * ((sample - a.input_mean) * a.input_inverse_std + a.shift), a)
        - a.output_mean
    ) * a.output_inverse_std


def ancestral(rec, seed, tid, calibrate):
    values = []
    rows = rec.source_prior.calibration_rows if calibrate else rec.task.rows
    keys = ("source-activation-calibration-v1",) if calibrate else ()
    for i, n in enumerate(rec.nodes):
        eps = (
            np.column_stack(
                [
                    rng(seed, tid, *keys, "innovation", i, j).normal(size=rows)
                    for j in range(len(n.bias))
                ]
            )
            * n.noise_scale
        )
        arg = np.zeros_like(eps) + n.bias
        if n.noise_position == "before_output" and not isinstance(n, p.MatrixCausalNode):
            arg += eps
        if n.parents:
            parents = np.concatenate([values[j] for j in n.parents], axis=1)
            arg = arg + stage(
                parents, n.input_activation, seed, tid, i, "input", calibrate
            ) @ np.array(n.weights)
        if isinstance(n, p.MatrixCausalNode) and n.noise_position == "before_output":
            arg = arg + eps
        val = stage(arg, n.output_activation, seed, tid, i, "output", calibrate)
        if n.noise_position == "after_output":
            val += eps
        values.append(val)
        raw[f"{i}-innovation"] = eps
        raw[f"{i}-values"] = val
    return values


# Independent matrix equations, using retained draws and NumPy only.
def vector(v):
    if v.version == "uniform-weight-vector-v1":
        return np.array(v.draws)
    logs = v.log_noise_scale * np.array(v.draws) - v.decay * np.log(np.arange(1, len(v.draws) + 1))
    if v.permutation:
        logs = logs[list(v.permutation)]
    w = np.exp(logs - logs.max())
    return w / np.sqrt(np.mean(w * w))


def matrix(w):
    kind = w.version
    if kind == "gaussian-weights-v1":
        return np.array(w.draws)
    if kind == "normalized-weights-v1":
        a = matrix(w.source) + w.jitter_scale * np.array(w.jitter)
        return a / (w.epsilon + np.linalg.norm(a, axis=1, keepdims=True))
    if kind == "transformed-weights-v1":
        a = matrix(w.child)
        return primitive(a.reshape(1, -1), w.activation).reshape(
            a.shape
        ) + w.jitter_scale * np.array(w.jitter)
    if kind == "importance-weights-v1":
        a = np.array(w.gaussian) * np.array([vector(v) for v in w.importance])
        return a / (w.epsilon + np.linalg.norm(a, axis=1, keepdims=True))
    if kind == "importance-factorized-weights-v1":
        return (np.array(w.left) * vector(w.diagonal)) @ np.array(w.right)
    if kind == "laplace-kernel-weights-v1":
        c = w.point_construction
        pts = np.array(
            [
                np.array(c.centers[k])
                + np.exp(c.scale_mean + c.scale_coefficient * c.scale_draws[k])
                * (np.array(c.factors[k]) @ (vector(c.factor_weights[k]) * z))
                for k, z in zip(c.assignments, c.innovations, strict=True)
            ]
        )
        close(np.concatenate([w.left_points, w.right_points]), pts, "point cloud construction")
        return np.array(w.signs) * np.exp(
            -w.distance_scale
            * np.linalg.norm(
                np.array(w.left_points)[:, None, :] - np.array(w.right_points)[None, :, :], axis=2
            )
        )
    raise AssertionError(kind)


controls = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
base = p.sample_causal_recipe(
    p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0),
    task=controls,
    seed=28,
    task_id=0,
    widths=(2, 1, 3, 1),
)
save("base.json", asdict(base))
# Fixed grid includes all five families, both vector laws, all stage placements,
# all modes and policies, and before/mixed/after noise. No search or retries.

cases = list(
    product(
        ("gaussian", "importance", "factorized", "kernel", "transformed"),
        ("input", "output", "both"),
        (0.0, 0.5, 1.0),
    )
)
try:
    for idx, (family, placement, prob) in enumerate(cases):
        mode = ("local", "global", "meta")[idx % 3]
        policy = ("shared", "refresh")[idx % 2]
        name = f"{idx:02d}-{family}-{placement}-{prob}-{mode}-{policy}"
        raw = {}
        pr = p.SourceActivationPrior(
            mode=mode,
            policy=policy,
            placement=placement,
            noise_after_probability=prob,
            calibration_rows=32,
        )
        mp = p.CausalMatrixPrior(
            mode=mode,
            categorical_policy=policy,
            families=(family,),
            simple_families=(("gaussian", "importance", "factorized", "kernel")[idx % 4],),
            vector_family=("polynomial", "uniform")[idx % 2],
            point_laws=(("gaussian-mixture", "random-covariance")[idx % 2],),
            covariance_bases=(("gaussian", "uniform")[idx % 2],),
        )
        save(name + "-priors.json", {"source": asdict(pr), "matrix": asdict(mp)})
        rec = p.with_source_matrices(base, prior=pr, matrix_prior=mp, seed=28, task_id=0)
        save(name + "-recipe.json", asdict(rec))
        for n in rec.nodes:
            if isinstance(n, p.MatrixCausalNode):
                close(n.weights, matrix(n.weight_construction), "matrix realization")
        ancestral(rec, 28, 0, True)
        np.savez(out / (name + "-calibration.npz"), **raw)
        longer = p.with_source_matrices(
            replace(base, task=replace(controls, rows=32, context_rows=17)),
            prior=pr,
            matrix_prior=mp,
            seed=28,
            task_id=0,
        )
        assert longer.nodes == rec.nodes
        assert longer.sampling_record == rec.sampling_record
        assert p.recipe_from_dict(json.loads(json.dumps(asdict(rec)))) == rec
        generated = p.generate_task(rec, seed=28, task_id=0)
        trace = p.explain_generation(generated, [15, 0, 7])
        save(name + "-task.json", asdict(generated))
        save(name + "-trace.json", trace)
        raw = {}
        values = ancestral(rec, 28, 0, False)
        np.savez(out / (name + "-generation.npz"), **raw)
        for n in trace["nodes"]:
            if n["id"].startswith("causal-"):
                close(n["values"], values[int(n["id"][7:])][[15, 0, 7]], n["id"])
        checks.append(name)
        save("progress.json", {"completed": checks})
    save(
        "summary.json",
        {
            "status": "PASS",
            "cases": checks,
            "max_absolute_oracle_error": maxerr,
            "max_scaled_error": maxscaled,
            "worst": worst,
            "tolerance": {"rtol": 2e-9, "atol": 2e-9},
        },
    )
except Exception:
    if raw:
        np.savez(out / "stopped-raw.npz", **raw)
    save(
        "summary.json",
        {
            "status": "STOPPED",
            "completed": checks,
            "case": locals().get("name"),
            "traceback": traceback.format_exc(),
        },
    )
    raise
