import argparse
import hashlib
import json
import pathlib
import sys
import traceback
from dataclasses import asdict, replace

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
        if n.noise_position == "before_output":
            arg += eps
        if n.parents:
            parents = np.concatenate([values[j] for j in n.parents], axis=1)
            arg = arg + stage(
                parents, n.input_activation, seed, tid, i, "input", calibrate
            ) @ np.array(n.weights)
        val = stage(arg, n.output_activation, seed, tid, i, "output", calibrate)
        if n.noise_position == "after_output":
            val += eps
        values.append(val)
        raw[f"{i}-innovation"] = eps
        raw[f"{i}-values"] = val
    return values


controls = p.LinearRecipe(rows=16, context_rows=8, features=2, signal_features=2, active_features=2)
graph = p.CausalGraphPrior(family="forward", nodes=4, edge_probability=1.0)
base = p.sample_causal_recipe(graph, task=controls, seed=28, task_id=0, widths=(2, 1, 3, 1))
save("base.json", asdict(base))
try:
    for mode in ("local", "global", "meta"):
        for placement in ("input", "output", "both"):
            for prob in (0.0, 0.5, 1.0):
                name = f"{mode}-{placement}-{prob}"
                raw = {}
                pr = p.SourceActivationPrior(
                    mode=mode,
                    placement=placement,
                    noise_after_probability=prob,
                    calibration_rows=32,
                )
                rec = p.with_source_activations(base, prior=pr, seed=28, task_id=0)
                save(name + "-recipe.json", asdict(rec))
                vals = ancestral(rec, 28, 0, True)
                np.savez(out / (name + "-calibration.npz"), **raw)
                draws = rec.rescale_preferences.draws
                prefs = rec.rescale_preferences.preferences
                assert len(prefs) == 1
                assert prefs[0].request.name == "rr_scale"
                assert prefs[0].mode == mode
                assert all(
                    d.preference == 0
                    and d.request.low == 1
                    and d.request.high == 10
                    and d.request.scale == "log"
                    for d in draws
                )
                for d in draws:
                    close(d.values[0], 10 ** d.units[0], "log rr_scale")
                if mode == "global":
                    assert len({d.values for d in draws}) == 1
                longer = p.with_source_activations(
                    replace(base, task=replace(controls, rows=32, context_rows=17)),
                    prior=pr,
                    seed=28,
                    task_id=0,
                )
                assert longer.nodes == rec.nodes
                assert longer.sampling_record == rec.sampling_record
                larger = p.with_source_activations(
                    base,
                    prior=pr,
                    seed=28,
                    task_id=0,
                    numerical_limits=p.PreferenceLimits(
                        max_draws=20000, max_values=200000, max_parameters=2000000
                    ),
                    categorical_limits=p.PreferenceLimits(
                        max_draws=20000, max_values=200000, max_parameters=2000000
                    ),
                )
                assert larger.nodes == rec.nodes
                restored = p.recipe_from_dict(json.loads(json.dumps(asdict(rec))))
                assert restored == rec
                generated = p.generate_task(rec, seed=28, task_id=0)
                save(name + "-task.json", asdict(generated))
                trace = p.explain_generation(generated, [15, 0, 7])
                save(name + "-trace.json", trace)
                assert trace["target_max_error"] < 1e-10
                raw = {}
                val = ancestral(rec, 28, 0, False)
                np.savez(out / (name + "-generation.npz"), **raw)
                for n in trace["nodes"]:
                    if n["id"].startswith("causal-"):
                        close(n["values"], val[int(n["id"][7:])][[15, 0, 7]], n["id"])
                checks.append(name)
    save(
        "summary.json",
        {
            "status": "PASS",
            "cases": checks,
            "max_absolute_oracle_error": maxerr,
            "max_error_divided_by_1_plus_abs_expected": maxscaled,
            "worst_scaled": worst,
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
            "max_absolute_oracle_error": maxerr,
            "traceback": traceback.format_exc(),
        },
    )
    raise
