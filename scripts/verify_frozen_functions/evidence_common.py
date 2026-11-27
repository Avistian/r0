"""Independent scalar equations and retained case receipts; no author tests."""

import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import r0
from r0.modules import priors as p

ROOT = Path(__file__).resolve().parent
assert Path(r0.__file__).resolve().is_relative_to(ROOT / "source")
torch.set_num_threads(1)
EVENTS = []
STEM = Path(sys.argv[0]).stem


def record_event(event):
    EVENTS.append(event)
    with (ROOT / f"{STEM}.jsonl").open("a") as stream:
        stream.write(json.dumps(event, allow_nan=True) + "\n")
    print(f"{event['status']} {event['name']}", flush=True)


def matrix(rows):
    return tuple(tuple(float(v) for v in row) for row in rows)


def activation(width, stage=0, kind="tanh", parameters=()):
    return p.CalibratedActivation(
        kind=kind,
        parameters=parameters,
        input_mean=tuple(0.1 * (stage + j) for j in range(width)),
        input_inverse_std=tuple(0.6 + 0.1 * j for j in range(width)),
        shift=tuple(0.2 - 0.1 * j for j in range(width)),
        scale=1.25 + 0.25 * stage,
        output_mean=tuple(-0.2 + 0.05 * j for j in range(width)),
        output_inverse_std=tuple(0.8 + 0.1 * j for j in range(width)),
        shift_rows=(0,) * width,
        calibration_rows=1,
        calibration_id=f"authored-fixture-stage-{stage}",
        calibration_digest="a" * 64,
    )


def act_oracle(a, rows):
    args = dict(a.parameters)
    z = a.scale * ((rows - a.input_mean) * a.input_inverse_std + np.array(a.shift))
    z = args.get("input_scale", 1.0) * z + args.get("input_shift", 0.0)
    if a.kind == "identity":
        out = z
    elif a.kind == "tanh":
        out = np.tanh(z)
    elif a.kind == "relu":
        out = np.maximum(z, 0)
    elif a.kind == "signed_power":
        out = np.sign(z) * np.abs(z) ** args.get("exponent", 2)
    elif a.kind == "reciprocal_abs_power":
        out = (np.abs(z) + args.get("epsilon", 1e-3)) ** -args.get("exponent", 2)
    elif a.kind == "softmax":
        exp = np.exp(z - z.max(axis=1, keepdims=True))
        out = exp / exp.sum(axis=1, keepdims=True)
    elif a.kind == "rank":
        out = np.argsort(np.argsort(z, axis=1, stable=True), axis=1, stable=True)
    elif a.kind == "argsort":
        out = np.argsort(z, axis=1, stable=True)
    else:
        raise AssertionError(f"unimplemented independent activation oracle {a.kind}")
    out = args.get("output_scale", 1.0) * out + args.get("output_shift", 0.0)
    return (out - a.output_mean) * a.output_inverse_std


def oracle(f, rows):
    if type(f) is p.FrozenLinearFunction:
        return np.array(
            [
                [
                    math.fsum(x * w for x, w in zip(row, weights, strict=True))
                    for weights in f.matrix
                ]
                for row in rows
            ]
        )
    if type(f) is p.FrozenQuadraticFunction:
        out = []
        for row in rows:
            z = [row[j] for j in f.indices] + [1.0]
            out.append(
                [
                    math.fsum(z[i] * q[i][j] * z[j] for i in range(len(z)) for j in range(len(z)))
                    for q in f.matrices
                ]
            )
        return np.array(out)
    if type(f) is p.FrozenTreeFunction:
        out = []
        for row in rows:
            leaves = []
            for features, thresholds, table in zip(f.features, f.thresholds, f.leaves, strict=True):
                idx = sum(
                    (row[j] > t) * 2**depth
                    for depth, (j, t) in enumerate(zip(features, thresholds, strict=True))
                )
                leaves.append(table[idx])
            out.append(
                [math.fsum(leaf[j] / len(leaves) for leaf in leaves) for j in range(f.output_width)]
            )
        return np.array(out)
    if type(f) is p.FrozenRFFFunction:
        out = []
        for row in rows:
            waves = [
                math.cos(
                    math.fsum(row[i] * f.frequencies[i][k] for i in range(f.input_width)) + phase
                )
                for k, phase in enumerate(f.phases)
            ]
            out.append(
                [
                    math.fsum(waves[k] * f.coefficients[k][j] for k in range(len(waves)))
                    for j in range(f.output_width)
                ]
            )
        return np.array(out)
    if type(f) is p.FrozenDiscretizationFunction:
        out = []
        for row in rows:
            distances = [
                math.fsum(abs(float(x) - y) ** f.p for x, y in zip(row, center, strict=True))
                for center in f.centers
            ]
            # The positive root preserves ordering and is unnecessary here.
            center = f.centers[min(range(len(distances)), key=distances.__getitem__)]
            out.append(oracle(f.linear, [center])[0])
        return np.array(out)
    if type(f) is p.FrozenEMFunction:
        out = []
        for row in rows:
            logits = []
            for center, scale in zip(f.centers, f.scales, strict=True):
                dist = math.fsum(
                    abs(float(x) - y) ** f.p for x, y in zip(row, center, strict=True)
                ) ** (1 / f.p)
                logits.append(-math.log(2 * math.pi) / 2 - math.log(scale) - (dist / scale) ** f.q)
            weights = [math.exp(logit - max(logits)) for logit in logits]
            out.append(oracle(f.linear, [[w / math.fsum(weights) for w in weights]])[0])
        return np.array(out)
    if type(f) is p.FrozenProductFunction:
        return oracle(f.left, rows) * oracle(f.right, rows)
    if type(f) is p.FrozenMLPFunction:
        result, index = np.array(rows), 0
        if f.start_activation:
            result = act_oracle(f.activations[index], result)
            index += 1
        for k, linear in enumerate(f.linears):
            result = oracle(linear, result)
            if k < len(f.linears) - 1 or f.end_activation:
                result = act_oracle(f.activations[index], result)
                index += 1
        return result
    raise AssertionError(type(f))


def check(name, f, rows, expected=None, *, rtol=2e-12, atol=2e-12):
    inputs = torch.tensor(rows, dtype=torch.float64)
    original = inputs.clone()
    before = asdict(f)
    rng = torch.random.get_rng_state().clone()
    expected = oracle(f, np.array(rows)) if expected is None else np.array(expected)
    event = {
        "name": name,
        "record": before,
        "inputs": inputs.tolist(),
        "expected": expected.tolist(),
        "rtol": rtol,
        "atol": atol,
    }
    try:
        result = p.evaluate_frozen_function(f, inputs)
        event["observed"] = result.tolist()
        np.testing.assert_allclose(result.numpy(), expected, rtol=rtol, atol=atol)
        restored = p.frozen_function_from_dict(json.loads(json.dumps(before)))
        assert restored == f
        chunks = torch.cat([p.evaluate_frozen_function(restored, row[None]) for row in inputs])
        np.testing.assert_allclose(chunks.numpy(), expected, rtol=rtol, atol=atol)
        result.fill_(12345)
        repeated = p.evaluate_frozen_function(f, inputs)
        np.testing.assert_allclose(repeated.numpy(), expected, rtol=rtol, atol=atol)
        assert inputs.equal(original)
        assert before == asdict(f)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert repeated.dtype == torch.float64
        assert repeated.device.type == "cpu"
    except Exception as exc:
        event.update(status="REJECT", error=repr(exc))
        record_event(event)
        raise
    event["status"] = "PASS"
    record_event(event)


def reject(name, callback, *, accepted=(ValueError, TypeError, KeyError), payload=None):
    try:
        callback()
    except accepted as exc:
        record_event({"name": name, "status": "PASS", "rejection": repr(exc), "payload": payload})
    else:
        record_event(
            {"name": name, "status": "REJECT", "error": "invalid case accepted", "payload": payload}
        )
        raise AssertionError(name)


def finish():
    imports = {}
    for name, module in sorted(sys.modules.items()):
        if name == "r0" or name.startswith("r0."):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(ROOT / "source")
            imports[name] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    receipt = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cases": len(EVENTS),
        "imports": imports,
        "programs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ROOT.glob("*.py")
        },
    }
    (ROOT / f"{STEM}-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS {len(EVENTS)} cases")
