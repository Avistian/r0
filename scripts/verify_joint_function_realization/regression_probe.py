"""Pinned-base regression plus strengthened retained-stage and scalar validation."""

import copy
import hashlib
import importlib.util
import json
import pickle
import sys
from dataclasses import asdict

import numpy as np
import torch
from common import INPUTS, OUTPUT, build, pin, plan_for, priors
from probe import leaves, reject, set_leaf

from r0.modules.priors.lib.calibrated_activations import calibrate_source_activation
from r0.modules.priors.lib.source_activation_law import FIXED_ACTIVATIONS


def old_module(name):
    filename = f"base_{name}.py"
    path = INPUTS / filename
    expected = json.loads((INPUTS / "base-pins.json").read_text())[filename]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(f"review_base_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def regression():
    old = old_module("calibrated_activations")
    sample = np.array([[-2.0, 4.0], [2.0, -1.0], [0.0, 1.0], [3.0, 2.0]])
    primitives = [priors.CausalActivation(kind) for kind in FIXED_ACTIVATIONS]
    primitives += [
        priors.CausalActivation(kind, (("exponent", 1.7),))
        for kind in ("relu_power", "signed_power")
    ]
    for activation in primitives:
        before, after = np.random.default_rng(35), np.random.default_rng(35)
        options = {"scale": 2.5, "calibration_id": "base-regression"}
        prior = old.calibrate_source_activation(activation, sample, rng=before, **options)
        current = calibrate_source_activation(activation, sample, rng=after, **options)
        assert asdict(prior) == asdict(current), activation.kind
        assert before.bit_generator.state == after.bit_generator.state
        np.testing.assert_array_equal(
            prior.apply(torch.tensor(sample)).numpy(), current.apply(torch.tensor(sample)).numpy()
        )
    old_eval = old_module("frozen_function_evaluation")
    plan, decisions = plan_for("em", 7)
    result = build(plan, decisions, sample)
    for query in (sample, np.zeros((1, 2)), sample[::-1].copy()):
        np.testing.assert_array_equal(
            old_eval.evaluate_frozen_function(result.function, torch.tensor(query)).numpy(),
            priors.evaluate_frozen_function(result.function, torch.tensor(query)).numpy(),
        )
    print("PASS exact baseline regression 29 standard activations and frozen EM", flush=True)


def mutations():
    plan, decisions = plan_for("mlp", 645)
    sample = np.array([[-2.0, 4.0], [2.0, -1.0], [0.0, 1.0], [3.0, 2.0]])
    result = build(plan, decisions, sample)
    payload = json.loads(json.dumps(asdict(result)))
    count = 0
    for path, value in leaves(payload):
        changed = copy.deepcopy(payload)
        set_leaf(changed, path, bool(value))
        reject(changed, decisions, ("MLP numeric bool", path))
        count += 1
    changed = copy.deepcopy(payload)
    stage = changed["activation_inputs"][0]
    stage["sample"][0][0] += 0.25
    array = np.array(stage["sample"], dtype="<f8")
    digest = hashlib.sha256(f"{array.shape[0]},{array.shape[1]}:".encode())
    digest.update(array.tobytes())
    stage["digest"] = digest.hexdigest()
    reject(changed, decisions, "stale MLP stage with recomputed valid digest")
    (OUTPUT / "stale-stage-valid-digest.json").write_text(
        json.dumps(changed, separators=(",", ":"))
    )
    print(f"PASS MLP {count} numeric-bool mutations and stale input with valid digest", flush=True)


def genuine_overflow():
    plan, decisions = plan_for("mlp", 1541, width=1, rows=10000)
    sample = np.zeros((10000, 1))
    # Sign chosen from the retained successful first matrix, independently of any
    # hidden mutable generator. The rare positive output overflows exp after fit.
    raw, fit = np.random.default_rng(101), np.random.default_rng(102)
    snapshot = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions))
    sample[0, 0] = -100
    try:
        build(plan, decisions, sample, raw, fit)
    except ValueError as exc:
        if "nonfinite" not in str(exc):
            raise AssertionError("unexpected overflow reason") from exc
        print(f"PASS genuine arithmetic rollback: {exc}", flush=True)
    else:
        raise AssertionError("expected finite-input activation overflow")
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions)) == snapshot


def main():
    pin()
    regression()
    mutations()
    genuine_overflow()


if __name__ == "__main__":
    main()
