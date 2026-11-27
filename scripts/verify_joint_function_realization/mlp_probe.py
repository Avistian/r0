"""Independent standard-wrapper arithmetic on every MLP layer/flag configuration."""

import copy
import json
from dataclasses import asdict

import numpy as np
import torch
from common import OUTPUT, build, pin, plan_for, priors
from oracles import close, gaussian_matrix
from probe import reject


def primitive(kind, x, parameters):
    p = dict(parameters)
    if kind in ("relu_power", "signed_power"):
        base = np.maximum(x, 0) if kind == "relu_power" else np.abs(x)
        return base ** p["exponent"] * (np.sign(x) if kind == "signed_power" else 1)
    if kind == "softmax":
        z = np.exp(x - x.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)
    if kind == "tied_max":
        return (x == x.max(axis=1, keepdims=True)).astype(float)
    if kind in ("rank", "argsort"):
        order = np.argsort(x, axis=1, kind="stable")
        return np.argsort(order, axis=1, kind="stable") if kind == "rank" else order
    functions = {
        "identity": lambda: x,
        "tanh": lambda: np.tanh(x),
        "sigmoid": lambda: 1 / (1 + np.exp(-x)),
        "relu": lambda: np.maximum(x, 0),
        "leaky_relu": lambda: np.where(x >= 0, x, 0.01 * x),
        "elu": lambda: np.where(x >= 0, x, np.expm1(x)),
        "selu": lambda: 1.0507009873554805 * np.where(x >= 0, x, 1.6732632423543772 * np.expm1(x)),
        "silu": lambda: x / (1 + np.exp(-x)),
        "softplus": lambda: np.logaddexp(0, x),
        "relu6": lambda: np.clip(x, 0, 6),
        "hardtanh": lambda: np.clip(x, -1, 1),
        "sign_positive": lambda: np.where(x >= 0, 1.0, -1.0),
        "heaviside_positive": lambda: (x >= 0).astype(float),
        "gaussian_bump": lambda: np.exp(-x * x),
        "exp": lambda: np.exp(x),
        "interval_abs_le_1": lambda: (np.abs(x) <= 1).astype(float),
        "sin": lambda: np.sin(x),
        "square": lambda: x * x,
        "abs": lambda: np.abs(x),
        "log_clip_abs": lambda: np.log(np.maximum(np.abs(x), p.get("epsilon", 1e-6))),
        "round": lambda: np.round(x),
        "modulo": lambda: np.remainder(x, p.get("period", 1.0)),
        "log_sigmoid": lambda: -np.logaddexp(0, -x),
    }
    return functions[kind]()


def run_case(seed, sample):
    plan, decisions = plan_for("mlp", seed, rows=len(sample))
    result = build(plan, decisions, sample)
    raw, fit = np.random.default_rng(303), np.random.default_rng(404)
    value = sample.copy()
    linear_index = activation_index = 0
    for stage in plan.ordered_children:
        if hasattr(stage, "rows"):
            matrix = gaussian_matrix(stage, raw, result.matrices[linear_index])[0]
            value = value @ matrix.T
            linear_index += 1
        else:
            activation = result.function.activations[activation_index]
            stored = np.array(result.activation_inputs[activation_index].sample)
            close(value, stored)
            # Check fitted moments independently; use the retained preceding output
            # for subsequent boundary-sensitive scalar arithmetic.
            close(activation.input_mean, stored.mean(axis=0))
            close(activation.input_inverse_std, 1 / (stored.std(axis=0) + 1e-4))
            indices = fit.integers(len(sample), size=stored.shape[1])
            np.testing.assert_array_equal(activation.shift_rows, indices)
            centered = (stored - np.array(activation.input_mean)) * activation.input_inverse_std
            shift = -centered[indices, np.arange(stored.shape[1])]
            close(activation.shift, shift)
            activated = primitive(
                stage.activation.kind, stage.scale * (centered + shift), stage.activation.parameters
            )
            close(activation.output_mean, activated.mean(axis=0))
            close(activation.output_inverse_std, 1 / (activated.std(axis=0) + 1e-4))
            value = (activated - np.array(activation.output_mean)) * activation.output_inverse_std
            activation_index += 1
    close(result.outputs, value)
    close(priors.evaluate_frozen_function(result.function, torch.tensor(sample)).numpy(), value)
    payload = json.loads(json.dumps(asdict(result)))
    assert priors.joint_function_realization_from_dict(payload, decisions=decisions) == result
    if result.activation_inputs:
        for target in ("activation_inputs", "outputs"):
            altered = copy.deepcopy(payload)
            if target == "activation_inputs":
                altered[target][0]["identity"] = "stale-ancestor"
            else:
                altered[target][0][0] += 0.125
            reject(altered, decisions, ("mlp", target))
    return plan, result


def main():
    pin()
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]])
    seen, kinds = set(), set()
    for seed in range(400):
        plan, _ = plan_for("mlp", seed)
        p = dict(plan.parameters)
        shape = (p["layers"], p["start_activation"], p["end_activation"])
        if shape in seen:
            continue
        plan, result = run_case(seed, sample)
        kinds.update(stage.activation.kind for stage in plan.activations)
        seen.add(shape)
        changed = sample * np.array([3.0, 0.4]) + np.array([2.0, -7.0])
        _, second = run_case(seed, changed)
        if plan.activations:
            assert result.activation_inputs[0].digest != second.activation_inputs[0].digest
            first_mean = result.function.activations[0].input_mean
            assert first_mean != second.function.activations[0].input_mean
        print(f"PASS MLP seed={seed} shape={shape}", flush=True)
        if len(seen) == 12:
            break
    assert len(seen) == 12
    (OUTPUT / "mlp-coverage.json").write_text(
        json.dumps(
            {"shapes": sorted(seen), "activation_kinds": sorted(kinds)}, separators=(",", ":")
        )
    )
    print("PASS MLP all 12 configurations and changed fitted moments", flush=True)


if __name__ == "__main__":
    main()
