"""Portable public-API selector, replay binding, policy and atomicity probes."""

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from r0.modules import priors

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
results = []
FIXED = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "selu",
    "silu",
    "relu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign_positive",
    "heaviside_positive",
    "gaussian_bump",
    "exp",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
    "log_clip_abs",
    "rank",
    "sigmoid",
    "round",
    "modulo",
    "softmax",
    "tied_max",
    "argsort",
    "log_sigmoid",
)


def record(name, detail):
    results.append({"name": name, "detail": detail})
    (args.output / "selectors.json").write_text(json.dumps(results, indent=2) + "\n")
    print(name, "PASS", detail, flush=True)


def reject(name, data):
    (args.output / f"invalid-{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    try:
        priors.source_activation_selections_from_dict(data)
    except (ValueError, TypeError, KeyError) as error:
        record(name, str(error))
    else:
        raise AssertionError(f"malformed source binding accepted: {name}")


slots_seen = set()
integers_seen = set()
records = []
for seed in range(12):
    rng = np.random.default_rng(seed)
    oracle = np.random.default_rng(seed)
    sampler = priors.SourceActivationSampler(rng, mode="local")
    for i in range(40):
        selector = "plain" if i % 2 == 0 else "standard"
        repeat = 8 if selector == "plain" else 4
        powers = ["relu_power", "signed_power"]
        if selector == "plain":
            powers += ["reciprocal_abs_power", "integer_power"]
        slots = [f"fixed-{j}" for j in range(repeat)] + powers
        slot = slots[int(oracle.random() * len(slots))]
        slots_seen.add((selector, slot))
        if slot.startswith("fixed-"):
            expected_kind = FIXED[int(oracle.random() * len(FIXED))]
            exponent = None
        else:
            expected_kind = slot
            unit = float(oracle.random())
            exponent = math.floor(2 + 4 * unit) if slot == "integer_power" else 10 ** (2 * unit - 1)
        activation = sampler.sample(selector)
        assert activation.kind == expected_kind
        if exponent is not None:
            actual = dict(activation.parameters)["exponent"]
            assert math.isclose(actual, exponent, rel_tol=2e-15)
            if slot == "integer_power":
                assert type(actual) is int
                assert 2 <= actual < 6
                integers_seen.add(actual)
        assert rng.bit_generator.state == oracle.bit_generator.state
    frozen = sampler.freeze()
    encoded = json.loads(json.dumps(asdict(frozen)))
    assert priors.source_activation_selections_from_dict(encoded) == frozen
    records.append(encoded)
assert len(slots_seen) == 18
assert integers_seen == {2, 3, 4, 5}
record("480-local-independent-slot-catalog-exponent-draws", sorted(slots_seen))
record("high-exclusive-integers", sorted(integers_seen))

for mode in ("local", "global", "meta"):
    for policy in ("shared", "refresh"):
        sampler = priors.SourceActivationSampler(
            np.random.default_rng(73), mode=mode, policy=policy
        )
        outputs = {"plain": [], "standard": []}
        for i in range(30):
            selector = "plain" if i % 2 else "standard"
            outputs[selector].append(sampler.sample(selector))
        frozen = sampler.freeze()
        raw = json.loads(json.dumps(asdict(frozen)))
        (args.output / f"{mode}-{policy}.json").write_text(json.dumps(raw, indent=2) + "\n")
        assert priors.source_activation_selections_from_dict(raw) == frozen
        if mode == "global":
            assert all(len(set(values)) == 1 for values in outputs.values())
        prefs = frozen.categorical.preferences
        names = [p.request.name for p in prefs]
        if mode == "meta" and policy == "refresh":
            assert len(prefs) == len(frozen.categorical.draws)
        else:
            assert len(names) == len(set(names))
        if mode == "meta":
            assert all(p.weights is not None and p.hyperparameters is not None for p in prefs)
            assert all(p.mean is not None for p in frozen.numerical.preferences)
        record(
            f"scope-{mode}-{policy}",
            {"preferences": len(prefs), "parameters": frozen.parameter_count},
        )

raw = records[0]
for label in (
    "drop",
    "reorder",
    "duplicate",
    "orphan",
    "alter-primitive",
    "wrong-slot",
    "wrong-child",
    "scope",
    "exponent",
):
    data = copy.deepcopy(raw)
    selections = data["selections"]
    if label == "drop":
        selections.pop()
    elif label == "reorder":
        selections[0], selections[1] = selections[1], selections[0]
    elif label == "duplicate":
        selections[1] = copy.deepcopy(selections[0])
    elif label == "orphan":
        data["categorical"]["draws"].append(copy.deepcopy(data["categorical"]["draws"][-1]))
    elif label == "alter-primitive":
        selections[0]["activation"]["kind"] = "identity"
    elif label == "wrong-slot":
        selections[0]["slot_draw"] = 1
    elif label == "wrong-child":
        selections[0]["primitive_draw"] = 2
    elif label == "scope":
        data["numerical"]["default_mode"] = "global"
    else:
        selected = next(s for s in selections if s["exponent_draw"] is not None)
        selected["activation"]["parameters"][0][1] += 1
    reject(label, data)

# Rejection after a retained successful draw must roll back both child scopes and RNG.
rng = np.random.default_rng(0)
sampler = priors.SourceActivationSampler(
    rng, mode="local", categorical_limits=priors.PreferenceLimits(max_draws=3)
)
sampler.sample("plain")
before_state, before_record = copy.deepcopy(rng.bit_generator.state), sampler.freeze()
try:
    sampler.sample("plain")
except ValueError as error:
    if "limit" not in str(error):
        raise
else:
    raise AssertionError("expected fixed-child admission rejection")
assert rng.bit_generator.state == before_state
assert sampler.freeze() == before_record
record("hierarchical-failure-after-success-atomic", "injected RNG and frozen scopes unchanged")

for kind in ("numerical", "categorical"):
    original_rng = np.random.default_rng(94)
    if kind == "numerical":
        scope = priors.ParameterSampler(original_rng, default_mode="local")
        first = priors.NumericalParameter("one", 0.0, 1.0)
        second = priors.NumericalParameter("two", 0.0, 1.0)
        scope.numerical(first)
    else:
        scope = priors.CategoricalSampler(original_rng, default_mode="local")
        first = priors.CategoricalParameter("one", ("a", "b"))
        second = priors.CategoricalParameter("two", ("a", "b"))
        scope.categorical(first)
    before = scope.freeze()
    before_rng = copy.deepcopy(original_rng.bit_generator.state)
    fork = scope.fork(np.random.default_rng(95))
    if kind == "numerical":
        fork.numerical(second)
    else:
        fork.categorical(second)
    assert scope.freeze() == before
    assert original_rng.bit_generator.state == before_rng
    assert len(fork.freeze().draws) == 2
    assert len(before.draws) == 1
    record(f"{kind}-fork-isolation", "original record and RNG unchanged")

# Exercise cached numerical powers explicitly; seed 4 chooses plain integer power.
sampler = priors.SourceActivationSampler(np.random.default_rng(4), mode="global")
outputs = [sampler.sample("plain") for _ in range(6)]
assert all(a.kind == "integer_power" for a in outputs)
assert len(set(outputs)) == 1
assert len(sampler.freeze().numerical.preferences) == 1
record("global-numerical-exponent-reuse", asdict(outputs[0]))

rng = np.random.default_rng(4)
sampler = priors.SourceActivationSampler(
    rng, mode="local", numerical_limits=priors.PreferenceLimits(max_parameters=1)
)
before_state, before_record = copy.deepcopy(rng.bit_generator.state), sampler.freeze()
try:
    sampler.sample("plain")
except ValueError as error:
    if "limit" not in str(error):
        raise
else:
    raise AssertionError("expected numerical-child admission rejection")
assert rng.bit_generator.state == before_state
assert sampler.freeze() == before_record
record("numerical-child-failure-atomic", "injected RNG and frozen scopes unchanged")

rng = np.random.default_rng(4)
before_state = copy.deepcopy(rng.bit_generator.state)
try:
    priors.calibrate_source_activation(
        priors.CausalActivation(),
        np.zeros((2, 1)),
        rng=rng,
        scale=1.0,
        calibration_id="reject-before-rng",
        limits=priors.CausalLimits(parameters=1),
    )
except ValueError as error:
    if "limit" not in str(error):
        raise
else:
    raise AssertionError("expected calibration resource rejection")
assert rng.bit_generator.state == before_state
record("calibration-admission-before-randomness", "injected RNG unchanged")
