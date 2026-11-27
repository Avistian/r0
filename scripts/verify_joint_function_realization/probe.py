"""Independent raw-law checks, JSON mutation probes and reproducible fixtures."""

import copy
import json
import pickle
from dataclasses import asdict

import numpy as np
import torch
from common import OUTPUT, build, pin, plan_for, priors
from oracles import close, oracle


def leaves(value, path=()):
    if type(value) is dict:
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif type(value) is list:
        for key, item in enumerate(value):
            yield from leaves(item, (*path, key))
    elif type(value) in (int, float):
        yield path, value


def set_leaf(payload, path, value):
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def reject(payload, decisions, label):
    try:
        priors.joint_function_realization_from_dict(payload, decisions=decisions)
    except (ValueError, TypeError, KeyError) as exc:
        return type(exc).__name__
    (OUTPUT / "first-defect.json").write_text(json.dumps(payload, separators=(",", ":")))
    raise AssertionError(f"Malformed payload accepted: {label}")


def verify(family, seed=0, width=2, rows=4, centers=None, adversarial=False):
    plan, decisions = plan_for(family, seed, width, rows, centers)
    sample = (np.arange(rows * width, dtype=float).reshape(rows, width) % 7) - 2.3
    sample[:, 0] *= 3
    raw, fit = np.random.default_rng(303), np.random.default_rng(404)
    before = pickle.dumps(decisions)
    result = build(plan, decisions, sample, raw, fit)
    expected_raw, expected_fit = np.random.default_rng(303), np.random.default_rng(404)
    evaluate = oracle(plan, result, sample, expected_raw, expected_fit)
    assert raw.bit_generator.state == expected_raw.bit_generator.state
    assert fit.bit_generator.state == expected_fit.bit_generator.state
    assert pickle.dumps(decisions) == before
    query = np.vstack((sample, np.zeros((1, width)), sample[0:1] + 0.37))
    actual = priors.evaluate_frozen_function(result.function, torch.tensor(query)).numpy()
    close(actual, evaluate(query))
    payload = json.loads(json.dumps(asdict(result)))
    assert priors.joint_function_realization_from_dict(payload, decisions=decisions) == result
    chunk = np.concatenate(
        [
            priors.evaluate_frozen_function(result.function, torch.tensor(row[None])).numpy()
            for row in query
        ]
    )
    close(actual, chunk)
    if adversarial:
        (OUTPUT / f"{family}-fixture.json").write_text(
            json.dumps(
                {
                    "payload": payload,
                    "decisions": asdict(decisions),
                    "sample": sample.tolist(),
                    "query": query.tolist(),
                    "expected": evaluate(query).tolist(),
                },
                separators=(",", ":"),
            )
        )
        count = 0
        for path, value in leaves(payload):
            changed = copy.deepcopy(payload)
            set_leaf(changed, path, bool(value))
            reject(changed, decisions, (family, path, "bool"))
            count += 1
        required = {"plan", "function"}
        required |= {"matrix"} if family in ("lin", "quad", "disc", "em") else set()
        for name in required:
            changed = copy.deepcopy(payload)
            del changed[name]
            reject(changed, decisions, (family, name, "missing"))
        print(f"PASS adversarial {family}: {count} numeric bool substitutions", flush=True)
    print(f"PASS {family} seed={seed} width={width} rows={rows}", flush=True)
    return plan


def main():
    pin()
    for family in ("lin", "quad", "gp", "tree", "disc", "em"):
        verify(family, adversarial=True)
    for width in (1, 20, 21, 24):
        verify("quad", width=width)
    for seed in range(6):
        verify("gp", seed=seed)
    for rows, centers in ((1, None), (2, None), (4, None), (4, 1), (4, 9)):
        verify("disc", rows=rows, centers=centers)
    seen = set()
    for seed in range(500):
        plan, _ = plan_for("prod", seed, rows=4)
        pair = tuple(child.family for child in plan.children)
        if pair in seen:
            continue
        verify("prod", seed=seed, adversarial=not seen)
        seen.add(pair)
        if len(seen) == 25:
            break
    assert len(seen) == 25, seen
    print("PASS all 25 ordered cheap pairs", flush=True)


if __name__ == "__main__":
    main()
