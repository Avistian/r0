"""Recursive adversarial field/type mutations of independently constructed registries."""

import copy
import json
from dataclasses import asdict

import numpy as np
from common import rejected, require, save, start

from r0.modules import priors


def walk(value, prefix=()):
    yield prefix, value
    if type(value) is dict:
        for key, item in value.items():
            yield from walk(item, (*prefix, key))
    elif type(value) is list:
        for key, item in enumerate(value):
            yield from walk(item, (*prefix, key))


def replace(original, path, value):
    result = copy.deepcopy(original)
    if not path:
        return value
    parent = result
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return result


def main():
    output = start()
    context = priors.JointSamplingContext(np.random.default_rng(812), mode="meta")
    with context.transaction("scope") as scope:
        with scope.transaction("empty"):
            pass
        for mode in ("local", "global", "meta"):
            scope.numerical(mode, priors.NumericalParameter(f"n-{mode}", -1.0, 3.0, mode=mode))
            scope.categorical(
                f"c-{mode}", priors.CategoricalParameter(f"c-{mode}", ("a", "b", "c"), mode=mode)
            )
    original = json.loads(json.dumps(asdict(context.freeze_registry())))
    require(
        priors.joint_operation_registry_from_dict(original) == context.freeze_registry(),
        "valid control failed",
    )
    save(output, "valid", original)
    count = 0
    covered = []
    for path, value in walk(original):
        variants = []
        if type(value) is dict:
            variants.extend(("record-type", item) for item in ([], "", 0, None))
            variants.append(("extra-field", {**value, "UNEXPECTED": 1}))
            variants.extend(
                (f"missing-{key}", {k: v for k, v in value.items() if k != key}) for key in value
            )
        elif type(value) is list:
            variants.extend(("array-type", item) for item in ({}, "", 0, None))
        for label, variant in variants:
            mutated = replace(original, path, variant)
            try:
                rejected(lambda mutated=mutated: priors.joint_operation_registry_from_dict(mutated))
            except Exception as error:
                save(output, "REJECT-payload", mutated)
                save(
                    output,
                    "REJECT-result",
                    {
                        "path": path,
                        "mutation": label,
                        "error": type(error).__name__,
                        "message": str(error),
                    },
                )
                raise
            count += 1
        if variants:
            covered.append({"path": path, "cases": len(variants)})
    for index in range(0, len(covered), 30):
        save(output, f"coverage-{index // 30:02}", covered[index : index + 30])
    require(any("weights" in item["path"] for item in covered), "weights not exercised")
    require(any("hyperparameters" in item["path"] for item in covered), "hyperparameters absent")
    save(
        output, "result", {"verdict": "PASS", "malformed_cases": count, "containers": len(covered)}
    )
    print(f"PASS: {count} malformed container/field cases across {len(covered)} containers")


if __name__ == "__main__":
    main()
