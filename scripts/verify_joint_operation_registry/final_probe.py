"""Scalar-type corruption and intentional baseline path-reservation difference."""

import json
from dataclasses import asdict

import numpy as np
from common import BUNDLE, rejected, require, save, start
from schema_probe import replace, walk

from r0.modules import priors


def main():
    output = start()
    context = priors.JointSamplingContext(np.random.default_rng(1))
    first = priors.sample_joint_weights(context, "empty", width=2, family="uniform")
    try:
        priors.sample_joint_weights(context, "empty", width=3, family="uniform")
    except ValueError:
        reuse = "REJECTED"
    else:
        reuse = "ACCEPTED"
    execution = json.loads((output / "execution.json").read_text())
    baseline = execution["head"] == "0c3a032c7c27b02ac878ae72a890f65c6c4c51e3"
    require(reuse == ("ACCEPTED" if baseline else "REJECTED"), "unexpected empty path behavior")
    scalars = 0
    if not baseline:
        original = json.loads((BUNDLE / "schema-final/valid.json").read_text())
        for path, value in walk(original):
            if type(value) in (dict, list):
                continue
            for replacement in ({}, []):
                mutated = replace(original, path, replacement)
                try:
                    rejected(
                        lambda mutated=mutated: priors.joint_operation_registry_from_dict(mutated)
                    )
                except Exception as error:
                    save(output, "REJECT-payload", mutated)
                    save(
                        output,
                        "REJECT-result",
                        {"path": path, "error": type(error).__name__, "message": str(error)},
                    )
                    raise
                scalars += 1
    save(
        output,
        "result",
        {
            "verdict": "PASS",
            "empty_reuse": reuse,
            "first_plan": asdict(first),
            "malformed_scalar_cases": scalars,
        },
    )
    print(f"PASS: empty reuse {reuse}; {scalars} malformed scalar cases")


if __name__ == "__main__":
    main()
