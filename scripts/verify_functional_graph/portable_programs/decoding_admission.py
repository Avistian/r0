"""Independent malformed-container matrix and preallocation resource sentinel."""

import copy
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r0.modules import priors as p
from r0.modules.priors import functional_recipe_from_dict

OUT = Path(os.environ["R0_VERIFY_OUT"])
r = p.FunctionalCausalRecipe(
    p.LinearRecipe(rows=16, features=1, signal_features=1, active_features=1, context_rows=8),
    (
        p.FunctionalCausalNode(
            (),
            p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0))),
            root=p.JointRootBase("uniform", 2),
            pipeline=p.CausalNodePipeline(2, stages=(p.CausalActivation(),)),
        ),
    ),
    ((0, 0),),
    ((0, 1),),
)
raw = json.loads(json.dumps(asdict(r)))


def reject(label, payload, **kwargs):
    try:
        functional_recipe_from_dict(payload, **kwargs)
    except ValueError as e:
        print("PASS reject", label, str(e), flush=True)
    else:
        raise AssertionError("REJECT unexpectedly admitted " + label)


for path in [
    ("nodes",),
    ("nodes", 0),
    ("nodes", 0, "parents"),
    ("nodes", 0, "function"),
    ("nodes", 0, "function", "matrix"),
    ("nodes", 0, "function", "matrix", 0),
    ("nodes", 0, "root"),
    ("nodes", 0, "exogenous"),
    ("nodes", 0, "exogenous", "inputs"),
    ("nodes", 0, "exogenous", "factor"),
    ("nodes", 0, "pipeline"),
    ("nodes", 0, "pipeline", "stages"),
    ("nodes", 0, "pipeline", "stages", 0),
    ("nodes", 0, "pipeline", "stages", 0, "parameters"),
    ("nodes", 0, "pipeline", "converters"),
    ("observed",),
    ("observed", 0),
    ("target",),
    ("limits",),
    ("task",),
]:
    payload = copy.deepcopy(raw)
    at = payload
    for key in path[:-1]:
        at = at[key]
    old = at[path[-1]]
    at[path[-1]] = {} if isinstance(old, list) else []
    reject("container " + repr(path), payload)
for path in [
    (),
    ("nodes", 0),
    ("nodes", 0, "function"),
    ("nodes", 0, "root"),
    ("nodes", 0, "pipeline"),
    ("nodes", 0, "pipeline", "stages", 0),
]:
    payload = copy.deepcopy(raw)
    at = payload
    for key in path:
        at = at[key]
    at["unrecognized"] = 0
    reject("unknown " + repr(path), payload)
for key, value in [
    ("nodes", 1),
    ("node_channels", 1),
    ("total_channels", 1),
    ("parameters", 1),
    ("working_bytes", 1024),
]:
    lim = replace(p.CausalLimits(), **{key: value})
    if key == "nodes":
        assert functional_recipe_from_dict(raw, limits=lim).limits.nodes == 1
        print("PASS one actual graph node with dimension 2")
    else:
        reject("caller " + key, raw, limits=lim)
    payload = copy.deepcopy(raw)
    payload["limits"][key] = value
    if key != "nodes":
        reject("payload " + key, payload)
# Payload cannot loosen caller-owned caps.
payload = copy.deepcopy(raw)
payload["limits"]["node_channels"] = 1000000
reject("payload escalation", payload, limits=p.CausalLimits(node_channels=1))
# Depth and retained text admission occur before typed decoding.
payload = copy.deepcopy(raw)
nested = []
for _ in range(26):
    nested = [nested]
payload["unexpected"] = nested
reject("depth cap", payload)
payload = copy.deepcopy(raw)
payload["unexpected"] = "x" * 4096
reject("text cap", payload, limits=p.CausalLimits(working_bytes=1024))
# Covariance validation must not allocate numerical arrays for an over-channel graph.
base = p.JointRootBase(
    "covariance",
    2,
    gaussian_factor=((1.0, 0.0), (0.0, 1.0)),
    weights=p.UniformWeightVector((0.5, 0.5)),
    factor=((0.5, 0.0), (0.0, 0.5)),
    covariance_base="gaussian",
)
payload = json.loads(json.dumps(asdict(replace(r, nodes=(replace(r.nodes[0], root=base),)))))
(OUT / "over_channel_covariance_payload.json").write_text(json.dumps(payload, indent=2) + "\n")
allocations = []
real = np.asarray


def observe(*args, **kwargs):
    result = real(*args, **kwargs)
    allocations.append(
        {"shape": list(result.shape), "dtype": str(result.dtype), "bytes": result.nbytes}
    )
    return result


with patch.object(np, "asarray", observe):
    reject("over-channel covariance", payload, limits=p.CausalLimits(node_channels=1))
report = {
    "caller_node_channels": 1,
    "payload_width": 2,
    "numpy_arrays_before_rejection": allocations,
}
(OUT / "preallocation_result.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
assert not allocations, (
    "REJECT: over-channel covariance decoded into numerical arrays before graph admission"
)
print("PASS: strict JSON and aggregate caller/payload admission before numerical arrays")
