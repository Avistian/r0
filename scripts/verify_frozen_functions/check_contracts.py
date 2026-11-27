"""Public JSON corruption, shape, version, recursion and geometry boundaries."""

import copy
import json
import math
from dataclasses import asdict, replace

import torch
from evidence_common import activation, check, finish, p, reject

linear = p.FrozenLinearFunction(((1.0, -2.0), (3.0, 4.0)))
quad = p.FrozenQuadraticFunction(2, (0, 1), (((1.0, 0.0, 2.0), (3.0, 4.0, 5.0), (6.0, 7.0, 8.0)),))
tree = p.FrozenTreeFunction(2, ((0, 1),), ((0.0, 1.0),), (((0.0,), (1.0,), (2.0,), (3.0,)),))
disc = p.FrozenDiscretizationFunction(((0.0, 1.0), (2.0, 3.0)), linear)
rff = p.FrozenRFFFunction(((0.0, 1.0), (2.0, 3.0)), (0.0, 1.0), ((1.0,), (2.0,)))
em = p.FrozenEMFunction(((0.0, 1.0), (2.0, 3.0)), (1.0, 2.0), linear)
product = p.FrozenProductFunction(linear, linear)
mlp = p.FrozenMLPFunction((linear,), (activation(2),), end_activation=True)
fixtures = [linear, quad, tree, disc, rff, em, product, mlp]
for f in fixtures:
    original = json.loads(json.dumps(asdict(f)))
    for key, value in (
        ("version", "unknown-v77"),
        ("version", [1]),
        ("extra", 1),
        ("version", None),
        ("version", True),
    ):
        payload = copy.deepcopy(original)
        payload[key] = value
        reject(
            f"{type(f).__name__}-{key}-{value}",
            lambda data=payload: p.frozen_function_from_dict(data),
            payload=payload,
        )
    payload = copy.deepcopy(original)
    del payload["version"]
    reject(
        f"{type(f).__name__}-missing-version",
        lambda data=payload: p.frozen_function_from_dict(data),
        payload=payload,
    )
    for key, value in original.items():
        if isinstance(value, list):
            for invalid in ("array-as-string", {"0": value[0]}):
                payload = copy.deepcopy(original)
                payload[key] = invalid
                reject(
                    f"{type(f).__name__}-{key}-{type(invalid).__name__}",
                    lambda data=payload: p.frozen_function_from_dict(data),
                    payload=payload,
                )

for value in (math.nan, math.inf, -math.inf, True, "2", None, 10**400):
    payload = {"version": "frozen-linear-function-v1", "matrix": [[value]]}
    reject(
        f"linear-bad-coefficient-{str(value)[:24]}",
        lambda data=payload: p.frozen_function_from_dict(data),
    )
for bad in ([], [[]], [[1.0], [2.0, 3.0]], [1.0], [[[]]]):
    reject(
        f"linear-malformed-{bad}",
        lambda value=bad: p.frozen_function_from_dict(
            {"version": "frozen-linear-function-v1", "matrix": value}
        ),
    )
for missing in (None, [], "record", 1):
    reject(f"root-not-object-{missing}", lambda data=missing: p.frozen_function_from_dict(data))
reject("nonstring-key", lambda: p.frozen_function_from_dict({1: 0}))
reject(
    "payload-cannot-own-limits",
    lambda: p.frozen_function_from_dict({**asdict(linear), "limits": {"parameters": 2**60}}),
)
cycle = []
cycle.append(cycle)
reject("cyclic-list", lambda: p.frozen_function_from_dict({"matrix": cycle}))
cycle_dict = {}
cycle_dict["self"] = cycle_dict
reject("cyclic-dict", lambda: p.frozen_function_from_dict(cycle_dict))
deep = [1]
for _ in range(26):
    deep = [deep]
reject("depth-27", lambda: p.frozen_function_from_dict({"x": deep}))
reject(
    "raw-node-count",
    lambda: p.frozen_function_from_dict(
        {"matrix": [None] * 100}, limits=p.CausalLimits(parameters=1, nodes=1)
    ),
)
reject(
    "raw-numeric-count",
    lambda: p.frozen_function_from_dict(asdict(linear), limits=p.CausalLimits(parameters=1)),
)
reject(
    "raw-text-bytes",
    lambda: p.frozen_function_from_dict(
        {"x": "é" * 1000}, limits=p.CausalLimits(working_bytes=1024)
    ),
)

for bad in (em, mlp, product):
    for side in ("left", "right"):
        reject(
            f"product-disallowed-{side}-{type(bad).__name__}",
            lambda field=side, child=bad: replace(product, **{field: child}),
        )
        payload = asdict(product)
        payload[side] = asdict(bad)
        reject(
            f"json-product-disallowed-{side}-{type(bad).__name__}",
            lambda data=payload: p.frozen_function_from_dict(data),
            payload=payload,
        )
reject("product-shape", lambda: p.FrozenProductFunction(linear, rff))
for field, value in (
    ("indices", (1, 0)),
    ("indices", (0, 0)),
    ("indices", (True, 1)),
    ("input_width", True),
    ("matrices", (((1.0,),),)),
):
    reject(f"quad-{field}-{value}", lambda k=field, v=value: replace(quad, **{k: v}))
for field, value in (
    ("features", ((True, 1),)),
    ("features", ((0, 2),)),
    ("leaves", (((0.0,),),)),
    ("thresholds", ((0.0,) * 8,)),
    ("thresholds", ((0.0,),) * 129),
):
    reject(f"tree-{field}-{str(value)[:30]}", lambda k=field, v=value: replace(tree, **{k: v}))
for field, value in (
    ("phases", (0.0, 2 * math.pi)),
    ("phases", (-0.1, 0.0)),
    ("phases", (0.0,)),
    ("coefficients", ((1.0,),)),
):
    reject(f"rff-{field}-{value}", lambda k=field, v=value: replace(rff, **{k: v}))
for value in (0.49, 4.01, True, math.nan):
    reject(f"disc-p-{value}", lambda v=value: replace(disc, p=v))
for field, value in (
    ("scales", (0.0, 1.0)),
    ("scales", (-1.0, 1.0)),
    ("scales", (math.inf, 1.0)),
    ("p", 0.5),
    ("q", 2.01),
    ("centers", ((0.0, 1.0),)),
):
    reject(f"em-{field}-{value}", lambda k=field, v=value: replace(em, **{k: v}))
for field, value in (
    ("linears", ()),
    ("linears", (linear,) * 4),
    ("start_activation", 1),
    ("end_activation", 1),
    ("activations", ()),
    ("activations", (activation(1),)),
    ("linears", (linear, rff)),
):
    reject(f"mlp-{field}-{str(value)[:30]}", lambda k=field, v=value: replace(mlp, **{k: v}))
for width in (1, 127):
    first = p.FrozenLinearFunction(((1.0, -1.0),) * width)
    last = p.FrozenLinearFunction(((1.0 / width,) * width,))
    check(
        f"mlp-hidden-bound-{width}",
        p.FrozenMLPFunction((first, last), (activation(width),)),
        [[1.0, -1.0], [-1.0, 0.0]],
    )
wide = p.FrozenLinearFunction(((1.0, 0.0),) * 128)
reject(
    "mlp-hidden-128",
    lambda: p.FrozenMLPFunction(
        (wide, p.FrozenLinearFunction(((1.0,) * 128,))), (activation(128),)
    ),
)
a = p.FrozenLinearFunction(((1.0, 0.0),) * 3)
b = p.FrozenLinearFunction(((1.0,) * 3,) * 4)
c = p.FrozenLinearFunction(((1.0,) * 4,))
reject(
    "mlp-hidden-inconsistent",
    lambda: p.FrozenMLPFunction((a, b, c), (activation(3), activation(4))),
)
for bad_input in (
    torch.zeros(2),
    torch.zeros(0, 2),
    torch.zeros(1, 3),
    torch.ones(1, 2, dtype=torch.int64),
    torch.tensor([[math.nan, 0.0]]),
    torch.tensor([[math.inf, 0.0]]),
    [[1.0, 2.0]],
):
    reject(f"input-{str(bad_input)}", lambda x=bad_input: p.evaluate_frozen_function(linear, x))


class TensorSubclass(torch.Tensor):
    pass


reject(
    "tensor-subclass",
    lambda: p.evaluate_frozen_function(linear, torch.zeros(1, 2).as_subclass(TensorSubclass)),
)
reject(
    "nonfinite-final-output",
    lambda: p.evaluate_frozen_function(
        p.FrozenLinearFunction(((1e308,),)), torch.tensor([[2.0]], dtype=torch.float64)
    ),
)
finish()
