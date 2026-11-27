"""Execute only cached source transform bodies with independently authored state."""

import ast
import copy
import hashlib
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from evidence_common import act_oracle, activation, finish, matrix, p, record_event

ROOT = Path(__file__).resolve().parent
source = (ROOT / "I2func.py").read_text()
assert hashlib.sha256(source.encode()).hexdigest() == (
    "ff2939197b353f6d11a3fbaf0e27586998bdf93527dad11f24698ff296ce2f6c"
)
namespace = {"torch": torch, "np": np}
methods = {}
extractions = {}
for cls in ast.parse(source).body:
    if not isinstance(cls, ast.ClassDef):
        continue
    for method in cls.body:
        if isinstance(method, ast.FunctionDef) and method.name == "_transform":
            node = copy.deepcopy(method)
            module = ast.Module(body=[node], type_ignores=[])
            exec(
                compile(ast.fix_missing_locations(module), str(ROOT / "I2func.py"), "exec"),
                namespace,
            )
            methods[cls.name] = namespace["_transform"]
            extractions[cls.name] = {
                "first_line": method.lineno,
                "last_line": method.end_lineno,
                "source": ast.get_source_segment(source, method),
            }
(ROOT / "legacy-transform-extractions.json").write_text(json.dumps(extractions, indent=2) + "\n")


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def legacy(f):
    """Bind frozen authored arrays to source attributes; no fits or random draws."""
    if type(f) is p.FrozenLinearFunction:
        kind, state = "RandomLinearFunction", {"matrix_": tensor(f.matrix).T}
    elif type(f) is p.FrozenQuadraticFunction:
        kind, state = (
            "RandomQuadraticFunction",
            {"idxs_": list(f.indices), "tensor_3d_": tensor(f.matrices)},
        )
    elif type(f) is p.FrozenTreeFunction:
        kind, state = (
            "RandomTreeFunction",
            {
                "split_dims_": torch.tensor(f.features, dtype=torch.long).flatten(),
                "split_points_": tensor(f.thresholds).flatten(),
                "n_trees_": len(f.features),
                "depth_": len(f.features[0]),
                "idx_multipliers_": 2 ** torch.arange(len(f.features[0])),
                "leaf_values_": tensor(f.leaves),
            },
        )
    elif type(f) is p.FrozenRFFFunction:
        kind, state = (
            "RandomGPFunction",
            {
                "freqs_": tensor(f.frequencies),
                "bias_": tensor(f.phases)[None],
                "weights_": tensor(f.coefficients),
            },
        )
    elif type(f) is p.FrozenDiscretizationFunction:
        # Source stores linearly transformed centers as targets during fit.
        centers = tensor(f.centers)
        kind, state = (
            "RandomDiscretizationFunction",
            {"centers_": centers, "targets_": legacy(f.linear)(centers), "p_": f.p},
        )
    elif type(f) is p.FrozenEMFunction:
        scales = tensor(f.scales)
        kind, state = (
            "RandomEMAssignmentFunction",
            {
                "x_ind_": tensor(f.centers),
                "stds_": scales,
                "consts": -torch.log(2 * math.pi * scales[None] ** 2) / 2,
                "out_func_": legacy(f.linear),
                "p_": f.p,
                "q_": f.q,
            },
        )
    elif type(f) is p.FrozenProductFunction:
        kind, state = "RandomProductFunction", {"fcts_": [legacy(f.left), legacy(f.right)]}
    else:
        assert type(f) is p.FrozenMLPFunction
        # Independent NumPy wrapper arithmetic verifies the source stage ordering.
        acts = [lambda x, a=a: tensor(act_oracle(a, x.numpy())) for a in f.activations]
        kind, state = (
            "RandomMLPFunction",
            {
                "start_act_": f.start_activation,
                "end_act_": f.end_activation,
                "linears_": [legacy(layer) for layer in f.linears],
                "acts_": acts,
            },
        )
    owner = SimpleNamespace(**state)
    method = methods[kind]
    return lambda x: method(owner, x)


def compare(name, f, rows):
    actual = p.evaluate_frozen_function(f, rows)
    expected = legacy(f)(rows)
    event = {
        "name": name,
        "record": asdict(f),
        "inputs": rows.tolist(),
        "source_transform_output": expected.tolist(),
        "observed": actual.tolist(),
        "rtol": 2e-12,
        "atol": 2e-12,
    }
    try:
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-12, atol=2e-12)
    except Exception as exc:
        event.update(status="REJECT", error=repr(exc))
        record_event(event)
        raise
    event["status"] = "PASS"
    record_event(event)


rng = np.random.default_rng(983270)
for width, outputs in itertools.product((1, 2, 4), (1, 3)):
    prefix = f"{width}x{outputs}"
    rows = tensor(rng.normal(size=(7, width)))
    linear = p.FrozenLinearFunction(matrix(rng.normal(size=(outputs, width))))
    quad = p.FrozenQuadraticFunction(
        width,
        tuple(range(width)),
        tuple(matrix(rng.normal(size=(width + 1, width + 1))) for _ in range(outputs)),
    )
    tree = p.FrozenTreeFunction(
        width,
        tuple(tuple(int(v) for v in row) for row in rng.integers(width, size=(3, 3))),
        matrix(rng.normal(size=(3, 3))),
        tuple(matrix(rng.normal(size=(8, outputs))) for _ in range(3)),
    )
    rff = p.FrozenRFFFunction(
        matrix(rng.normal(size=(width, 5))),
        tuple(float(v) for v in rng.uniform(0, 2 * math.pi, size=5)),
        matrix(rng.normal(size=(5, outputs))),
    )
    disc = p.FrozenDiscretizationFunction(matrix(rng.normal(size=(4, width))), linear, 0.75)
    em = p.FrozenEMFunction(
        matrix(rng.normal(size=(4, width))),
        (0.75, 1.0, 1.5, 2.0),
        p.FrozenLinearFunction(matrix(rng.normal(size=(outputs, 4)))),
        1.25,
        1.5,
    )
    cheap = [linear, quad, tree, rff, disc]
    for f in [*cheap, em]:
        compare(f"legacy-{prefix}-{type(f).__name__}", f, rows)
    for i, j in itertools.product(range(5), repeat=2):
        compare(
            f"legacy-{prefix}-product-{i}-{j}", p.FrozenProductFunction(cheap[i], cheap[j]), rows
        )
    for depth, start, end in itertools.product(range(1, 4), (False, True), (False, True)):
        widths = [width] + [3] * (depth - 1) + [outputs]
        linears = tuple(
            p.FrozenLinearFunction(matrix(rng.normal(size=(b, a))))
            for a, b in zip(widths[:-1], widths[1:], strict=True)
        )
        act_widths = ([width] if start else []) + widths[1:-1] + ([outputs] if end else [])
        acts = tuple(
            activation(w, i, kind="tanh" if i % 2 else "relu") for i, w in enumerate(act_widths)
        )
        compare(
            f"legacy-{prefix}-mlp-{depth}-{start}-{end}",
            p.FrozenMLPFunction(linears, acts, start, end),
            rows,
        )
finish()
