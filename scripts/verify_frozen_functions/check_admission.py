"""Independent hand-derived geometry totals and exact/one-below admission."""

from dataclasses import asdict, replace
from unittest.mock import patch

import torch
from evidence_common import activation, finish, p, record_event, reject

linear = p.FrozenLinearFunction(((1.0, -2.0), (3.0, 4.0)))
quad = p.FrozenQuadraticFunction(
    2, (0, 1), (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),) * 2
)
tree = p.FrozenTreeFunction(2, ((0,),), ((0.0,),), (((1.0, 2.0), (3.0, 4.0)),))
disc = p.FrozenDiscretizationFunction(((0.0, 1.0), (2.0, 3.0), (-1.0, 4.0)), linear)
rff = p.FrozenRFFFunction(
    ((1.0, 0.0, 2.0), (0.0, 1.0, -1.0)), (0.0, 1.0, 2.0), ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
)
em = p.FrozenEMFunction(
    disc.centers, (1.0, 2.0, 3.0), p.FrozenLinearFunction(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
)
mlp = p.FrozenMLPFunction(
    (p.FrozenLinearFunction(((1.0, 0.0),) * 3), p.FrozenLinearFunction(((1.0, 0.0, 0.0),) * 2)),
    (activation(3),),
)
# Hand-derived table columns: node count, max width, total width, admitted parameters,
# per-row bytes, retained-parameter bytes, minimum working rows. Retained numbers
# plus 2*total widths form admitted parameters; bytes include all nested live work.
fixtures = [
    ("linear", linear, (2, 2, 4, 12, 448, 1536, 0)),
    ("quadratic", quad, (3, 3, 7, 35, 976, 4480, 0)),
    ("tree", tree, (2, 2, 4, 15, 544, 1920, 0)),
    ("rff", rff, (3, 3, 7, 29, 688, 3712, 0)),
    ("discretization", disc, (3, 3, 7, 25, 1488, 3200, 3)),
    ("em", em, (3, 3, 7, 31, 1488, 3968, 3)),
    ("product-simple", p.FrozenProductFunction(linear, tree), (4, 2, 8, 27, 928, 3456, 0)),
    ("product-nested", p.FrozenProductFunction(disc, rff), (6, 3, 14, 54, 2112, 6912, 3)),
    ("mlp", mlp, (4, 3, 10, 54, 1168, 6912, 0)),
]
for name, f, (nodes, width, total, params, row_bytes, fixed, floor) in fixtures:
    for rows in (1, 3, 4):
        expected_bytes = fixed + row_bytes * max(rows, floor)
        limits = p.CausalLimits(nodes, width, total, params, expected_bytes)
        values = torch.zeros((rows, 2), dtype=torch.float32)
        # Limits and expected totals are literals, never implementation-derived estimates.
        p.evaluate_frozen_function(f, values, limits=limits)
        p.frozen_function_from_dict(asdict(f), limits=limits)
        record_event(
            {
                "name": f"{name}-exact-{rows}",
                "status": "PASS",
                "rows": rows,
                "geometry": [nodes, width, total, params],
                "bytes": expected_bytes,
                "record": asdict(f),
            }
        )
        for field in ("nodes", "node_channels", "total_channels", "parameters", "working_bytes"):
            low = replace(limits, **{field: getattr(limits, field) - 1})
            # Resource rejection must happen before conversion or parameter tensor allocation.
            with (
                patch.object(
                    torch.Tensor, "to", side_effect=AssertionError("conversion before admission")
                ),
                patch.object(
                    torch.Tensor,
                    "new_tensor",
                    side_effect=AssertionError("allocation before admission"),
                ),
            ):
                reject(
                    f"{name}-{rows}-{field}-one-below",
                    lambda cap=low, function=f, x=values: p.evaluate_frozen_function(
                        function, x, limits=cap
                    ),
                    accepted=(ValueError,),
                )
            if field != "working_bytes" or rows <= max(1, floor):
                reject(
                    f"{name}-{rows}-{field}-decode-one-below",
                    lambda cap=low, function=f: p.frozen_function_from_dict(
                        asdict(function), limits=cap
                    ),
                    accepted=(ValueError,),
                )
    cap = p.CausalLimits(nodes, width, total, params, fixed + row_bytes * max(1, floor))
    reject(
        f"{name}-actual-row-overflow",
        lambda limit=cap, function=f, count=floor: p.evaluate_frozen_function(
            function, torch.zeros((max(1, count) + 1, 2)), limits=limit
        ),
        accepted=(ValueError,),
    )
finish()
