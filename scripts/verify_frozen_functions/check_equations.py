"""All eight source equations, compositions and fixed-state behavior."""

import itertools
import math
from decimal import Decimal, localcontext

import numpy as np
from evidence_common import activation, check, finish, matrix, p

X = [[-2.0, 0.5], [0.0, 0.0], [1.25, -3.0], [2.0, 2.0]]
linear = p.FrozenLinearFunction(((1.0, -2.0), (3.0, 0.5)))
quad = p.FrozenQuadraticFunction(
    2,
    (0, 1),
    (
        ((1.0, 2.0, 3.0), (-4.0, 5.0, 6.0), (7.0, 8.0, -9.0)),
        ((0.0, 1.0, -2.0), (3.0, 0.0, 4.0), (5.0, -6.0, 7.0)),
    ),
)
tree = p.FrozenTreeFunction(
    2,
    ((0, 1), (1, 0)),
    ((0.0, 0.5), (-1.0, 1.25)),
    (
        ((1.0, 2.0), (3.0, 5.0), (7.0, 11.0), (13.0, 17.0)),
        ((-1.0, 0.0), (2.0, -3.0), (4.0, -5.0), (6.0, -7.0)),
    ),
)
rff = p.FrozenRFFFunction(
    ((0.5, -1.0, 2.0), (-0.25, 3.0, -2.5)),
    (0.0, math.pi / 3, math.nextafter(2 * math.pi, 0.0)),
    ((1.0, -2.0), (3.0, 4.0), (-5.0, 6.0)),
)
disc = p.FrozenDiscretizationFunction(((-2.0, 0.0), (1.0, 2.0), (3.0, -1.0)), linear, 0.5)
em = p.FrozenEMFunction(
    ((-2.0, 0.0), (1.0, 2.0), (3.0, -1.0)),
    (0.7, 1.4, 2.1),
    p.FrozenLinearFunction(((1.0, -2.0, 3.0), (4.0, 0.0, -5.0))),
    1.5,
    1.75,
)
for name, function in [
    ("linear", linear),
    ("quadratic", quad),
    ("tree", tree),
    ("rff", rff),
    ("discretization", disc),
    ("em", em),
]:
    check(name, function, X)

cheap = [linear, quad, tree, disc, rff]
for i, j in itertools.product(range(5), repeat=2):
    check(f"product-{i}-{j}", p.FrozenProductFunction(cheap[i], cheap[j]), X)

rng = np.random.default_rng(20261127)
for depth, start, end in itertools.product(range(1, 4), (False, True), (False, True)):
    widths = [2] + [3] * (depth - 1) + [2]
    layers = tuple(
        p.FrozenLinearFunction(matrix(rng.normal(size=(b, a))))
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    )
    act_widths = ([2] if start else []) + widths[1:-1] + ([2] if end else [])
    acts = tuple(
        activation(
            w,
            i,
            kind=("tanh", "relu", "signed_power")[i % 3],
            parameters=(("exponent", 1.5),) if i % 3 == 2 else (),
        )
        for i, w in enumerate(act_widths)
    )
    check(f"mlp-{depth}-{start}-{end}", p.FrozenMLPFunction(layers, acts, start, end), X)
for kind in ("softmax", "rank", "argsort"):
    check(
        f"mlp-channelwise-{kind}",
        p.FrozenMLPFunction((linear,), (activation(2, kind=kind),), end_activation=True),
        X,
    )
params = (
    ("input_scale", 1.5),
    ("input_shift", 0.1),
    ("output_scale", -2.0),
    ("output_shift", 0.3),
    ("epsilon", 0.25),
    ("exponent", 0.5),
)
check(
    "mlp-six-authored-parameters",
    p.FrozenMLPFunction(
        (linear,),
        (activation(2, kind="reciprocal_abs_power", parameters=params),),
        end_activation=True,
    ),
    X,
)

for width in (1, 20, 21):
    indices = tuple(range(width)) if width <= 20 else tuple(range(20, 0, -1))
    k = len(indices) + 1
    q = matrix(np.arange(k * k).reshape(k, k) / (k * k))
    check(
        f"quadratic-width-{width}",
        p.FrozenQuadraticFunction(width, indices, (q,)),
        [list(np.linspace(-1, 1, width)), [0.0] * width],
    )
for trees, depth in ((1, 1), (128, 7)):
    f = p.FrozenTreeFunction(
        2,
        ((0,) * depth,) * trees,
        ((0.0,) * depth,) * trees,
        tuple(tuple((float(t + leaf),) for leaf in range(2**depth)) for t in range(trees)),
    )
    check(f"tree-bound-{trees}-{depth}", f, [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
for leaves in (
    (1e308, 1e308),
    (-1e308, -1e308),
    (1e308, 1e308, -1e308),
    (1e308, 1e308, -1e308, -1e308),
):
    with localcontext() as ctx:
        ctx.prec = 1100
        expected = float(sum(Decimal.from_float(v) for v in leaves) / len(leaves))
    f = p.FrozenTreeFunction(
        1, ((0,),) * len(leaves), ((0.0,),) * len(leaves), tuple(((v,), (v,)) for v in leaves)
    )
    check(f"tree-extreme-{leaves}", f, [[-1.0], [1.0]], [[expected], [expected]], atol=0)

for power in (0.5, 1.0, 2.0, 4.0):
    check(
        f"disc-p-boundary-{power}", p.FrozenDiscretizationFunction(disc.centers, linear, power), X
    )
    check(
        f"disc-sole-center-{power}",
        p.FrozenDiscretizationFunction(((2.0, -3.0),), linear, power),
        X,
    )
    for magnitude in (math.ulp(0.0), 1e-308, 1.0, 1e10, 1e308):
        far = math.nextafter(magnitude, math.inf)
        for dims in (1, 3):
            centers = ((far,) + (0.0,) * (dims - 1), (magnitude,) + (0.0,) * (dims - 1))
            f = p.FrozenDiscretizationFunction(
                centers, p.FrozenLinearFunction(((1.0,) + (0.0,) * (dims - 1),)), power
            )
            check(
                f"disc-sparse-{power}-{magnitude}-{dims}",
                f,
                [[0.0] * dims],
                [[magnitude]],
                rtol=0,
                atol=0,
            )
    # Opposite-sign finite inputs overflow direct subtraction, yet ordering is defined.
    f = p.FrozenDiscretizationFunction(
        ((-1.7e308, 0.0), (-1.6e308, 0.0)), p.FrozenLinearFunction(((1e-308, 0.0),)), power
    )
    check(f"disc-overflow-{power}", f, [[1.7e308, 0.0]], [[-1.6]], rtol=2e-15, atol=0)

for power, exponent in itertools.product((1.0, 4.0), (1.0, 2.0)):
    check(
        f"em-boundaries-{power}-{exponent}",
        p.FrozenEMFunction(em.centers, em.scales, em.linear, power, exponent),
        X,
    )
# Constants have no dimension multiplier: at coincident centers, probability is inverse sigma.
check(
    "em-scalar-normalization",
    p.FrozenEMFunction(
        ((0.0, 0.0, 0.0),) * 2, (1.0, 2.0), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
    ),
    [[0.0] * 3],
    [[2 / 3, 1 / 3]],
)
for count, outputs in ((2, 1), (16, 1), (34, 17)):
    f = p.FrozenEMFunction(
        ((0.0,),) * count,
        (1.0,) * count,
        p.FrozenLinearFunction(
            tuple(tuple(float(i + j) for j in range(count)) for i in range(outputs))
        ),
    )
    check(f"em-count-bound-{count}-{outputs}", f, [[0.0], [1.0]])
# Decimal gives an independent stable result even when original squared sigma would overflow.
f = p.FrozenEMFunction(
    ((0.0,), (0.0,)), (1e308, 5e307), p.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
)
check("em-extreme-sigma", f, [[0.0]], [[1 / 3, 2 / 3]], rtol=1e-12)
# p=.5 ties with integer square coordinates: (a^2,b^2) and ((a+b)^2,0).
for a, b in itertools.product(range(1, 6), repeat=2):
    total = float((a + b) ** 2)
    f = p.FrozenDiscretizationFunction(
        ((total, 0.0), (float(a * a), float(b * b))), p.FrozenLinearFunction(((1.0, 0.0),)), 0.5
    )
    check(f"disc-exact-quasi-tie-{a}-{b}", f, [[0.0, 0.0]], [[total]], rtol=0, atol=0)
finish()
