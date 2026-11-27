"""Cross-consumer preference identity, event sequence and literal edge fixtures."""

import copy
import itertools
import math
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from support import check, close, finish, raw, reject

from r0.modules import priors as p

for mode, category in itertools.product(("meta", "local", "global"), ("shared", "refresh")):
    ctx = p.JointSamplingContext(np.random.default_rng(985), mode=mode, categorical_policy=category)
    prior = p.JointMatrixPrior(families=("importance",))
    plans = [
        p.sample_joint_matrix(ctx, "matrix" + str(i), rows=2, columns=width, batch=2, prior=prior)
        for i, width in enumerate((2, 5))
    ]
    cloud = p.sample_joint_points(ctx, "cloud", count=7, dimension=3)
    node = p.sample_joint_weights(ctx, "node", width=7, batch=2)
    record = ctx.freeze()
    weights = [plan.weights for plan in plans] + [cloud.mixture_weights, cloud.factor_weights, node]
    for weight in weights:
        roles = [e.role for e in record.events if e.operation == weight.operation]
        check(
            roles
            == [f"decay:{i}" for i in range(weight.batch)]
            + [f"scale:{i}" for i in range(weight.batch)],
            "all decays before scales",
        )
    for name in ("simple_random_weights_decay_rate", "simple_random_weights_std_scale"):
        draws = [d for d in record.numerical.draws if d.request.name == name]
        check(len({d.preference for d in draws}) == 1, "single outer numerical preference identity")
        if mode == "global":
            check(len({d.units for d in draws}) == 1, "global shared scalar unit")
            for draw in draws:
                u = draw.units[0]
                if draw.request.boundary_mass:
                    u = min(1, max(0, 1.2 * u - 0.1))
                expected = math.exp(
                    (1 - u) * math.log(draw.request.low) + u * math.log(draw.request.high)
                )
                close(draw.values, (expected,), "explicit width-specific quantile remap")
        if mode == "meta":
            preference = record.numerical.preferences[draws[0].preference]
            check(
                preference.mean is not None and preference.concentration is not None,
                "shared Beta hyperparameters",
            )
    selector_draws = [d for d in record.categorical.draws if d.request.name == "random_matrix_type"]
    check(
        len({d.preference for d in selector_draws})
        == (2 if mode == "meta" and category == "refresh" else 1),
        "categorical shared versus refresh identity",
    )
    if mode == "meta":
        for pref in record.categorical.preferences:
            check(
                pref.hyperparameters is not record.numerical, "categorical nested scope is separate"
            )
            check(
                all(
                    d.request.name.startswith("simple_random_weights")
                    for d in pref.hyperparameters.draws
                ),
                "nested weight requests retained",
            )
    for plan in plans:
        plan.validate(record)
    cloud.validate(record)
    node.validate(record)

# Source array contracts use exact shape; scalar quantile refuses arrays without renaming.
for policy in ("scalar-quantile-v1", "strict-contract-v1"):
    rng = np.random.default_rng(18)
    ctx = p.JointSamplingContext(rng, mode="global", global_policy=policy)
    prior = p.JointMatrixPrior(families=("importance",), scalarization="source-weight-batches-v1")
    if policy == "strict-contract-v1":
        p.sample_joint_matrix(ctx, "first", rows=2, columns=3, batch=2, prior=prior)
    before, state = ctx.freeze(), copy.deepcopy(rng.bit_generator.state)
    reject(
        lambda ctx=ctx, prior=prior: p.sample_joint_matrix(
            ctx, "bad", rows=3, columns=5, batch=2, prior=prior
        ),
        "array policy mismatch " + policy,
    )
    check(
        ctx.freeze() == before and rng.bit_generator.state == state,
        "array mismatch transaction rollback",
    )

# Literal non-diagonal factors distinguish column weighting from rows or transpose.
cloud = p.KernelPointCloud(
    centers=((1.0, 2.0),),
    factors=(((1.0, 2.0), (3.0, 4.0)),),
    factor_weights=(p.UniformWeightVector((0.5, 0.25)),),
    component_weights=p.UniformWeightVector((0.75,)),
    assignments=(0,),
    innovations=((2.0, 4.0),),
    scale_mean=0.0,
    scale_coefficient=-2.0,
    scale_draws=(0.5,),
)
close(
    cloud.materialize(),
    ((1 + 3 / math.e, 2 + 7 / math.e),),
    "literal factor columns and signed exponent",
)
reject(
    lambda: replace(cloud, component_weights=p.UniformWeightVector((0.0,))).materialize(),
    "zero total mass",
)
invalid = replace(
    cloud,
    centers=((1.0, 2.0), (3.0, 4.0)),
    factors=cloud.factors * 2,
    factor_weights=cloud.factor_weights * 2,
    component_weights=p.UniformWeightVector((0.0, 0.75)),
    scale_draws=(0.5, 0.5),
)
reject(invalid.materialize, "zero mass component cannot receive assignment")

# Source component count request with endpoint units, independently calculated integer law.
ctx = p.JointSamplingContext(np.random.default_rng(43), mode="local")
plan = p.sample_joint_points(ctx, "points", count=3, dimension=2, vector_family="uniform")
draw = ctx.freeze().numerical.draws[0]
check(
    (
        draw.request.name,
        draw.request.low,
        draw.request.high,
        draw.request.integer,
        draw.request.scale,
    )
    == ("random_gmp_n_centers", 1, 16, "floor", "log"),
    "source count request",
)
for unit, expected in ((0.0, 1), (0.5, 4), (1.0, 15)):
    changed = replace(draw, units=(unit,), values=(expected,))
    check(changed.values == (expected,), "count endpoint floor/clamp")

# Bound decoding must reject oversized traversal/depth and stale point-weight bindings.
ctx = p.JointSamplingContext(np.random.default_rng(78))
plan = p.sample_joint_matrix(
    ctx, "kernel", rows=2, columns=3, batch=2, prior=p.JointMatrixPrior(families=("kernel",))
)
result = p.materialize_joint_matrix(plan, ctx.freeze(), rng=np.random.default_rng(8))
data = raw(result)
for field in ("decay", "log_noise_scale"):
    bad = copy.deepcopy(data)
    bad["point_cloud"]["factor_weights"][0][field] = 0
    reject(
        lambda bad=bad: p.joint_matrix_realization_from_dict(bad, decisions=ctx.freeze()),
        "stale cloud " + field,
    )
for payload in ({"oversized": [0] * 1100}, {"key" * 100: 0}, {"unknown": "x" * 129}):
    reject(
        lambda payload=payload: p.joint_matrix_realization_from_dict(
            payload, decisions=ctx.freeze(), limits=p.CausalLimits(parameters=1)
        ),
        "raw budget/string/key",
    )
deep = None
for _ in range(26):
    deep = [deep]
reject(
    lambda: p.joint_matrix_realization_from_dict({"unknown": deep}, decisions=ctx.freeze()),
    "raw depth",
)
# Numerical failure at cloud evaluation is atomic even after its weights and raw arrays exist.
rng = np.random.default_rng(14)
state = copy.deepcopy(rng.bit_generator.state)
with patch.object(
    p.KernelPointCloud, "materialize", side_effect=ValueError("nonfinite arithmetic")
):
    reject(
        lambda: p.materialize_joint_matrix(plan, ctx.freeze(), rng=rng),
        "matrix cloud late numerical failure",
    )
check(rng.bit_generator.state == state, "outer raw stream rollback after cloud failure")
finish("sharing")
