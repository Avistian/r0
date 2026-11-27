"""Independent weight order, arithmetic, admission and transaction oracles."""

import copy
import itertools
import math
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
from common import check, raw, reject, save

from r0.modules import priors as p

for mode, policy in itertools.product(
    ("local", "meta", "global"), ("scalarized-weight-batches-v1", "source-weight-batches-v1")
):
    rng = np.random.default_rng(53)
    context = p.JointSamplingContext(
        rng,
        mode=mode,
        global_policy="strict-contract-v1" if policy.startswith("source") else "scalar-quantile-v1",
    )
    plan = p.sample_joint_weights(context, "weights", width=4, batch=3, scalarization=policy)
    record = context.freeze()
    save(mode + "-" + policy, {"plan": plan, "record": record})
    expected_roles = (
        ("decay", "scale")
        if policy.startswith("source")
        else ("decay:0", "decay:1", "decay:2", "scale:0", "scale:1", "scale:2")
    )
    check("weight/ordered", tuple(e.role for e in record.events) == expected_roles)
    check(
        "weight/metadata-count",
        plan.parameter_count == 8 and plan.realization_parameter_count == 30,
    )
    check("weight/roundtrip", p.joint_weight_plan_from_dict(raw(plan), decisions=record) == plan)
    if mode == "global" and policy.startswith("scalarized"):
        check("weight/global-ties", len(set(plan.decays)) == len(set(plan.scales)) == 1)
    control = np.random.default_rng(410)
    gaussian = control.normal(size=(3, 4))
    permutations = [control.permutation(4) for _ in range(3)]
    realize_rng = np.random.default_rng(410)
    # Literal accounting: 3*(4 draws+4 indices+2 hyperparameters)+2*4 geometry=38.
    # Work: 8*3*(6*4+2*4+4*4+4+4*4)+128*38=6496 bytes.
    limits = p.CausalLimits(parameters=38, working_bytes=6496)
    vectors = p.materialize_joint_weights(plan, record, rng=realize_rng, limits=limits)
    check("weight/raw-rng-order", realize_rng.bit_generator.state == control.bit_generator.state)
    for i, vector in enumerate(vectors):
        logits = (-plan.decays[i] * np.log(np.arange(1, 5)) + plan.scales[i] * gaussian[i])[
            permutations[i]
        ]
        stable = np.exp(logits - logits.max())
        expected = stable * math.sqrt(4 / np.sum(stable**2))
        check(
            "weight/arithmetic",
            np.allclose(vector.values(), expected, rtol=2e-14, atol=1e-15),
            {"expected": expected, "actual": vector.values()},
        )
        check(
            "weight/raw-draws",
            vector.draws == tuple(gaussian[i]) and vector.permutation == tuple(permutations[i]),
        )
    for key, limit in [("parameters", 37), ("working_bytes", 6495)]:
        snapshot = copy.deepcopy(realize_rng.bit_generator.state)
        reject(
            "weight/one-below-" + key,
            lambda key=key, limit=limit, args=(plan, record, realize_rng, limits): (
                p.materialize_joint_weights(
                    args[0], args[1], rng=args[2], limits=replace(args[3], **{key: limit})
                )
            ),
        )
        check("weight/admission-rollback", realize_rng.bit_generator.state == snapshot)
    # Replay with exact observed ledger boundary, then one-below each bound.
    counts = (record.draw_count, record.value_count, record.parameter_count)
    for key in ("exact", "max_draws", "max_values", "max_parameters"):
        cap = dict(zip(("max_draws", "max_values", "max_parameters"), counts, strict=True))
        if key != "exact":
            cap[key] -= 1
        rr = np.random.default_rng(53)
        cc = p.JointSamplingContext(
            rr,
            mode=mode,
            global_policy=context.freeze().global_policy,
            limits=p.PreferenceLimits(**cap),
        )
        before = (asdict(cc.freeze()), copy.deepcopy(rr.bit_generator.state))

        def sample(cc=cc, policy=policy):
            return p.sample_joint_weights(cc, "weights", width=4, batch=3, scalarization=policy)

        if key == "exact":
            check("weight/choice-exact", sample() == plan)
        else:
            reject("weight/choice-one-below-" + key, sample)
            check("weight/choice-rollback", (asdict(cc.freeze()), rr.bit_generator.state) == before)
    for field, value in [
        ("width", True),
        ("batch", True),
        ("batch", 2**62),
        ("decays", []),
        ("scales", "aaa"),
        ("operation", {"weights": 1}),
        ("scalarization", "bad"),
        ("family", "uniform"),
    ]:
        data = raw(plan)
        data[field] = value
        reject(
            "weight/decode-" + field,
            lambda data=data, record=record: p.joint_weight_plan_from_dict(data, decisions=record),
        )

for mode in ("local", "meta", "global"):
    rng = np.random.default_rng(31)
    cc = p.JointSamplingContext(rng, mode=mode)
    before = copy.deepcopy(rng.bit_generator.state)
    uniform = p.sample_joint_weights(cc, "uniform", width=2, batch=3, family="uniform")
    check("uniform/event-free", cc.freeze().events == () and rng.bit_generator.state == before)
    rr = np.random.default_rng(31)
    expected = np.random.default_rng(31).random((3, 2))
    vectors = p.materialize_joint_weights(uniform, cc.freeze(), rng=rr)
    check("uniform/unchanged", np.array_equal([v.values() for v in vectors], expected))
    check("uniform/zero-valid", p.UniformWeightVector((0.0, 0.0)).values() == (0.0, 0.0))

for policy in ("scalarized-weight-batches-v1", "source-weight-batches-v1"):
    cc = p.JointSamplingContext(np.random.default_rng(72), mode="meta")
    for index, (width, batch) in enumerate(((2, 1), (6, 3), (3, 2))):
        p.sample_joint_weights(cc, str(index), width=width, batch=batch, scalarization=policy)
    record = cc.freeze()
    check("meta/variable-geometry-shares", len(record.numerical.preferences) == 2)

cc = p.JointSamplingContext(np.random.default_rng(14), mode="global")
a = p.sample_joint_weights(cc, "a", width=2, batch=2)
b = p.sample_joint_weights(cc, "b", width=7, batch=3)
u_a = math.log(a.decays[0] / (0.1 / math.log(3))) / math.log(6 / (0.1 / math.log(3)))
u_b = math.log(b.decays[0] / (0.1 / math.log(8))) / math.log(6 / (0.1 / math.log(8)))
check("global/quantile-remap", abs(u_a - u_b) < 1e-14 and a.scales[0] == b.scales[0])
for changed in ({"width": 5, "batch": 3}, {"width": 4, "batch": 2}):
    rng = np.random.default_rng(14)
    cc = p.JointSamplingContext(rng, mode="global", global_policy="strict-contract-v1")
    p.sample_joint_weights(cc, "first", width=4, batch=3, scalarization="source-weight-batches-v1")
    snapshot = (asdict(cc.freeze()), copy.deepcopy(rng.bit_generator.state))
    reject(
        "global/strict-incompatible",
        lambda changed=changed, cc=cc: p.sample_joint_weights(
            cc, "second", scalarization="source-weight-batches-v1", **changed
        ),
    )
    check("global/strict-rollback", snapshot == (asdict(cc.freeze()), rng.bit_generator.state))
    p.sample_joint_weights(cc, "second", width=4, batch=3, scalarization="source-weight-batches-v1")
    check("global/path-reusable-after-failure", len(cc.freeze().events) == 4)
# A late failure after nested successful plans rolls back their decisions and stream.
rng = np.random.default_rng(8)
cc = p.JointSamplingContext(rng)
snapshot = (asdict(cc.freeze()), copy.deepcopy(rng.bit_generator.state))


def late():
    with cc.transaction("outer") as outer:
        p.sample_joint_activation(outer, "act")
        p.sample_joint_weights(outer, "weights", width=3, batch=2)
        raise ValueError("independent late transaction failure")


reject("transaction/late-failure", late)
check("transaction/full-rollback", snapshot == (asdict(cc.freeze()), rng.bit_generator.state))
check(
    "literal/weight-vector",
    np.allclose(
        p.PolynomialWeightVector(
            (0.0, 0.0), decay=1.0, log_noise_scale=0.0, permutation=(1, 0)
        ).values(),
        (math.sqrt(2 / 5), 2 * math.sqrt(2 / 5)),
    ),
)
print("weights complete")

# Inject a late record-construction failure after a completed row and a second permutation.
cc = p.JointSamplingContext(np.random.default_rng(27))
plan = p.sample_joint_weights(cc, "atomic", width=4, batch=3)
ledger = cc.freeze()
rng = np.random.default_rng(91)
before = (asdict(ledger), copy.deepcopy(rng.bit_generator.state))
original = p.PolynomialWeightVector.__post_init__
constructed = []


def fail_second(vector):
    constructed.append(vector.draws)
    if len(constructed) == 2:
        raise ValueError("injected late vector validation failure")
    original(vector)


with patch.object(p.PolynomialWeightVector, "__post_init__", fail_second):
    reject(
        "materialization/late-injected-failure",
        lambda: p.materialize_joint_weights(plan, ledger, rng=rng),
    )
check("materialization/reached-second-row", len(constructed) == 2)
check("materialization/late-rollback", before == (asdict(ledger), rng.bit_generator.state))
