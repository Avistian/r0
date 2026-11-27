"""Literal source catalogs, ordering, ownership, and policy probes (no pytest)."""

import copy
import itertools
from dataclasses import replace

import numpy as np
from common import check, raw, reject, save

from r0.modules import priors as p

FIXED = (
    "tanh",
    "leaky_relu",
    "elu",
    "identity",
    "selu",
    "silu",
    "relu",
    "softplus",
    "relu6",
    "hardtanh",
    "sign_positive",
    "heaviside_positive",
    "gaussian_bump",
    "exp",
    "interval_abs_le_1",
    "sin",
    "square",
    "abs",
    "log_clip_abs",
    "rank",
    "sigmoid",
    "round",
    "modulo",
    "softmax",
    "tied_max",
    "argsort",
    "log_sigmoid",
)
# Literal catalog is checked as an ordered sequence, not just a set.
observed = {}
for selector in ("plain", "standard"):
    found = {}
    slots = set()
    for seed in range(1800):
        context = p.JointSamplingContext(np.random.default_rng(seed), mode="local")
        plan = p.sample_joint_activation(context, "act", selector=selector)
        record = context.freeze()
        slot = record.categorical.draws[0]
        repeats = 8 if selector == "plain" else 4
        powers = ("relu_power", "signed_power") + (
            ("reciprocal_abs_power", "integer_power") if selector == "plain" else ()
        )
        check(
            f"{selector}/{seed}/selector",
            slot.request.choices == tuple(f"fixed-{i}" for i in range(repeats)) + powers,
        )
        branch = "primitive" if slot.value.startswith("fixed-") else "exponent"
        roles = ("selector", "scale", branch) if selector == "standard" else ("selector", branch)
        check(f"{selector}/{seed}/roles", tuple(e.role for e in record.events) == roles)
        if branch == "primitive":
            check(
                f"{selector}/{seed}/catalog",
                record.categorical.draws[1].request.choices == FIXED,
                record.categorical.draws[1].request.choices,
            )
        else:
            request = record.numerical.draws[-1].request
            names = {
                "relu_power": "relu_random_power_exponent",
                "signed_power": "random_power_exponent",
                "reciprocal_abs_power": "neg_random_power_exponent",
                "integer_power": "rand_int_power_exponent",
            }
            check(
                f"{selector}/{seed}/exponent",
                request.name == names[slot.value]
                and plan.activation.parameters
                == (("exponent", record.numerical.draws[-1].values[0]),),
            )
            check(
                f"{selector}/{seed}/range",
                (request.low, request.high, request.scale, request.integer)
                == (
                    (2, 6, "linear", "floor")
                    if slot.value == "integer_power"
                    else (0.1, 10.0, "log", "none")
                ),
            )
        check(
            f"{selector}/{seed}/roundtrip",
            p.joint_activation_plan_from_dict(raw(plan), decisions=record) == plan,
        )
        slots.add(slot.value)
        found.setdefault(plan.activation.kind, (seed, plan, record))
        if set(found) == set(FIXED) | set(powers) and len(slots) == repeats + len(powers):
            break
    check(selector + "/complete", set(found) == set(FIXED) | set(powers))
    observed[selector] = found
save("activation-fixtures", observed)

for mode, policy in itertools.product(("local", "meta", "global"), ("shared", "refresh")):
    context = p.JointSamplingContext(
        np.random.default_rng(77), mode=mode, categorical_policy=policy
    )
    plans = [p.sample_joint_activation(context, str(i)) for i in range(8)]
    record = context.freeze()
    for plan in plans:
        plan.validate(record)
    rr = [d for d in record.numerical.draws if d.request.name == "rr_scale"]
    check(
        f"{mode}/{policy}/rr-sharing",
        len({d.preference for d in rr}) == 1,
    )
    selectors = [d for d in record.categorical.draws if d.request.name == "act_type"]
    check(
        f"{mode}/{policy}/selector-sharing",
        len({d.preference for d in selectors})
        == (8 if policy == "refresh" and mode == "meta" else 1),
    )
    if mode == "global":
        check(f"{mode}/{policy}/global-scale", len({plan.scale for plan in plans}) == 1)
    outer_names = {d.request.name for d in record.numerical.draws}
    check(f"{mode}/{policy}/nested-separate", "simple_random_weights_decay_rate" not in outer_names)
    save(f"policy-{mode}-{policy}", {"plans": plans, "record": record})

for allow_k, allow_a, restricted, variant in itertools.product(
    (False, True),
    (False, True),
    (False, True),
    ("historical-tuple", "historical-graph", "readout-only"),
):
    prior = p.JointNumericalConverterPrior(allow_k, allow_a, restricted, variant)
    expected = ("id",) + (("kumar",) if allow_k else ()) + (("act",) if allow_a else ())
    check("converter/catalog", prior.modes == expected)
    found = set()
    for seed in range(80):
        rng = np.random.default_rng(seed)
        before = copy.deepcopy(rng.bit_generator.state)
        context = p.JointSamplingContext(rng, mode="local")
        plan = p.sample_joint_converter(
            context, "conv", channel=0, feature_group="y" if restricted else "x", prior=prior
        )
        record = context.freeze()
        found.add(plan.mode)
        check(
            "converter/roundtrip",
            p.joint_converter_plan_from_dict(raw(plan), decisions=record) == plan,
        )
        if restricted:
            check(
                "converter/forced",
                len(record.events) == 1
                and record.events[0].kind == "forced_identity"
                and rng.bit_generator.state == before,
            )
            break
        check("converter/catalog-retained", record.categorical.draws[0].request.choices == expected)
        if plan.mode == "kumar":
            check(
                "converter/ab-order",
                tuple(e.role for e in record.events) == ("mode", "a", "b")
                and tuple(d.request.name for d in record.numerical.draws)
                == ("kumaraswamy_a", "kumaraswamy_b"),
            )
            check(
                "converter/ab-contract",
                all(
                    (d.request.low, d.request.high, d.request.scale) == (0.2, 5.0, "log")
                    for d in record.numerical.draws
                ),
            )
        if plan.mode == "act":
            check(
                "converter/child",
                plan.activation.operation == ("conv", "activation")
                and record.events[1].operation == ("conv", "activation"),
            )
        if found == set(expected):
            break
    check("converter/coverage", found == ({"id"} if restricted else set(expected)))

for group in ("Y", "y-extra"):
    context = p.JointSamplingContext(np.random.default_rng(0), mode="global")
    plan = p.sample_joint_converter(
        context,
        "conv",
        channel=0,
        feature_group=group,
        prior=p.JointNumericalConverterPrior(disallow_target_warping=True),
    )
    check("converter/exact-target-" + group, context.freeze().events[0].kind == "categorical")

seed, plan, record = observed["standard"]["signed_power"]
for key, value in [
    ("operation", "act"),
    ("operation", {"act": 0}),
    ("operation", ["elsewhere"]),
    ("selector", "plain"),
    ("scale", 99.0),
    ("version", "future"),
]:
    data = raw(plan)
    data[key] = value
    reject(
        "activation/" + key + "/" + str(value),
        lambda data=data: p.joint_activation_plan_from_dict(data, decisions=record),
    )
for value in [{"exponent": 2.0}, [["exponent", 2.0], ["exponent", 3.0]], ["ab"]]:
    data = raw(plan)
    data["activation"]["parameters"] = value
    reject(
        "activation/parameters",
        lambda data=data: p.joint_activation_plan_from_dict(data, decisions=record),
    )
reject(
    "activation/value",
    lambda: replace(
        plan, activation=p.CausalActivation("signed_power", (("exponent", 0.7),))
    ).validate(record),
)
for change in (
    {"role": "wrong"},
    {"operation": ("different",)},
    {"kind": "forced_identity", "draw_index": None, "reason": "fake"},
):
    reject(
        "activation/event-adversary",
        lambda change=change: plan.validate(
            replace(record, events=(replace(record.events[0], **change), *record.events[1:]))
        ),
    )
# Different valid ledgers are adversaries for the same plan; validated record alone is insufficient.
for request in (
    p.CategoricalParameter("wrong", record.categorical.draws[0].request.choices),
    p.CategoricalParameter("act_type", ("id", "relu")),
    p.CategoricalParameter("act_type", record.categorical.draws[0].request.choices, mode="local"),
):
    context = p.JointSamplingContext(np.random.default_rng(1))
    with context.transaction("act") as child:
        child.categorical("selector", request)
    reject("activation/request-adversary", lambda context=context: plan.validate(context.freeze()))
print("choices complete")

# Reached node/plain/converter operations join one actual outer scope.
for mode in ("local", "meta", "global"):
    for seed in range(30):
        cc = p.JointSamplingContext(np.random.default_rng(seed), mode=mode)
        node = p.sample_joint_activation(cc, "node")
        matrix = p.sample_joint_activation(cc, "matrix", selector="plain")
        converter = p.sample_joint_converter(cc, "converter", channel=0, feature_group="x")
        if converter.mode == "act":
            break
    check("cross-consumer/act-reached-" + mode, converter.mode == "act")
    ledger = cc.freeze()
    scales = [d for d in ledger.numerical.draws if d.request.name == "rr_scale"]
    check(
        "cross-consumer/scale-preference-" + mode,
        len(scales) == 2 and scales[0].preference == scales[1].preference,
    )
    if mode == "global":
        check("cross-consumer/global-scale", node.scale == converter.activation.scale)
    save(
        "cross-consumer-" + mode,
        {
            "seed": seed,
            "node": node,
            "matrix_plain": matrix,
            "converter": converter,
            "ledger": ledger,
        },
    )
