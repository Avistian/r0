"""Independent function plan review: public API and cached-source law oracle."""

import copy
import json
import math
from dataclasses import asdict, replace

import numpy as np
from verification import prepare

from r0.modules import priors as p

OUT, identity = prepare(__file__, p)
FAMILIES = ("mlp", "tree", "disc", "lin", "quad", "gp", "em", "prod")
MATRIX = p.JointMatrixPrior(families=("gaussian",))
COUNT = 0


def check(condition, label):
    global COUNT
    assert condition, label
    COUNT += 1


def reject(call, label):
    try:
        call()
    except (ValueError, TypeError, KeyError) as exc:
        print("REJECTED", label, type(exc).__name__, str(exc))
        check(True, label)
    else:
        raise AssertionError("ACCEPTED INVALID: " + label)


def sample(family, seed=71, mode="local", rows=9, **kwargs):
    context = p.JointSamplingContext(np.random.default_rng(seed), mode=mode)
    prior = p.JointFunctionPrior(functions=family, matrix=MATRIX, **kwargs)
    plan = p.sample_joint_function(
        context,
        "f",
        input_width=3,
        output_width=2,
        calibration_rows=rows,
        prior=prior,
    )
    return plan, context.freeze()


def source_value(draw):
    request = draw.request
    u = np.asarray(draw.units)
    if request.boundary_mass:
        u = np.clip(1.2 * u - 0.1, 0, 1)
    if request.scale == "log":
        result = np.exp(np.log(request.low) + (np.log(request.high) - np.log(request.low)) * u)
    else:
        result = request.low + (request.high - request.low) * u
    if request.integer == "round":
        result = np.rint(result)
    elif request.integer in ("floor", "source-clip-floor-v1"):
        result = np.clip(np.floor(result), request.low, request.high - 1)
    return result


def owned_events(plan, record):
    return [e for e in record.events if e.operation[: len(plan.operation)] == plan.operation]


def verify_branch(plan, record):
    # Oracle authored from cached I2func.py: no implementation request helpers.
    laws = {
        "mlp": [
            ("nn_width", 1, 128, "log", "source-clip-floor-v1"),
            ("nn_n_layers", 1, 4, "log", "source-clip-floor-v1"),
        ],
        "tree": [
            ("rt_n_trees", 1, 128, "log", "round"),
            ("rt_depth", 1, 8, "linear", "source-clip-floor-v1"),
        ],
        "disc": [
            (
                "rand_discr_n_centers",
                2,
                min(plan.calibration_rows, 256),
                "log",
                "source-clip-floor-v1",
            ),
            ("cat_disc_p", 0.5, 4, "log", "none"),
        ],
        "em": [
            ("random_em_components", 2, 17, "log", "source-clip-floor-v1"),
            ("gen_em_assignment_std_factor", 0, 1, "linear", "none"),
            ("gen_em_assignment_p_inv", 1, 4, "log", "none"),
            ("gen_em_assignment_q", 1, 2, "linear", "none"),
        ],
        "gp": [("gp_a", 2, 20, "log", "none")],
        "lin": [],
        "quad": [],
        "prod": [],
    }
    parameters = dict(plan.parameters)
    if plan.family == "gp" and not parameters["product_kernel"]:
        laws["gp"].append(("rand_gen_gp_scale", 0.5, 10, "log", "none"))
    own = [
        record.numerical.draws[e.draw_index]
        for e in record.events
        if e.operation == plan.operation and e.kind == "numerical"
    ]
    observed = [
        (d.request.name, d.request.low, d.request.high, d.request.scale, d.request.integer)
        for d in own
    ]
    check(observed == laws[plan.family], ("laws", plan.family, observed))
    check(
        all(
            d.request.shape == () and d.request.mode == "meta" and d.request.boundary_mass is False
            for d in own
        ),
        "scalar laws",
    )
    events = owned_events(plan, record)
    if plan.family == "mlp":
        expected = []
        act = 0
        if parameters["start_activation"]:
            expected.append("activation:0")
            act += 1
        for linear in range(parameters["layers"]):
            expected.append(f"linear:{linear}")
            if linear < parameters["layers"] - 1 or parameters["end_activation"]:
                expected.append(f"activation:{act}")
                act += 1
        actual = list(
            dict.fromkeys(
                e.operation[len(plan.operation)]
                for e in events
                if len(e.operation) > len(plan.operation)
            )
        )
        check(actual == expected, ("MLP stage order", actual, expected))
    if plan.family == "prod":
        actual = list(dict.fromkeys(e.operation[len(plan.operation)] for e in events[1:]))
        check(actual == ["left", "right"], "product complete child order")
        check(
            all(
                c.cheap and c.family in ("gp", "tree", "disc", "lin", "quad") for c in plan.children
            ),
            "product cheap catalog",
        )
        for child in plan.children:
            verify_branch(child, record)
    if plan.family == "gp":
        check(plan.matrices == (), "GP excludes matrix selector/wrapper")
        if parameters["product_kernel"]:
            check(plan.weights is None and len(events) == 3, "axis aligned GP request scope")
        else:
            check(
                [e.role for e in events]
                == ["family", "a", "product_kernel", "decay:0", "scale:0", "scale"],
                "radial weights before scalar scale",
            )
    plan.validate(record)


fixtures = {}
seen_cheap, seen_gp, seen_mlp = set(), set(), set()
for family in FAMILIES:
    for seed in range(12):
        plan, record = sample(family, seed=seed)
        verify_branch(plan, record)
        for draw in record.numerical.draws:
            check(
                np.allclose(source_value(draw), draw.values, rtol=2e-14, atol=1e-14),
                ("source transform oracle", draw.request.name),
            )
        payload = json.loads(json.dumps(asdict(plan)))
        check(p.joint_function_plan_from_dict(payload, decisions=record) == plan, "roundtrip")
        if seed == 0:
            fixtures[family] = {"plan": payload, "record": asdict(record)}
        if family == "prod":
            seen_cheap.update(c.family for c in plan.children)
        if family == "gp":
            seen_gp.add(dict(plan.parameters)["product_kernel"])
        if family == "mlp":
            v = dict(plan.parameters)
            seen_mlp.add((v["layers"], v["start_activation"], v["end_activation"]))
check(seen_cheap == {"gp", "tree", "disc", "lin", "quad"}, "all five cheap families visited")
check(seen_gp == {False, True}, "both GP spectra visited")
print("COVERAGE", sorted(seen_cheap), sorted(seen_gp), sorted(seen_mlp))
(OUT / "fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")

catalog = p.JointFunctionPrior(functions="gp,tabpfnv2,lin,lin,default")
slots = catalog.request().choices
check(
    tuple(catalog.family(s) for s in slots)
    == ("gp", "mlp", "tree", "disc", "lin", "lin", *FAMILIES),
    "ordered expansion",
)
check(len(set(slots)) == len(slots), "duplicate slots distinct")
for invalid in (" lin", "lin,", "default,unknown"):
    reject(
        lambda invalid=invalid: p.JointFunctionPrior(functions=invalid),
        "invalid catalog " + invalid,
    )

for mode in ("local", "meta", "global"):
    rng = np.random.default_rng(482)
    context = p.JointSamplingContext(rng, mode=mode)
    prior = p.JointFunctionPrior(functions="tree")
    for name in ("a", "b", "c"):
        p.sample_joint_function(
            context, name, input_width=3, output_width=2, calibration_rows=9, prior=prior
        )
    record = context.freeze()
    check(len(record.numerical.preferences) == 2, "one preference per shared name " + mode)
    units = [d.units for d in record.numerical.draws if d.request.name == "rt_n_trees"]
    check((len(set(units)) == 1) == (mode == "global"), "scope law " + mode)
    check(all(pref.mode == mode for pref in record.numerical.preferences), "recorded mode")
    check(
        all(d.request.name in ("rt_n_trees", "rt_depth") for d in record.numerical.draws),
        "categorical hyperprior weights isolated from outer numerical preferences",
    )
    print("SHARING", mode, units)

for rows, cap, explicit, expected in (
    (1, 1, None, 1),
    (2, 256, None, 1),
    (9, 2, None, 1),
    (9, 1, 20, 9),
):
    plan, record = sample(
        "disc", rows=rows, max_discretization_cardinality=cap, discretization_centers=explicit
    )
    check(plan.center_count == expected, "discretization special case")
    has_count = any(d.request.name == "rand_discr_n_centers" for d in record.numerical.draws)
    check(has_count == (rows > 1 and explicit is None), "discretization omitted draw")
    print("DISCRETIZATION", rows, cap, explicit, plan.center_rule, plan.center_count)

for mode in ("local", "meta", "global"):
    rng = np.random.default_rng(91)
    context = p.JointSamplingContext(rng, mode=mode)
    p.sample_joint_activation(context, "existing", selector="standard")
    before, state = context.freeze(), copy.deepcopy(rng.bit_generator.state)
    reject(
        lambda context=context: p.sample_joint_function(
            context,
            "retry",
            input_width=3,
            output_width=2,
            calibration_rows=9,
            prior=p.JointFunctionPrior(functions="disc", max_discretization_cardinality=1),
        ),
        "named failure rollback " + mode,
    )
    check(context.freeze() == before and rng.bit_generator.state == state, "full rollback")
    p.sample_joint_function(
        context,
        "retry",
        input_width=3,
        output_width=2,
        calibration_rows=9,
        prior=p.JointFunctionPrior(functions="tree"),
    )
    reject(
        lambda context=context: p.sample_joint_function(
            context,
            "retry",
            input_width=3,
            output_width=2,
            calibration_rows=9,
            prior=p.JointFunctionPrior(functions="tree"),
        ),
        "completed operation alias",
    )
    rng = np.random.default_rng(91)
    context = p.JointSamplingContext(rng, mode=mode, limits=p.PreferenceLimits(max_draws=2))
    before, state = context.freeze(), copy.deepcopy(rng.bit_generator.state)
    reject(
        lambda context=context: p.sample_joint_function(
            context,
            "deep",
            input_width=3,
            output_width=2,
            calibration_rows=9,
            prior=p.JointFunctionPrior(functions="prod", matrix=MATRIX),
        ),
        "nested child limit rollback " + mode,
    )
    check(context.freeze() == before and rng.bit_generator.state == state, "nested full rollback")

plan, record = sample("tree")
payload = json.loads(json.dumps(asdict(plan)))
for key, value in [
    ("input_width", True),
    ("parameters", [["trees", 1], ["depth", 0]]),
    ("version", "v2"),
    ("operation", ["elsewhere"]),
    ("unknown", 1),
    ("parameters", [["trees", float("nan")], ["depth", 2]]),
    ("parameters", [["trees"]]),
    ("prior", []),
    ("children", [None]),
]:
    bad = copy.deepcopy(payload)
    bad[key] = value
    reject(lambda bad=bad: p.joint_function_plan_from_dict(bad, decisions=record), "tamper " + key)
bad = copy.deepcopy(payload)
bad["parameters"][0][1] = 1 if bad["parameters"][0][1] != 1 else 2
reject(
    lambda bad=bad: p.joint_function_plan_from_dict(bad, decisions=record), "in-range changed value"
)
request = replace(record.numerical.draws[1].request, integer="floor")
draws = (*record.numerical.draws[:1], replace(record.numerical.draws[1], request=request))
preferences = (
    *record.numerical.preferences[:1],
    replace(record.numerical.preferences[1], request=request),
)
wrong = replace(record, numerical=replace(record.numerical, draws=draws, preferences=preferences))
reject(lambda: plan.validate(wrong), "self-consistent ledger with wrong integer request")
wrong = replace(
    record, events=(record.events[0], replace(record.events[1], role="other"), record.events[2])
)
reject(lambda: plan.validate(wrong), "self-consistent ledger with wrong role")
for key, value in [
    ("operation", ["x"] * 17),
    ("matrices", [None] * 1000),
    ("prior", {"functions": "x" * 10000}),
]:
    bad = copy.deepcopy(payload)
    bad[key] = value
    reject(
        lambda bad=bad: p.joint_function_plan_from_dict(bad, decisions=record), "raw limit " + key
    )
bad = copy.deepcopy(payload)
nested = []
for _ in range(30):
    nested = [nested]
bad["parameters"] = nested
reject(lambda bad=bad: p.joint_function_plan_from_dict(bad, decisions=record), "depth cap")
bad = copy.deepcopy(payload)
bad["parameters"] = np.array([[1, 2]])
reject(lambda bad=bad: p.joint_function_plan_from_dict(bad, decisions=record), "non-JSON array")


def raw_cost(value):
    if type(value) in (int, float, bool):
        return 1, 64
    if type(value) is str:
        return 0, len(value.encode("utf-8"))
    if value is None:
        return 0, 0
    leaves = list(value) + list(value.values()) if type(value) is dict else value
    costs = [raw_cost(child) for child in leaves]
    return sum(n for n, _ in costs), 32 + sum(b for _, b in costs)


for family, fixture in fixtures.items():
    plan, record = sample(family, seed=0)
    payload = fixture["plan"]
    numbers, budget = raw_cost(payload)
    limits = p.CausalLimits(parameters=numbers, working_bytes=budget)
    check(
        p.joint_function_plan_from_dict(payload, decisions=record, limits=limits) == plan,
        "exact raw bound " + family,
    )
    reject(
        lambda payload=payload, record=record, limits=limits, numbers=numbers: (
            p.joint_function_plan_from_dict(
                payload, decisions=record, limits=replace(limits, parameters=numbers - 1)
            )
        ),
        "one numeric below " + family,
    )
    reject(
        lambda payload=payload, record=record, limits=limits, budget=budget: (
            p.joint_function_plan_from_dict(
                payload, decisions=record, limits=replace(limits, working_bytes=budget - 1)
            )
        ),
        "one byte below " + family,
    )
    print("RAW_COST", family, numbers, budget)

for integer in ("floor", "source-clip-floor-v1"):
    for scale in ("linear", "log"):
        request = p.NumericalParameter("x", 2, 8, integer=integer, scale=scale)
        for u in (0.0, 0.123, 0.499, 0.877, 1.0):
            v = (
                math.exp(math.log(2) + (math.log(8) - math.log(2)) * u)
                if scale == "log"
                else 2 + 6 * u
            )
            value = int(np.clip(np.floor(v), 2, 7))
            p.NumericalDraw(request, 0, (u,), (value,))
            check(True, "legacy integer law unchanged")
request = p.NumericalParameter("degenerate", 2, 2, integer="source-clip-floor-v1", scale="log")
for unit in (0.0, 0.5, 1.0):
    p.NumericalDraw(request, 0, (unit,), (1,))
reject(lambda: p.NumericalParameter("legacy", 2, 2, integer="floor"), "legacy equal bounds")
print("PASS_COMPONENT", COUNT, "assertions; cached source only; no realization/graph claims")
