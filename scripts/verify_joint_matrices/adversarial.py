"""Malformed raw JSON, exact component gates and staged rollback via public APIs."""

import copy
import json
from dataclasses import replace
from unittest.mock import patch

import numpy as np
from support import ROOT, check, finish, raw, reject

from r0.modules import priors as p

fixtures = json.loads((ROOT / "results" / "fixtures.json").read_text())
attempts = ROOT / "results" / "mutations"
attempts.mkdir(exist_ok=True)
serial = 0


def corrupt(payload, record, path, value, label):
    global serial
    serial += 1
    bad = copy.deepcopy(payload)
    cursor = bad
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    (attempts / f"{serial:04d}.json").write_text(json.dumps({"label": label, "data": bad}))
    reject(lambda: p.joint_matrix_realization_from_dict(bad, decisions=record), label)


for fixture in fixtures:
    payload = fixture["realization"]
    record = p.joint_sampling_from_dict(fixture["decisions"])
    family = payload["plan"]["family"]
    for path, value in (
        (("unknown",), 0),
        (("version",), "unknown"),
        (("plan", "rows"), True),
        (("plan", "rows"), 2.0),
        (("plan", "columns"), 0),
        (("plan", "batch"), -1),
        (("plan", "batch"), 2**63),
        (("plan", "operation"), "matrix"),
        (("plan", "operation"), ["x" * 129]),
        (("plan", "prior", "unknown"), 0),
        (("plan", "prior", "families"), [family, family]),
        (("plan", "family"), "missing"),
        (("matrices",), {}),
        (("matrices",), payload["matrices"][:1]),
        (("matrices", 0, "unknown"), 0),
        (("matrices", 0, "version"), "x"),
        (("matrices", 0, "epsilon"), 1e-4),
        (("matrices", 0, "jitter_scale"), 1e-3),
        (("matrices", 0, "jitter"), {"0": [0, 0, 0], "1": [0, 0, 0]}),
        (("matrices", 0, "jitter"), [[0, 0], [0, 0]]),
        (("matrices", 0, "jitter", 0, 0), True),
        (("matrices", 0, "jitter", 0, 0), "0"),
        (("matrices", 0, "jitter", 0, 0), {}),
        (("matrices", 0, "jitter", 0, 0), float("inf")),
        (("matrices", 0, "source", "unknown"), 0),
    ):
        corrupt(payload, record, path, value, family + ":" + str(path))
    if family in ("importance", "factorized"):
        path = (
            ("matrices", 0, "source", "importance", 0)
            if family == "importance"
            else ("matrices", 0, "source", "diagonal")
        )
        for key, value in (
            ("decay", 0),
            ("log_noise_scale", 0),
            ("permutation", [0, 0, 0]),
            ("draws", "000"),
            ("unknown", 0),
        ):
            corrupt(payload, record, (*path, key), value, family + ":weight:" + key)
        for key, value in (
            ("operation", ["matrix", "other"]),
            ("width", 9),
            ("decays", [0, 0]),
            ("scalarization", "unknown"),
        ):
            corrupt(payload, record, ("plan", "weights", key), value, family + ":weightplan:" + key)
    if family == "transformed":
        corrupt(
            payload, record, ("matrices", 0, "source", "jitter_scale"), 1e-6, "transform jitter"
        )
        corrupt(
            payload,
            record,
            ("matrices", 0, "source", "activation", "kind"),
            "identity",
            "stale activation",
        )
        corrupt(
            payload,
            record,
            ("matrices", 0, "source", "activation", "parameters"),
            {},
            "parameter map",
        )
        corrupt(
            payload, record, ("plan", "child", "child"), payload["plan"]["child"], "recursive plan"
        )
        corrupt(
            payload,
            record,
            ("matrices", 0, "source", "child", "source"),
            payload["matrices"][0]["source"],
            "recursive construction",
        )
    kernel = (
        family == "kernel"
        or family == "transformed"
        and payload["plan"]["child"]["family"] == "kernel"
    )
    if kernel:
        for key, value in (
            ("centers", []),
            ("innovations", []),
            ("assignments", [False] * 10),
            ("scale_draws", []),
            ("unknown", 0),
            ("law", "random-covariance"),
            ("base_law", "uniform"),
            ("factors", []),
            ("component_weights", {}),
        ):
            corrupt(payload, record, ("point_cloud", key), value, "cloud:" + key)
        source_path = (
            ("matrices", 0, "source")
            if family == "kernel"
            else ("matrices", 0, "source", "child", "source")
        )
        corrupt(payload, record, (*source_path, "left_points", 0, 0), 987.0, "cloud slice mismatch")
        corrupt(payload, record, (*source_path, "distance_scale"), 0.37, "stale distance")
        corrupt(
            payload,
            record,
            (*source_path, "point_construction"),
            payload["point_cloud"],
            "duplicate cloud",
        )
    else:
        corrupt(payload, record, ("point_cloud",), {}, "irrelevant cloud")
    # A valid ledger with a surplus owned event must fail exact subtree binding.
    extra = p.JointDecision(("matrix", "extra"), "extra", "forced_identity", reason="adversarial")
    amended = replace(record, events=(*record.events, extra))
    reject(
        lambda payload=payload, amended=amended: p.joint_matrix_plan_from_dict(
            payload["plan"], decisions=amended
        ),
        "extra owned subtree",
    )
    # Swapping complete references (not just invalid indexes) tests semantic order.
    if len(record.events) > 2:
        reordered = list(record.events)
        reordered[1], reordered[2] = reordered[2], reordered[1]
        reject(
            lambda reordered=reordered, record=record, payload=payload: (
                p.joint_matrix_plan_from_dict(
                    payload["plan"], decisions=replace(record, events=tuple(reordered))
                )
            ),
            "reordered descendant events",
        )

# Independent Gaussian retained/gate arithmetic: 2*(2*2*3+2)=28, +2*3=34.
# bytes: 8*(2*2)*(6*3+2*3+4*3+4+8*(2+3)) + 128*34 = 6912.
ctx = p.JointSamplingContext(np.random.default_rng(35), mode="local")
plan = p.sample_joint_matrix(
    ctx, "g", rows=2, columns=3, batch=2, prior=p.JointMatrixPrior(families=("gaussian",))
)
record = ctx.freeze()
retained = 2 * (2 * 2 * 3 + 2)
gate = retained + 2 * 3
working = 8 * 4 * (6 * 3 + 2 * 3 + 4 * 3 + 4 + 8 * (2 + 3)) + 128 * gate
check((retained, gate, working) == (28, 34, 6912), "independent admission formula")
rng = np.random.default_rng(271)
before = copy.deepcopy(rng.bit_generator.state)
for limits in (p.CausalLimits(parameters=gate - 1), p.CausalLimits(working_bytes=working - 1)):
    # deepcopy is the first staging step after admission; blocked payload must never reach it.
    with patch("copy.deepcopy", side_effect=AssertionError("allocation/staging reached")):
        reject(
            lambda limits=limits: p.materialize_joint_matrix(plan, record, rng=rng, limits=limits),
            "before-staging one-below gate",
        )
    check(rng.bit_generator.state == before, "admission RNG rollback")
limits = p.CausalLimits(parameters=gate, working_bytes=working)
result = p.materialize_joint_matrix(plan, record, rng=rng, limits=limits)
check(result.parameter_count == retained, "exact admission retained count")
check(
    p.joint_matrix_realization_from_dict(raw(result), decisions=record, limits=limits) == result,
    "exact decoder gate",
)
for low in (p.CausalLimits(parameters=gate - 1), p.CausalLimits(working_bytes=working - 1)):
    with patch.object(
        p.GaussianWeights, "__post_init__", side_effect=AssertionError("conversion reached")
    ):
        reject(
            lambda low=low: p.joint_matrix_realization_from_dict(
                raw(result), decisions=record, limits=low
            ),
            "decode admission before construction",
        )

# Fault injection after raw draws, independently at point arithmetic and final matrix arithmetic.
for family, child in (("gaussian", "gaussian"), ("kernel", "gaussian"), ("transformed", "kernel")):
    ctx = p.JointSamplingContext(np.random.default_rng(35), mode="local")
    prior = p.JointMatrixPrior(families=(family,), simple_families=(child,))
    plan = p.sample_joint_matrix(ctx, "late", rows=2, columns=3, batch=2, prior=prior)
    record = ctx.freeze()
    rng = np.random.default_rng(271)
    state = copy.deepcopy(rng.bit_generator.state)
    with patch.object(
        p.NormalizedWeights, "materialize", side_effect=ValueError("late arithmetic")
    ):
        reject(
            lambda plan=plan, record=record, rng=rng: p.materialize_joint_matrix(
                plan, record, rng=rng
            ),
            "late matrix rollback " + family,
        )
    check(rng.bit_generator.state == state, "late RNG unchanged")
    ordinary = p.materialize_joint_matrix(plan, record, rng=rng)
    control = p.materialize_joint_matrix(plan, record, rng=np.random.default_rng(271))
    check(ordinary == control, "post-failure replay")
    if family == "kernel":
        rng = np.random.default_rng(271)
        with patch.object(
            p.KernelPointCloud, "materialize", side_effect=ValueError("late point arithmetic")
        ):
            reject(
                lambda plan=plan, record=record, rng=rng: p.materialize_joint_points(
                    plan.points, record, rng=rng
                ),
                "late point rollback",
            )
        check(rng.bit_generator.state == state, "late point RNG unchanged")
    named = np.random.default_rng(123)
    ctx = p.JointSamplingContext(named)
    initial, named_state = ctx.freeze(), copy.deepcopy(named.bit_generator.state)
    with patch.object(
        p.JointMatrixPlan, "validate", side_effect=ValueError("late frozen validation")
    ):
        reject(
            lambda ctx=ctx, prior=prior: p.sample_joint_matrix(
                ctx, "late", rows=2, columns=3, prior=prior
            ),
            "planning late freeze rollback " + family,
        )
    check(ctx.freeze() == initial and named.bit_generator.state == named_state, "named rollback")
    retry = p.sample_joint_matrix(ctx, "late", rows=2, columns=3, prior=prior)
    fresh = p.JointSamplingContext(np.random.default_rng(123))
    check(
        retry == p.sample_joint_matrix(fresh, "late", rows=2, columns=3, prior=prior),
        "path/cache rollback",
    )
    check(ctx.freeze() == fresh.freeze(), "full ledger/cache replay")
finish("adversarial")
