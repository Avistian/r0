"""Independent exact/minus-one resource arithmetic for every matrix branch."""

import copy
import json

import numpy as np
from support import ROOT, check, finish, matrix_count, point_count, raw, reject, vector_count

from r0.modules import priors as p


def weight_gate(plan):
    w, b = plan.width, plan.batch
    parameters = vector_count(plan) + 2 * w
    return parameters, 8 * b * (16 * w + 4) + 128 * parameters


def point_gate(plan):
    c, d, q = plan.components, plan.dimension, plan.count
    parameters = point_count(plan) + 2 * d
    working = 8 * q * (16 * d + 8 + c * (d * d + 3 * d + 3)) + 128 * parameters
    return [
        (parameters, working),
        weight_gate(plan.mixture_weights),
        weight_gate(plan.factor_weights),
    ]


def matrix_gates(plan):
    b, n, m = plan.batch, plan.rows, plan.columns
    k = max(n, m)
    parameters = matrix_count(plan) + 2 * k
    working = 8 * b * n * (6 * k + 6 * m + 4 + 8 * (n + m)) + 128 * parameters
    gates = [(parameters, working)]
    if plan.weights is not None:
        gates.append(weight_gate(plan.weights))
    if plan.points is not None:
        gates.extend(point_gate(plan.points))
    if plan.child is not None:
        gates.extend(matrix_gates(plan.child))
    return gates


summary = []
for fixture in json.loads((ROOT / "results" / "fixtures.json").read_text()):
    record = p.joint_sampling_from_dict(fixture["decisions"])
    plan = p.joint_matrix_plan_from_dict(fixture["realization"]["plan"], decisions=record)
    gates = matrix_gates(plan)
    parameter_gate = max(g[0] for g in gates)
    byte_gate = max(g[1] for g in gates)
    rng = np.random.default_rng(817)
    initial = copy.deepcopy(rng.bit_generator.state)
    for limit in (
        p.CausalLimits(parameters=parameter_gate - 1),
        p.CausalLimits(working_bytes=byte_gate - 1),
    ):
        reject(
            lambda plan=plan, record=record, rng=rng, limit=limit: p.materialize_joint_matrix(
                plan, record, rng=rng, limits=limit
            ),
            "one-below aggregate local gates",
        )
        check(rng.bit_generator.state == initial, "local child admission rollback")
    result = p.materialize_joint_matrix(
        plan,
        record,
        rng=rng,
        limits=p.CausalLimits(parameters=parameter_gate, working_bytes=byte_gate),
    )
    check(result.parameter_count == matrix_count(plan), "exact local gate")
    # Decoder admits declared whole batch; materialization retains separate child gates.
    count, working = gates[0]
    restored = p.joint_matrix_realization_from_dict(
        raw(result),
        decisions=record,
        limits=p.CausalLimits(parameters=count, working_bytes=working),
    )
    check(restored == result, "exact decoder matrix gate")
    for limit in (p.CausalLimits(parameters=count - 1), p.CausalLimits(working_bytes=working - 1)):
        reject(
            lambda result=result, record=record, limit=limit: p.joint_matrix_realization_from_dict(
                raw(result), decisions=record, limits=limit
            ),
            "one-below decoder gate",
        )
    summary.append(
        {
            "family": plan.family,
            "child": None if plan.child is None else plan.child.family,
            "retained": result.parameter_count,
            "matrix_gate": gates[0],
            "all_local_gates": gates,
            "exact_materialization_limits": [parameter_gate, byte_gate],
        }
    )
(ROOT / "results" / "gate-formulas.json").write_text(json.dumps(summary, indent=2))
finish("gates")
