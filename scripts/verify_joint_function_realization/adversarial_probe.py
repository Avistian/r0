"""Failure atomicity, preallocation admission and literal discriminating equations."""

import copy
import json
import pickle
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch
from common import OUTPUT, build, pin, plan_for, priors
from oracles import close
from probe import reject

from r0.modules.priors.lib import joint_function_realization as dispatch
from r0.modules.priors.lib.causal_limits import CausalLimits
from r0.modules.priors.lib.joint_function_admission import admit_product_function


class InjectedFailure(ValueError):
    """Deliberate post-draw verifier fault."""


def rollback(family, seed=0):
    plan, decisions = plan_for(family, seed)
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]])
    baseline = build(plan, decisions, sample)
    raw, fit = np.random.default_rng(303), np.random.default_rng(404)
    snapshot = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions))
    reached = []

    def fail_after_construction(result, record):
        assert result == baseline
        assert record == decisions
        reached.append(True)
        raise InjectedFailure("after complete retained construction and fitted draws")

    with patch.object(type(baseline), "validate", fail_after_construction):
        try:
            build(plan, decisions, sample, raw, fit)
        except InjectedFailure:
            pass
        else:
            raise AssertionError("fault was not propagated")
    assert reached
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions)) == snapshot
    assert build(plan, decisions, sample, raw, fit) == baseline
    print(f"PASS post-construction rollback and recovery {family}", flush=True)


def product_rollback():
    plan, decisions = plan_for("prod", 24)
    sample = np.array([[0.0, 1.0], [2.0, -1.0], [-2.0, 4.0], [3.0, 2.0]])
    raw, fit = np.random.default_rng(303), np.random.default_rng(404)
    before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions))
    original = dispatch.materialize_joint_function
    completed = []

    def child_then_fail(*args, **kwargs):
        result = original(*args, **kwargs)
        completed.append(result.plan.operation[-1])
        if len(completed) == 2:
            raise InjectedFailure("after successful left and right constructions")
        return result

    with patch.object(dispatch, "materialize_joint_function", child_then_fail):
        try:
            build(plan, decisions, sample, raw, fit)
        except InjectedFailure:
            pass
        else:
            raise AssertionError("product fault not propagated")
    assert completed == ["left", "right"]
    assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state, decisions)) == before
    print("PASS product rollback after successful left and complete right", flush=True)


def admission():
    sample = np.zeros((4, 2))
    for family in ("lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"):
        plan, decisions = plan_for(family)
        raw, fit = np.random.default_rng(303), np.random.default_rng(404)
        before = pickle.dumps((raw.bit_generator.state, fit.bit_generator.state))
        with (
            patch("numpy.array", side_effect=AssertionError("array before admission")),
            patch("numpy.asarray", side_effect=AssertionError("asarray before admission")),
            patch("copy.deepcopy", side_effect=AssertionError("copy before admission")),
        ):
            try:
                priors.materialize_joint_function(
                    plan,
                    decisions,
                    rng=raw,
                    fit_rng=fit,
                    calibration=sample,
                    calibration_id="bounded",
                    limits=CausalLimits(parameters=1),
                )
            except ValueError as exc:
                if "limit" not in str(exc):
                    raise AssertionError("unexpected admission reason") from exc
            else:
                raise AssertionError("oversized construction admitted")
        assert pickle.dumps((raw.bit_generator.state, fit.bit_generator.state)) == before
        result = build(plan, decisions, sample)
        payload = json.loads(json.dumps(asdict(result)))
        with patch("numpy.array", side_effect=AssertionError("decoder array before admission")):
            try:
                priors.joint_function_realization_from_dict(
                    payload, decisions=decisions, limits=CausalLimits(parameters=1)
                )
            except ValueError as exc:
                if "limit" not in str(exc):
                    raise AssertionError("unexpected admission reason") from exc
            else:
                raise AssertionError("oversized JSON admitted")
    plan, decisions = plan_for("prod", 24)
    report = admit_product_function(plan, decisions, CausalLimits())
    cap = CausalLimits(parameters=report["parameters"] - 1)
    with patch.object(dispatch, "materialize_joint_function", side_effect=AssertionError("child")):
        try:
            priors.materialize_joint_function(
                plan,
                decisions,
                rng=np.random.default_rng(1),
                fit_rng=np.random.default_rng(2),
                calibration=sample,
                calibration_id="bounded",
                limits=cap,
            )
        except ValueError as exc:
            if "aggregate product" not in str(exc):
                raise AssertionError("unexpected aggregate reason") from exc
        else:
            raise AssertionError("aggregate product not admitted first")
    print("PASS preallocation limits all eight families and aggregate-before-child", flush=True)


def literals():
    def tensor(x):
        return torch.tensor(x, dtype=torch.float64)

    linear = priors.FrozenLinearFunction(((1.0, 0.0), (0.0, 1.0)))
    em = priors.FrozenEMFunction(((0.0, 0.0), (0.0, 0.0)), (1.0, 2.0), linear, 2.0, 1.0)
    close(priors.evaluate_frozen_function(em, tensor([[0.0, 0.0]])).numpy(), [[2 / 3, 1 / 3]])
    quadratic = priors.FrozenQuadraticFunction(1, (0,), (((1.0, 3.0), (5.0, 7.0)),))
    close(priors.evaluate_frozen_function(quadratic, tensor([[2.0]])).numpy(), [[27.0]])
    tree = priors.FrozenTreeFunction(
        2,
        ((0, 1), (0, 1)),
        ((1.0, 2.0), (1.0, 2.0)),
        (((10.0,), (20.0,), (30.0,), (40.0,)), ((2.0,), (4.0,), (6.0,), (8.0,))),
    )
    close(priors.evaluate_frozen_function(tree, tensor([[1.0, 3.0]])).numpy(), [[18.0]])
    disc = priors.FrozenDiscretizationFunction(
        ((-1.0,), (1.0,)), priors.FrozenLinearFunction(((2.0,),)), 0.5
    )
    close(priors.evaluate_frozen_function(disc, tensor([[0.0]])).numpy(), [[-2.0]])
    plan, decisions = plan_for("gp", 0)
    result = build(plan, decisions, np.zeros((4, 2)))
    if dict(plan.parameters)["product_kernel"]:
        for seed in range(1, 20):
            plan, decisions = plan_for("gp", seed)
            if not dict(plan.parameters)["product_kernel"]:
                result = build(plan, decisions, np.zeros((4, 2)))
                break
    payload = json.loads(json.dumps(asdict(result)))
    zero = copy.deepcopy(payload)
    for row in zero["directions"]:
        row[0] = 0.0
    reject(zero, decisions, "zero GP direction")
    (OUTPUT / "zero-direction-rejected.json").write_text(json.dumps(zero, separators=(",", ":")))
    print("PASS literal EM constant, quadratic, tree bits/equality, disc tie, zero GP", flush=True)


def main():
    pin()
    for family in ("lin", "quad", "gp", "tree", "disc", "em", "mlp", "prod"):
        rollback(family)
    product_rollback()
    admission()
    literals()


if __name__ == "__main__":
    main()
